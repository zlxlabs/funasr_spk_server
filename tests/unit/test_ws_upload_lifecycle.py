"""上传生命周期回归：缓存命中、分片引擎校验与共享文件删除保护。"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.websocket_handler import WebSocketHandler
from src.core.database import DatabaseManager
from src.core.task_manager import TaskManager
from src.models.schemas import FileUploadRequest, TaskStatus, TranscriptionResult, TranscriptionSegment


@pytest.fixture
def real_task_manager(monkeypatch):
    """用真实 TaskManager 隔离上传任务状态，不启动转写 worker。"""
    import src.core.task_manager as task_manager_module

    manager = TaskManager()
    monkeypatch.setattr(task_manager_module, "task_manager", manager)
    yield manager
    manager.executor.shutdown(wait=True)


@pytest.fixture
async def real_cache_db(tmp_path, monkeypatch):
    """单测使用独立 SQLite 缓存，缓存读写保持真实实现。"""
    import src.core.database as database_module

    database = DatabaseManager(db_path=str(tmp_path / "upload-cache.db"))
    await database.init_db()
    monkeypatch.setattr(database_module, "db_manager", database)
    return database


@pytest.fixture
def websocket():
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.remote_address = ("127.0.0.1", 1234)
    return ws


def sent_messages(websocket):
    return [json.loads(call.args[0]) for call in websocket.send.await_args_list]


def cacheable_result(file_hash: str, file_name: str = "audio.wav") -> TranscriptionResult:
    return TranscriptionResult(
        task_id="cache-source",
        file_name=file_name,
        file_hash=file_hash,
        duration=1.0,
        segments=[TranscriptionSegment(start_time=0.0, end_time=1.0, text="缓存结果", speaker="Speaker1")],
        speakers=["Speaker1"],
        processing_time=0.1,
    )


@pytest.mark.asyncio
async def test_single_upload_cache_hit_does_not_create_a_task(
    real_task_manager, real_cache_db, websocket
):
    """命中真实 SQLite 缓存时直接完成，TaskManager 不应留下 PENDING 任务。"""
    from src.core.config import config

    file_hash = "cache-hit-upload"
    engine = config.transcription.default_engine
    await real_cache_db.save_result(cacheable_result(file_hash), raw_result=None, engine=engine)
    handler = WebSocketHandler()

    await handler._handle_upload_request(websocket, "conn-cache", {
        "file_name": "audio.wav",
        "file_size": 128,
        "file_hash": file_hash,
        "force_refresh": False,
    })

    messages = sent_messages(websocket)
    assert [message["type"] for message in messages] == ["task_complete"]
    assert real_task_manager.tasks == {}


@pytest.mark.asyncio
async def test_chunked_upload_rejects_mismatched_engine_before_accepting_chunks(
    real_task_manager, websocket, monkeypatch
):
    """不匹配引擎必须在 upload_request 阶段拒绝，不创建会话也不接收分片。"""
    from loguru import logger
    from src.core.config import config

    monkeypatch.setattr(config.transcription, "default_engine", "funasr")
    log_output = io.StringIO()
    sink_id = logger.add(log_output.write, format="{message}")
    handler = WebSocketHandler()
    task_id = "e5d2c9e0-7019-4f0f-8aab-e2a9a64bb8a1"
    try:
        await handler._handle_message(websocket, "conn-chunk-engine", {
            "type": "upload_request",
            "data": {
                "file_name": "audio.wav",
                "file_size": 128,
                "file_hash": "chunk-engine-reject",
                "engine": "qwen3",
                "upload_mode": "chunked",
                "total_chunks": 2,
            },
        })
        await handler._handle_message(websocket, "conn-chunk-engine", {
            "type": "upload_chunk",
            "data": {"task_id": task_id, "chunk_index": 0, "chunk_data": "YQ=="},
        })
    finally:
        logger.remove(sink_id)

    messages = sent_messages(websocket)
    assert messages[0]["type"] == "error"
    assert messages[0]["data"]["error"] == "upload_error"
    assert messages[1]["data"]["error"] == "session_not_found"
    assert not any(message["type"] == "chunk_received" for message in messages)
    assert handler.upload_sessions == {}
    assert real_task_manager.tasks == {}
    assert "UPLOAD-ENGINE-REJECTED" in log_output.getvalue()


@pytest.mark.asyncio
async def test_chunked_srt_cache_key_includes_output_format(
    real_task_manager, real_cache_db, websocket, monkeypatch
):
    """分片 SRT 缓存查询须以 SRT 折维，避免误查 JSON word-align 缓存 tag。"""
    from src.core.config import config

    monkeypatch.setattr(config.transcription, "default_engine", "qwen3")
    content = b"chunked cache payload"
    file_hash = hashlib.md5(content).hexdigest()
    await real_cache_db.save_result(
        cacheable_result(file_hash), raw_result=None, engine="qwen3"
    )
    handler = WebSocketHandler()

    await handler._handle_message(websocket, "conn-chunk-cache", {
        "type": "upload_request",
        "data": {
            "file_name": "audio.wav",
            "file_size": len(content),
            "file_hash": file_hash,
            "upload_mode": "chunked",
            "total_chunks": 1,
            "chunk_size": len(content),
            "output_format": "srt",
            "word_align": True,
        },
    })
    ready = sent_messages(websocket)[0]
    await handler._handle_message(websocket, "conn-chunk-cache", {
        "type": "upload_chunk",
        "data": {
            "task_id": ready["data"]["task_id"],
            "chunk_index": 0,
            "chunk_data": base64.b64encode(content).decode("ascii"),
        },
    })

    messages = sent_messages(websocket)
    assert any(message["type"] == "task_complete" for message in messages)
    assert real_task_manager.tasks == {}
    assert handler.upload_sessions == {}


@pytest.mark.asyncio
async def test_expired_chunk_sweep_preserves_final_file_used_by_live_task(
    real_task_manager, websocket, monkeypatch, tmp_path
):
    """无 task 对象的过期 session sweep 也须保护 live 同 hash 文件。"""
    from src.core.config import config
    from src.utils.file_utils import save_uploaded_file

    monkeypatch.setattr(config.server, "upload_dir", str(tmp_path / "uploads"))
    content = b"live task owns this content-addressed file"
    file_hash = hashlib.md5(content).hexdigest()
    shared_path, _ = await save_uploaded_file(content, "shared.wav")
    live_task = await real_task_manager.create_task(FileUploadRequest(
        file_name="live.wav",
        file_size=len(content),
        file_hash=file_hash,
    ))
    live_task.file_path = shared_path

    stale_temp = tmp_path / "stale-chunk.tmp"
    stale_temp.write_bytes(b"stale")
    handler = WebSocketHandler()
    stale_id = "stale-upload"
    handler.upload_sessions[stale_id] = {
        "file_hash": file_hash,
        "temp_file_path": str(stale_temp),
        "finalized_file_path": shared_path,
        "created_at": time.time() - config.transcription.upload_session_ttl_seconds - 1,
    }
    await handler._handle_message(websocket, "conn-sweep", {
        "type": "upload_request",
        "data": {
            "file_name": "new.wav",
            "file_size": 1,
            "file_hash": "new-session-hash",
            "upload_mode": "chunked",
            "total_chunks": 1,
        },
    })

    assert os.path.exists(shared_path)
    assert not stale_temp.exists()
    assert stale_id not in handler.upload_sessions
    assert real_task_manager.get_task(live_task.task_id) is live_task
    new_session_id = next(iter(handler.upload_sessions))
    await handler._cleanup_upload_session(new_session_id)


@pytest.mark.asyncio
async def test_queue_full_rejection_preserves_file_used_by_same_hash_pending_task(
    real_task_manager, websocket, monkeypatch, tmp_path
):
    """队列满拒绝同 hash 上传时，不得删掉另一 PENDING 任务共享的内容寻址文件。"""
    from src.core.config import config
    from src.utils.file_utils import save_uploaded_file

    monkeypatch.setattr(config.server, "upload_dir", str(tmp_path / "uploads"))
    content = b"identical shared upload bytes"
    file_hash = hashlib.md5(content).hexdigest()
    shared_path, _ = await save_uploaded_file(content, "shared.wav")
    existing = await real_task_manager.create_task(FileUploadRequest(
        file_name="existing.wav",
        file_size=len(content),
        file_hash=file_hash,
        force_refresh=True,
    ))
    existing.file_path = shared_path
    assert existing.status is TaskStatus.PENDING

    queue_capacity = real_task_manager.task_queue.maxsize
    assert queue_capacity > 0
    for index in range(queue_capacity):
        real_task_manager.task_queue.put_nowait(f"queued-{index}")

    handler = WebSocketHandler()
    await handler._handle_upload_request(websocket, "conn-full", {
        "file_name": "new-copy.wav",
        "file_size": len(content),
        "file_hash": file_hash,
        "force_refresh": True,
    })
    ready = next(message for message in sent_messages(websocket) if message["type"] == "upload_ready")
    await handler._handle_upload_data(websocket, "conn-full", {
        "task_id": ready["data"]["task_id"],
        "file_data": base64.b64encode(content).decode("ascii"),
    })

    messages = sent_messages(websocket)
    assert any(message["type"] == "queue_full" for message in messages)
    assert os.path.exists(shared_path)
    assert real_task_manager.get_task(existing.task_id) is existing
    assert existing.status is TaskStatus.PENDING
    assert not any(task.file_hash == file_hash and task.task_id != existing.task_id
                   for task in real_task_manager.tasks.values())
