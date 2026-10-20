"""WebSocket 发送、分片边界、finalize 与上传会话 TTL 回归测试。"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from src.api.websocket_handler import WebSocketHandler


@pytest.fixture
def websocket():
    """模拟 websocket 的网络 I/O，保留真实 handler 路径。"""
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.remote_address = ("127.0.0.1", 1234)
    return ws


def sent_messages(websocket):
    """解析 websocket 实际收到的序列化消息。"""
    return [json.loads(call.args[0]) for call in websocket.send.await_args_list]


async def create_chunk_session(handler, websocket, tmp_path, monkeypatch, *, total_chunks=2, chunk_size=4):
    """从消息分发入口在隔离上传目录创建真实分片上传会话。"""
    from src.core.config import config

    monkeypatch.setattr(config.server, "upload_dir", str(tmp_path))
    await handler._handle_message(websocket, "conn-upload", {
        "type": "upload_request",
        "data": {
            "file_name": "audio.wav",
            "file_size": total_chunks * chunk_size,
            "file_hash": "f" * 32,
            "force_refresh": True,
            "upload_mode": "chunked",
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
        },
    })
    ready = next(message for message in sent_messages(websocket) if message["type"] == "upload_ready")
    return ready["data"]["task_id"]


@pytest.mark.asyncio
async def test_send_message_times_out_logs_and_cleans_associated_connection(
    websocket, monkeypatch
):
    """卡住的 send 必须限时失败、关闭连接并解除任务关联。"""
    import src.api.websocket_handler as websocket_handler_module

    monkeypatch.setattr(websocket_handler_module, "WS_SEND_TIMEOUT_SECONDS", 0.01, raising=False)
    handler = WebSocketHandler()
    handler.connections["conn-slow"] = websocket
    handler.task_connections["task-slow"] = {"conn-slow"}

    async def blocked_send(_message):
        await asyncio.Event().wait()

    websocket.send = AsyncMock(side_effect=blocked_send)
    log_output = []
    sink_id = logger.add(log_output.append, format="{message}")
    started = time.monotonic()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                handler._send_message(websocket, "task_progress", {"task_id": "task-slow"}),
                timeout=0.2,
            )
    finally:
        logger.remove(sink_id)

    assert time.monotonic() - started < 0.2
    websocket.close.assert_awaited_once()
    assert "conn-slow" not in handler.connections
    assert "task-slow" not in handler.task_connections
    assert any("WS-SEND-TIMEOUT" in line for line in log_output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunk_index", "payload"),
    [
        (-1, b"abcd"),
        (2, b"abcd"),
        ("0", b"abcd"),
        (True, b"abcd"),
        (0, b"abc"),
        (1, b"abcde"),
    ],
)
async def test_invalid_chunk_index_or_length_is_not_written_or_counted(
    websocket, tmp_path, monkeypatch, chunk_index, payload
):
    """边界错误的分片通过消息入口被拒，不能扩展临时文件或推进计数。"""
    handler = WebSocketHandler()
    task_id = await create_chunk_session(handler, websocket, tmp_path, monkeypatch)
    session = handler.upload_sessions[task_id]

    await handler._handle_message(websocket, "conn-upload", {
        "type": "upload_chunk",
        "data": {
            "task_id": task_id,
            "chunk_index": chunk_index,
            "chunk_data": base64.b64encode(payload).decode("ascii"),
        },
    })

    messages = sent_messages(websocket)
    assert any(
        message["type"] == "error" and message["data"]["error"] == "upload_error"
        for message in messages
    )
    assert os.path.getsize(session["temp_file_path"]) == 0
    assert session["received_chunks"] == 0
    assert session["chunks_received"] == set()
    await handler._cleanup_upload_session(task_id)


@pytest.mark.asyncio
async def test_valid_chunk_refreshes_session_activity(websocket, tmp_path, monkeypatch):
    """被接受的分片刷新最近活动时间，不按初始创建时间清扫。"""
    handler = WebSocketHandler()
    task_id = await create_chunk_session(handler, websocket, tmp_path, monkeypatch)
    session = handler.upload_sessions[task_id]
    session["last_activity_at"] = time.time() - 100

    await handler._handle_message(websocket, "conn-upload", {
        "type": "upload_chunk",
        "data": {
            "task_id": task_id,
            "chunk_index": 0,
            "chunk_data": base64.b64encode(b"abcd").decode("ascii"),
        },
    })

    assert session["last_activity_at"] > time.time() - 2
    assert session["received_chunks"] == 1
    await handler._cleanup_upload_session(task_id)


@pytest.mark.asyncio
async def test_finalize_hash_runs_off_event_loop_and_mismatch_cleans_session(
    websocket, tmp_path, monkeypatch
):
    """finalize 通过消息入口异步算 MD5；hash 不符会删除 session 与临时文件。"""
    handler = WebSocketHandler()
    content = b"real file content"
    temp_path = tmp_path / "chunk-upload.tmp"
    temp_path.write_bytes(content)
    task_id = "task-finalize-hash"
    handler.upload_sessions[task_id] = {
        "task_id": task_id,
        "file_name": "audio.wav",
        "file_size": len(content),
        "file_hash": "0" * 32,
        "chunk_size": len(content),
        "total_chunks": 1,
        "received_chunks": 1,
        "temp_file_path": str(temp_path),
        "chunks_received": {0},
        "output_format": "json",
        "force_refresh": True,
        "connection_id": "conn-upload",
        "engine": "funasr",
        "language": None,
        "diarize": True,
        "word_align": None,
        "terms": [],
        "state": "ready",
        "created_at": time.time(),
        "last_activity_at": time.time() - 10,
        "finalized_file_path": None,
    }
    loop_thread = threading.get_ident()
    worker_threads = []
    activity_at_hash = []
    original_hash = handler._calculate_file_hash

    def record_hash_thread(path):
        worker_threads.append(threading.get_ident())
        activity_at_hash.append(handler.upload_sessions[task_id]["last_activity_at"])
        return original_hash(path)

    monkeypatch.setattr(handler, "_calculate_file_hash", record_hash_thread)
    await handler._handle_message(websocket, "conn-upload", {
        "type": "finalize_upload", "data": {"task_id": task_id},
    })

    assert worker_threads and worker_threads[0] != loop_thread
    assert activity_at_hash[0] > time.time() - 2
    assert task_id not in handler.upload_sessions
    assert not temp_path.exists()
    assert any(message["data"].get("error") == "file_hash_mismatch" for message in sent_messages(websocket))


@pytest.mark.asyncio
async def test_queue_full_receipt_refreshes_session_activity(websocket, tmp_path, monkeypatch):
    """队列满回执刷新活动时间，保留会话供客户端 finalize 重试。"""
    from types import SimpleNamespace

    import src.api.websocket_handler as websocket_handler_module
    import src.core.task_manager as task_manager_module
    from src.core.config import config
    from src.core.task_manager import TaskManager

    monkeypatch.setattr(config.server, "upload_dir", str(tmp_path))
    content = b"full queue upload"
    temp_path = tmp_path / "queue-full.tmp"
    temp_path.write_bytes(content)
    task_id = "task-queue-full"
    session = {
        "task_id": task_id,
        "file_name": "audio.wav",
        "file_size": len(content),
        "file_hash": hashlib.md5(content).hexdigest(),
        "chunk_size": len(content),
        "total_chunks": 1,
        "received_chunks": 1,
        "temp_file_path": str(temp_path),
        "chunks_received": {0},
        "output_format": "json",
        "force_refresh": True,
        "connection_id": "conn-upload",
        "engine": "funasr",
        "language": None,
        "diarize": True,
        "word_align": None,
        "terms": [],
        "state": "ready",
        "created_at": time.time() - 100,
        "last_activity_at": time.time() - 100,
        "finalized_file_path": None,
    }
    handler = WebSocketHandler()
    handler.upload_sessions[task_id] = session
    final_path = tmp_path / f"{session['file_hash']}.wav"
    task_manager = TaskManager()
    queue_capacity = task_manager.task_queue.maxsize
    assert queue_capacity > 0
    for index in range(queue_capacity):
        task_manager.task_queue.put_nowait(f"full-{index}")
    monkeypatch.setattr(task_manager_module, "task_manager", task_manager)
    ticks = iter([1000.0, 1001.0])
    monkeypatch.setattr(
        websocket_handler_module, "time", SimpleNamespace(time=lambda: next(ticks))
    )

    import builtins

    original_open = builtins.open

    class BoundedReader:
        """拒绝读完整文件，锁定 finalize 只允许分块读取。"""

        def __init__(self, file_object):
            self.file_object = file_object

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.file_object.close()

        def read(self, size=-1):
            if size < 0:
                raise AssertionError("finalize 不得无界读取整个上传文件")
            return self.file_object.read(size)

        def __getattr__(self, name):
            return getattr(self.file_object, name)

    def guarded_open(path, mode="r", *args, **kwargs):
        file_object = original_open(path, mode, *args, **kwargs)
        if os.fspath(path) == str(temp_path) and "r" in mode:
            return BoundedReader(file_object)
        return file_object

    monkeypatch.setattr(builtins, "open", guarded_open)

    await handler._handle_message(websocket, "conn-upload", {
        "type": "finalize_upload", "data": {"task_id": task_id},
    })

    assert session["state"] == "ready"
    assert session["last_activity_at"] == 1001.0
    assert final_path.exists()
    assert final_path.stat().st_size == len(content)
    assert temp_path.exists()
    assert any(message["type"] == "queue_full" for message in sent_messages(websocket))
    final_path.unlink()
    temp_path.unlink()
    handler.upload_sessions.pop(task_id)
    task_manager.executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_sweep_uses_recent_activity_and_skips_active_finalize(
    websocket, tmp_path
):
    """创建时间超 TTL 但有活动的会话保留；无活动清理，finalize 中跳过。"""
    from src.core.config import config

    handler = WebSocketHandler()
    ttl = config.transcription.upload_session_ttl_seconds
    for task_id, activity_age, state in (
        ("active-upload", 1, "uploading"),
        ("stale-upload", ttl + 10, "ready"),
        ("finalizing-upload", ttl + 10, "finalizing"),
    ):
        path = tmp_path / f"{task_id}.bin"
        path.write_bytes(b"data")
        handler.upload_sessions[task_id] = {
            "task_id": task_id,
            "file_hash": task_id,
            "temp_file_path": str(path),
            "created_at": time.time() - ttl - 100,
            "last_activity_at": time.time() - activity_age,
            "state": state,
        }

    handler._sweep_upload_sessions()

    assert "active-upload" in handler.upload_sessions
    assert (tmp_path / "active-upload.bin").exists()
    assert "stale-upload" not in handler.upload_sessions
    assert not (tmp_path / "stale-upload.bin").exists()
    assert "finalizing-upload" in handler.upload_sessions
    assert (tmp_path / "finalizing-upload.bin").exists()
