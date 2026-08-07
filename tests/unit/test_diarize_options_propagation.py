"""diarize 开关 (1b) — diarize 字段进 TranscribeOptions + 全链路穿透 + 部署假设

链路: FileUploadRequest.diarize → TranscribeOptions → task.options →
task_manager._process_task → 两套 pool (inproc 直传 / file-based model_dump 进
.task JSON) → worker 解析 → Qwen3DiarizeTranscriber.transcribe(options=...).

部署假设 (Codex T4): 老 server 的 Pydantic 忽略未知 diarize 字段 (静默照常带
speaker) → server 先升级、客户端后启用字段. 单测把部署假设钉成被测事实.

funasr pool 协议 (T-D #2): FunASR 路径不传 extra_task_fields, options 不写进
funasr 任务文件.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from src.models.schemas import (
    FileUploadRequest,
    TranscribeOptions,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
)


# ==================== schema ====================


def test_file_upload_request_diarize_default_true():
    req = FileUploadRequest(file_name="a.wav", file_size=1, file_hash="h")
    assert req.diarize is True


def test_file_upload_request_diarize_set_false():
    req = FileUploadRequest(file_name="a.wav", file_size=1, file_hash="h", diarize=False)
    assert req.diarize is False


def test_transcribe_options_diarize_default_true():
    opts = TranscribeOptions()
    assert opts.diarize is True


def test_transcribe_options_model_dump_roundtrip():
    opts = TranscribeOptions(language="eng", diarize=False)
    d = opts.model_dump()
    # word_align (2026-06-16) 进 options, model_dump 默认 False
    assert d == {"language": "eng", "diarize": False, "word_align": False}
    restored = TranscribeOptions(**d)
    assert restored.diarize is False
    assert restored.language == "eng"
    assert restored.word_align is False


# ==================== 部署假设: 老 server 忽略未知 diarize ====================


class _OldFileUploadRequest(BaseModel):
    """diarize 落地前的 FileUploadRequest 形状复刻 (老 server 部署假设)"""
    file_name: str
    file_size: int
    file_hash: str
    force_refresh: bool = False
    output_format: str = "json"
    engine: Optional[str] = None
    language: Optional[str] = None

    model_config = {"protected_namespaces": ()}


def test_old_server_ignores_unknown_diarize_field():
    """老 server Pydantic (默认 extra=ignore) 收到带 diarize 的 payload 不报错且无此属性.

    这是「server 先升级、客户端后启用字段」部署顺序的前提; 若此测试挂了
    (例如有人把 model_config 改成 extra='forbid'), 部署顺序约束随之失效.
    """
    payload = {
        "file_name": "a.wav", "file_size": 1, "file_hash": "h", "diarize": False,
    }
    req = _OldFileUploadRequest(**payload)
    assert not hasattr(req, "diarize")


def test_current_schema_extra_ignore_semantics():
    """当前 FileUploadRequest 同样 extra=ignore — 未来字段对本版本同理成立"""
    req = FileUploadRequest(
        file_name="a.wav", file_size=1, file_hash="h", future_unknown_field=123,
    )
    assert not hasattr(req, "future_unknown_field")


# ==================== create_task / 分片 session 回填 ====================


@pytest.mark.asyncio
async def test_create_task_propagates_diarize_false():
    from src.core.task_manager import TaskManager

    tm = TaskManager()
    req = FileUploadRequest(
        file_name="a.wav", file_size=1, file_hash="h", diarize=False
    )
    task = await tm.create_task(req, task_id="t-d1")
    assert task.options.diarize is False


@pytest.fixture
def fake_ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.remote_address = ("127.0.0.1", 1234)
    return ws


@pytest.mark.asyncio
async def test_chunked_session_captures_diarize(fake_ws):
    from src.api.websocket_handler import WebSocketHandler

    handler = WebSocketHandler()
    data = {
        "file_name": "x.wav", "file_size": 10000, "file_hash": "h-x",
        "total_chunks": 1, "diarize": False,
    }
    await handler._handle_chunked_upload_request(fake_ws, "conn-1", data, request=FileUploadRequest(**data))
    session = next(iter(handler.upload_sessions.values()))
    assert session.get("diarize") is False


@pytest.mark.asyncio
async def test_chunked_finalize_builds_request_with_diarize(fake_ws, tmp_path):
    """分片 finalize 把 session 记录的 diarize 回填进 FileUploadRequest"""
    from src.api.websocket_handler import WebSocketHandler

    handler = WebSocketHandler()
    temp_file = tmp_path / "chunks.bin"
    temp_file.write_bytes(b"\x00" * 8)
    session = {
        "task_id": "t-fin", "file_name": "x.wav", "file_size": 8,
        "file_hash": "match", "chunk_size": 8, "total_chunks": 1,
        "received_chunks": 1, "temp_file_path": str(temp_file),
        "chunks_received": {0}, "output_format": "json",
        "force_refresh": True,  # 跳过缓存查询路径
        "connection_id": "conn-1", "engine": None,
        "language": "eng", "diarize": False,
    }
    handler.upload_sessions["t-fin"] = session

    captured = {}

    async def fake_create_task(request, task_id=None):
        captured["request"] = request
        return TranscriptionTask(
            task_id=task_id, file_name=request.file_name, file_path="",
            file_size=request.file_size, file_hash=request.file_hash,
        )

    with patch.object(handler, "_calculate_file_hash", return_value="match"), \
         patch("src.utils.file_utils.save_uploaded_file",
               new=AsyncMock(return_value=(str(tmp_path / "saved.wav"), None))), \
         patch("src.core.task_manager.task_manager") as mock_tm:
        mock_tm.create_task = AsyncMock(side_effect=fake_create_task)
        mock_tm.submit_task = AsyncMock(return_value=None)
        await handler._finalize_chunked_upload(fake_ws, "t-fin")

    req = captured["request"]
    assert req.diarize is False
    assert req.language == "eng"


# ==================== word_align 分片 session 穿透 (2026-06-16 显存落地, codex #2) ====================


@pytest.mark.asyncio
async def test_chunked_session_captures_word_align(fake_ws):
    from src.api.websocket_handler import WebSocketHandler

    handler = WebSocketHandler()
    data = {
        "file_name": "x.wav", "file_size": 10000, "file_hash": "h-wa",
        "total_chunks": 1, "word_align": True,
    }
    await handler._handle_chunked_upload_request(fake_ws, "conn-1", data, request=FileUploadRequest(**data))
    session = next(iter(handler.upload_sessions.values()))
    assert session.get("word_align") is True


@pytest.mark.asyncio
async def test_chunked_finalize_builds_request_with_word_align(fake_ws, tmp_path):
    """codex #2: 分片 finalize 把 session 记录的 word_align 回填进 FileUploadRequest,
    保证早返回缓存路径 (在 create_task 之前) 用的是请求级 word_align."""
    from src.api.websocket_handler import WebSocketHandler

    handler = WebSocketHandler()
    temp_file = tmp_path / "chunks.bin"
    temp_file.write_bytes(b"\x00" * 8)
    session = {
        "task_id": "t-finwa", "file_name": "x.wav", "file_size": 8,
        "file_hash": "match", "chunk_size": 8, "total_chunks": 1,
        "received_chunks": 1, "temp_file_path": str(temp_file),
        "chunks_received": {0}, "output_format": "json",
        "force_refresh": True,  # 跳过缓存查询路径
        "connection_id": "conn-1", "engine": None,
        "language": "eng", "diarize": True, "word_align": True,
    }
    handler.upload_sessions["t-finwa"] = session

    captured = {}

    async def fake_create_task(request, task_id=None):
        captured["request"] = request
        return TranscriptionTask(
            task_id=task_id, file_name=request.file_name, file_path="",
            file_size=request.file_size, file_hash=request.file_hash,
        )

    with patch.object(handler, "_calculate_file_hash", return_value="match"), \
         patch("src.utils.file_utils.save_uploaded_file",
               new=AsyncMock(return_value=(str(tmp_path / "saved.wav"), None))), \
         patch("src.core.task_manager.task_manager") as mock_tm:
        mock_tm.create_task = AsyncMock(side_effect=fake_create_task)
        mock_tm.submit_task = AsyncMock(return_value=None)
        await handler._finalize_chunked_upload(fake_ws, "t-finwa")

    req = captured["request"]
    assert req.word_align is True


# ==================== task_manager → transcriber 穿透 ====================


@pytest.mark.asyncio
async def test_process_task_passes_options_to_transcriber(tmp_path):
    from src.core.task_manager import TaskManager

    mgr = TaskManager()
    task = TranscriptionTask(
        task_id="t-opt", file_name="x.wav", file_path=str(tmp_path / "x.wav"),
        file_size=100, file_hash="h", engine="qwen3",
        options=TranscribeOptions(language="eng", diarize=False),
    )
    mgr.tasks["t-opt"] = task
    (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

    fake_result = TranscriptionResult(
        task_id="t-opt", file_name="x.wav", file_hash="h", duration=1.0,
        segments=[TranscriptionSegment(start_time=0, end_time=1, text="hi", speaker="Speaker1")],
        speakers=["Speaker1"], processing_time=0.1,
    )
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe = AsyncMock(return_value=(fake_result, {}))

    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
         patch("src.core.task_manager.db_manager") as mock_db:
        mock_db.save_result = AsyncMock()
        with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
            await mgr._process_task("t-opt")

    kwargs = fake_transcriber.transcribe.call_args.kwargs
    opts = kwargs.get("options")
    assert isinstance(opts, TranscribeOptions)
    assert opts.diarize is False
    assert opts.language == "eng"


# ==================== 两套 pool 穿透 ====================


@pytest.mark.asyncio
async def test_inproc_pool_forwards_options():
    from src.core.qwen3_inproc_pool import Qwen3InProcPool

    inner = MagicMock()
    inner.initialize = AsyncMock(return_value=None)
    inner.transcribe = AsyncMock(return_value="R")
    pool = Qwen3InProcPool(pool_size=1, transcriber_factory=lambda: inner)
    await pool.initialize()
    opts = TranscribeOptions(language="eng", diarize=False)
    await pool.transcribe(
        audio_path="a.wav", task_id="t", progress_callback=None,
        output_format="json", options=opts,
    )
    kwargs = inner.transcribe.call_args.kwargs
    assert kwargs.get("options") is opts


@pytest.mark.asyncio
async def test_file_pool_serializes_options_into_task_fields():
    """file-based pool: options 用 model_dump() 进 extra_task_fields (T-D #1)"""
    from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

    custom_pool = MagicMock()
    custom_pool.generate_with_pool = AsyncMock(return_value={"format": "srt", "content": "x"})
    wrapper = Qwen3PoolTranscriber(pool_size=1, pool=custom_pool)
    await wrapper.transcribe(
        audio_path="/fake/a.wav", task_id="t", output_format="srt",
        options=TranscribeOptions(language="kor", diarize=False),
    )
    fields = custom_pool.generate_with_pool.call_args.kwargs["extra_task_fields"]
    assert fields["output_format"] == "srt"
    assert fields["options"] == {"language": "kor", "diarize": False, "word_align": False}
    # options 必须是 JSON 可序列化 dict (写 .task 文件)
    json.dumps(fields)


# ==================== worker 解析 options ====================


def _capture_transcriber(captured: dict):
    fake = MagicMock()

    async def fake_transcribe(audio_path, task_id, progress_callback=None,
                              output_format="json", options=None):
        captured["options"] = options
        return {
            "format": "srt", "content": "1\n00:00:00,000 --> 00:00:01,000\nx\n",
            "file_name": Path(audio_path).name, "file_hash": "h",
            "duration": 1.0, "processing_time": 0.01,
            "raw_result": {"engine": "qwen3"},
        }

    fake.transcribe = fake_transcribe
    return fake


def test_worker_parses_nested_options(tmp_path):
    """worker 读 .task JSON 嵌套 options dict → TranscribeOptions (T-D #10)"""
    from src.core import qwen3_worker_process as wp

    task_file = tmp_path / "worker_0_t1.task"
    task_file.write_text(json.dumps({
        "task_id": "t1", "audio_path": "/fake/a.wav",
        "source_audio_path": "/fake/a.wav", "output_format": "srt",
        "options": {"language": "eng", "diarize": False, "terms": ["Alpha"]},
    }), encoding="utf-8")

    captured = {}
    wp.process_task(0, _capture_transcriber(captured), str(task_file), str(tmp_path))
    opts = captured["options"]
    assert isinstance(opts, TranscribeOptions)
    assert opts.language == "eng"
    assert opts.diarize is False
    assert opts.terms == ["Alpha"]


def test_worker_falls_back_to_flat_language_for_old_task_files(tmp_path):
    """老任务文件 (无 options, 平铺 language) → 兜底构造 options, diarize 默认 True"""
    from src.core import qwen3_worker_process as wp

    task_file = tmp_path / "worker_0_t2.task"
    task_file.write_text(json.dumps({
        "task_id": "t2", "audio_path": "/fake/a.wav",
        "source_audio_path": "/fake/a.wav", "output_format": "srt",
        "language": "jpn",
    }), encoding="utf-8")

    captured = {}
    wp.process_task(0, _capture_transcriber(captured), str(task_file), str(tmp_path))
    opts = captured["options"]
    assert isinstance(opts, TranscribeOptions)
    assert opts.language == "jpn"
    assert opts.diarize is True
    assert opts.terms == []


# ==================== funasr pool 协议: options 不进任务文件 ====================


@pytest.mark.asyncio
async def test_funasr_pool_call_has_no_extra_task_fields():
    """FunASR 路径 generate_with_pool 不带 extra_task_fields (options 不写进
    funasr 任务文件, T-D #2) — funasr 的 diarize 语义由 serve 层投影实现 (D4)."""
    from src.core.funasr_transcriber import FunASRTranscriber

    tx = FunASRTranscriber.__new__(FunASRTranscriber)  # 跳过 __init__ 模型加载
    tx.is_initialized = True
    tx.concurrency_mode = "pool"
    tx.config = {"funasr": {"batch_size_s": 300}, "transcription": {}}
    tx.model_pool = MagicMock()
    tx.model_pool.generate_with_pool = AsyncMock(
        return_value=[{"sentence_info": [{"start": 0, "end": 1000, "text": "你好", "spk": 0}]}]
    )

    with patch("src.core.funasr_transcriber.get_audio_duration", return_value=10.0), \
         patch("src.utils.file_utils.calculate_file_hash", new=AsyncMock(return_value="h")), \
         patch("src.core.funasr_transcriber.release_accelerator_memory"):
        await tx.transcribe(
            audio_path="/fake/a.wav", task_id="t", output_format="json",
            options=TranscribeOptions(language="eng", diarize=False),
        )

    kwargs = tx.model_pool.generate_with_pool.call_args.kwargs
    assert "extra_task_fields" not in kwargs or kwargs["extra_task_fields"] is None
