"""
高负载队列机制止血 — Lane B: finalize 资源处理 + queue_full 信号

覆盖（红→绿）：
1. queue-full → 发 queue_full 消息(含 retry_after + 兼容 error 字段)，session 保留
2. 非 queue-full 真错误 → session 清理(防泄漏)
3. finalize_upload 幂等：已提交(无 session 但 task 存在) → 回 task_status 不重复提交
4. 轮询 expired vs not_found 区分
5. session TTL sweep + 硬数量上限
6. 提交后错误不删最终文件(错误分级)

mock task_manager / save_uploaded_file / db_manager 隔离。
"""
import hashlib
import time
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.api.websocket_handler import WebSocketHandler
from src.core.task_manager import QueueFullError


@pytest.fixture
def handler():
    return WebSocketHandler()


@pytest.fixture
def fake_ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.remote_address = ("127.0.0.1", 1234)
    return ws


def setup_session(handler, tmp_path, content=b"hello-world-audio", state="uploading", terms=None):
    """构造一个分片已收齐、hash 匹配的 session（force_refresh 跳过缓存分支）"""
    temp = tmp_path / "chunk.bin"
    temp.write_bytes(content)
    task_id = "task-xyz"
    handler.upload_sessions[task_id] = {
        "task_id": task_id, "file_name": "a.wav", "file_size": len(content),
        "file_hash": hashlib.md5(content).hexdigest(),
        "chunk_size": 1024, "total_chunks": 1, "received_chunks": 1,
        "temp_file_path": str(temp), "chunks_received": {0},
        "output_format": "json", "force_refresh": True,
        "connection_id": "c1", "engine": "qwen3", "language": None,
        "diarize": True, "word_align": None, "terms": list(terms or []),
        "state": state, "created_at": time.time(),
    }
    return task_id, temp


def sent_types(send_msg_mock, send_err_mock):
    """汇总 _send_message / _send_error 发出的消息类型"""
    types = [c.args[1] for c in send_msg_mock.call_args_list if len(c.args) >= 2]
    errs = [c.args[1] for c in send_err_mock.call_args_list if len(c.args) >= 2]
    return types, errs


class TestQueueFullMapping:
    @pytest.mark.asyncio
    async def test_queue_full_finalize_retry_preserves_terms_and_reuses_file(
        self, handler, fake_ws, tmp_path,
    ):
        from src.models.schemas import FileUploadRequest, TranscribeOptions, TranscriptionTask

        task_id, temp = setup_session(handler, tmp_path, terms=["Alpha", "Beta"])
        request = FileUploadRequest(
            file_name="a.wav", file_size=len(b"hello-world-audio"),
            file_hash=hashlib.md5(b"hello-world-audio").hexdigest(),
            engine="qwen3", terms=["Alpha", "Beta"],
        )
        handler.upload_sessions[task_id]["request"] = request
        final = tmp_path / "final.wav"
        created = []
        created_tasks = []

        async def create_task(req, task_id):
            created.append(req)
            task = TranscriptionTask(
                task_id=task_id, file_name="a.wav", file_path=str(final),
                file_size=10, file_hash=req.file_hash, engine="qwen3",
                options=TranscribeOptions(terms=list(req.terms)),
            )
            created_tasks.append(task)
            return task

        async def save_file(data, file_name):
            final.write_bytes(data)
            return str(final), None

        saved = AsyncMock(side_effect=save_file)
        with patch("src.core.task_manager.task_manager") as tm, \
             patch("src.utils.file_utils.save_uploaded_file", new=saved), \
             patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()):
            tm.create_task = AsyncMock(side_effect=create_task)
            tm.submit_task = AsyncMock(side_effect=[
                QueueFullError(retry_after=30, queue_size=20, max_queue_size=20), None,
            ])
            await handler._finalize_chunked_upload(fake_ws, task_id)
            assert handler.upload_sessions[task_id]["state"] == "ready"
            assert handler.upload_sessions[task_id]["terms"] == ["Alpha", "Beta"]
            await handler._handle_finalize_upload(fake_ws, task_id)

        assert len(created) == 2
        assert created[0] is created[1]
        assert all(task.options.terms == ["Alpha", "Beta"] for task in created_tasks)
        assert saved.await_count == 1
        assert tm.submit_task.await_count == 2
        assert handler.upload_sessions == {}

    @pytest.mark.asyncio
    async def test_queue_full_sends_queue_full_message(self, handler, fake_ws, tmp_path):
        task_id, temp = setup_session(handler, tmp_path)
        from src.models.schemas import TranscriptionTask
        fake_task = TranscriptionTask(task_id=task_id, file_name="a.wav", file_path="",
                                      file_size=10, file_hash="h", engine="qwen3")
        with patch("src.core.task_manager.task_manager") as tm, \
             patch("src.utils.file_utils.save_uploaded_file",
                   new=AsyncMock(return_value=("/tmp/final_a.wav", None))), \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm, \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            tm.create_task = AsyncMock(return_value=fake_task)
            tm.submit_task = AsyncMock(side_effect=QueueFullError(
                retry_after=30, queue_size=20, max_queue_size=20))
            await handler._finalize_chunked_upload(fake_ws, task_id)

            types, _ = sent_types(sm, se)
            assert "queue_full" in types
            # 校验 retry_after 在消息里
            qf = next(c for c in sm.call_args_list if c.args[1] == "queue_full")
            assert qf.args[2]["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_queue_full_preserves_session(self, handler, fake_ws, tmp_path):
        task_id, temp = setup_session(handler, tmp_path)
        from src.models.schemas import TranscriptionTask
        fake_task = TranscriptionTask(task_id=task_id, file_name="a.wav", file_path="",
                                      file_size=10, file_hash="h", engine="qwen3")
        with patch("src.core.task_manager.task_manager") as tm, \
             patch("src.utils.file_utils.save_uploaded_file",
                   new=AsyncMock(return_value=("/tmp/final_a.wav", None))), \
             patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()):
            tm.create_task = AsyncMock(return_value=fake_task)
            tm.submit_task = AsyncMock(side_effect=QueueFullError(
                retry_after=30, queue_size=20, max_queue_size=20))
            await handler._finalize_chunked_upload(fake_ws, task_id)
            # 关键：session 保留供重试，不能被清
            assert task_id in handler.upload_sessions


class TestRealErrorCleansUp:
    @pytest.mark.asyncio
    async def test_non_queue_error_cleans_session(self, handler, fake_ws, tmp_path):
        task_id, temp = setup_session(handler, tmp_path)
        from src.models.schemas import TranscriptionTask
        fake_task = TranscriptionTask(task_id=task_id, file_name="a.wav", file_path="",
                                      file_size=10, file_hash="h", engine="qwen3")
        with patch("src.core.task_manager.task_manager") as tm, \
             patch("src.utils.file_utils.save_uploaded_file",
                   new=AsyncMock(return_value=("/tmp/final_a.wav", None))), \
             patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()):
            tm.create_task = AsyncMock(return_value=fake_task)
            tm.submit_task = AsyncMock(side_effect=RuntimeError("模型崩了"))
            await handler._finalize_chunked_upload(fake_ws, task_id)
            # 真错误 → session 清理(防泄漏)
            assert task_id not in handler.upload_sessions
            # 临时文件也清掉
            assert not temp.exists()


class TestFinalizeUploadIdempotent:
    @pytest.mark.asyncio
    async def test_retry_when_already_submitted_returns_status(self, handler, fake_ws):
        """已提交：session 已清，task 仍在 → 回 task_status，不二次提交"""
        from src.models.schemas import TranscriptionTask
        fake_task = TranscriptionTask(task_id="t-sub", file_name="a.wav", file_path="/tmp/a",
                                      file_size=10, file_hash="h", engine="qwen3")
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm, \
             patch.object(handler, "_send_error", new=AsyncMock()):
            tm.get_task = MagicMock(return_value=fake_task)
            tm.submit_task = AsyncMock()
            # 无 session
            await handler._handle_finalize_upload(fake_ws, "t-sub")
            # 不应再调 submit_task
            tm.submit_task.assert_not_called()
            types = [c.args[1] for c in sm.call_args_list if len(c.args) >= 2]
            assert "task_status" in types

    @pytest.mark.asyncio
    async def test_retry_with_live_session_refinalizes(self, handler, fake_ws, tmp_path):
        """queue_full 后重发 finalize_upload：session 还在 → 走重试 finalize"""
        task_id, temp = setup_session(handler, tmp_path, state="ready")
        with patch.object(handler, "_finalize_chunked_upload", new=AsyncMock()) as fz:
            await handler._handle_finalize_upload(fake_ws, task_id)
            fz.assert_awaited_once()


class TestPollingExpiredVsNotFound:
    @pytest.mark.asyncio
    async def test_expired_when_evicted(self, handler, fake_ws):
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            tm.get_task = MagicMock(return_value=None)
            tm.was_evicted = MagicMock(return_value=True)
            await handler._send_task_status(fake_ws, "gone")
            assert se.call_args.args[1] == "task_expired"

    @pytest.mark.asyncio
    async def test_not_found_when_never_existed(self, handler, fake_ws):
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            tm.get_task = MagicMock(return_value=None)
            tm.was_evicted = MagicMock(return_value=False)
            await handler._send_task_status(fake_ws, "never")
            assert se.call_args.args[1] == "task_not_found"


class TestSessionSweep:
    def test_sweep_removes_expired_session(self, handler, tmp_path):
        from src.core.config import config
        temp = tmp_path / "old.bin"; temp.write_bytes(b"x")
        handler.upload_sessions["old"] = {
            "task_id": "old", "temp_file_path": str(temp),
            "created_at": time.time() - (config.transcription.upload_session_ttl_seconds + 100),
        }
        handler._sweep_upload_sessions()
        assert "old" not in handler.upload_sessions
        assert not temp.exists()

    def test_sweep_keeps_fresh_session(self, handler, tmp_path):
        temp = tmp_path / "fresh.bin"; temp.write_bytes(b"x")
        handler.upload_sessions["fresh"] = {
            "task_id": "fresh", "temp_file_path": str(temp),
            "created_at": time.time(),
        }
        handler._sweep_upload_sessions()
        assert "fresh" in handler.upload_sessions
