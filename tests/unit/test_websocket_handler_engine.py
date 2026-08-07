"""
PR1 — websocket_handler engine 字段透传测试

验证：
1. 单文件上传：cache 查询时把 engine 传给 db_manager
2. 分片上传：session 字典记录 engine
3. 分片上传完成：重建 FileUploadRequest 时带上 engine
"""
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.api.websocket_handler import WebSocketHandler
from src.models.schemas import FileUploadRequest, TranscribeOptions, TranscriptionTask


@pytest.fixture
def handler():
    return WebSocketHandler()


@pytest.fixture
def fake_websocket():
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.remote_address = ("127.0.0.1", 1234)
    return ws


class TestChunkedUploadSessionCapturesEngine:
    @pytest.mark.asyncio
    async def test_session_captures_engine_field(self, handler: WebSocketHandler, fake_websocket):
        data = {
            "file_name": "x.wav",
            "file_size": 10000,
            "file_hash": "h-x",
            "total_chunks": 1,
            "engine": "qwen3",
        }
        await handler._handle_chunked_upload_request(fake_websocket, "conn-1", data, request=FileUploadRequest(**data))
        # 应该新建一个 session
        assert len(handler.upload_sessions) == 1
        session = next(iter(handler.upload_sessions.values()))
        assert session.get("engine") == "qwen3"

    @pytest.mark.asyncio
    async def test_session_engine_defaults_to_none_when_missing(self, handler: WebSocketHandler, fake_websocket):
        data = {
            "file_name": "x.wav",
            "file_size": 10000,
            "file_hash": "h-x",
            "total_chunks": 1,
        }
        await handler._handle_chunked_upload_request(fake_websocket, "conn-1", data, request=FileUploadRequest(**data))
        session = next(iter(handler.upload_sessions.values()))
        assert session.get("engine") is None


class TestSingleUploadCacheLookupPassesEngine:
    @pytest.mark.asyncio
    async def test_cache_lookup_includes_engine_from_request(self, handler: WebSocketHandler, fake_websocket):
        """单文件上传：早期 cache 短路时必须用 task.engine 查"""
        data = {
            "file_name": "x.wav",
            "file_size": 100,
            "file_hash": "h-1",
            "engine": "qwen3",
            "force_refresh": False,
        }
        # mock task_manager.create_task → 返回带 engine=qwen3 的 task
        from src.models.schemas import TranscriptionTask
        fake_task = TranscriptionTask(
            task_id="t-1",
            file_name="x.wav",
            file_path="",
            file_size=100,
            file_hash="h-1",
            engine="qwen3",
        )
        with patch("src.core.task_manager.task_manager") as mock_tm, \
             patch("src.core.database.db_manager") as mock_db:
            mock_tm.create_task = AsyncMock(return_value=fake_task)
            mock_db.get_cached_result = AsyncMock(return_value=None)
            await handler._handle_upload_request(fake_websocket, "conn-1", data)

            assert mock_db.get_cached_result.called
            call = mock_db.get_cached_result.call_args
            kwargs = call.kwargs
            # 接受位置或关键字传参
            engine_arg = kwargs.get("engine")
            if engine_arg is None and len(call.args) >= 3:
                engine_arg = call.args[2]
            assert engine_arg == "qwen3", f"cache lookup 应带 engine=qwen3，调用: {call}"


@pytest.mark.asyncio
async def test_nonempty_terms_skip_single_upload_cache_lookup(handler, fake_websocket):
    data = {"file_name": "x.wav", "file_size": 100, "file_hash": "h-terms", "terms": ["Alpha"]}
    task = TranscriptionTask(task_id="t-terms", file_name="x.wav", file_path="", file_size=100,
                             file_hash="h-terms", options=TranscribeOptions(terms=["Alpha"]))
    with patch("src.core.task_manager.task_manager") as mock_tm, \
         patch("src.core.database.db_manager") as mock_db, \
         patch("src.core.database.cache_params_for") as params:
        mock_tm.create_task = AsyncMock(return_value=task)
        mock_db.get_cached_result = AsyncMock(return_value=None)
        await handler._handle_upload_request(fake_websocket, "conn-terms", data)
    mock_db.get_cached_result.assert_not_called()
    params.assert_not_called()


@pytest.mark.asyncio
async def test_nonempty_terms_skip_chunked_finalize_cache_lookup(handler, fake_websocket, tmp_path):
    temp_file = tmp_path / "chunks.bin"
    temp_file.write_bytes(b"x")
    request = FileUploadRequest(file_name="x.wav", file_size=1, file_hash="h-terms", terms=["Alpha"])
    task = TranscriptionTask(task_id="t-ch-terms", file_name="x.wav", file_path="", file_size=1,
                             file_hash="h-terms", options=TranscribeOptions(terms=["Alpha"]))
    handler.upload_sessions["t-ch-terms"] = {
        "task_id": "t-ch-terms", "file_name": "x.wav", "file_size": 1, "file_hash": "h-terms",
        "temp_file_path": str(temp_file), "finalized_file_path": str(temp_file), "force_refresh": False,
        "output_format": "json", "engine": None, "language": None, "diarize": True,
        "word_align": None, "terms": ["Alpha"], "request": request,
    }
    with patch.object(handler, "_calculate_file_hash", return_value="h-terms"), \
         patch.object(handler, "_cleanup_upload_session", new=AsyncMock()), \
         patch("src.core.database.db_manager") as db, \
         patch("src.core.database.cache_params") as params, \
         patch("src.core.task_manager.task_manager") as mock_tm:
        db.get_cached_result = AsyncMock(return_value=None)
        mock_tm.create_task = AsyncMock(return_value=task)
        mock_tm.submit_task = AsyncMock(return_value=None)
        await handler._finalize_chunked_upload(fake_websocket, "t-ch-terms")
    db.get_cached_result.assert_not_called()
    params.assert_not_called()
