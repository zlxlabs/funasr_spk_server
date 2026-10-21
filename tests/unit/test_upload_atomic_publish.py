"""上传文件内容寻址路径必须以原子替换发布，不能截断正在被读取的文件。"""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import src.utils.file_utils as file_utils
from src.api.websocket_handler import WebSocketHandler
from src.core.config import config


@pytest.mark.asyncio
async def test_save_uploaded_file_replaces_path_without_truncating_existing_reader(
    tmp_path, monkeypatch,
):
    """重复保存同内容后，旧文件句柄仍指向完整旧 inode。"""
    monkeypatch.setattr(config.server, "upload_dir", str(tmp_path))
    content = b"identical upload content" * 1024

    file_path, file_hash = await file_utils.save_uploaded_file(content, "audio.wav")
    old_reader = open(file_path, "rb")
    try:
        old_inode = os.fstat(old_reader.fileno()).st_ino
        second_path, second_hash = await file_utils.save_uploaded_file(content, "audio.wav")

        assert second_path == file_path
        assert second_hash == file_hash == hashlib.md5(content).hexdigest()
        assert os.stat(file_path).st_ino != old_inode
        assert old_reader.read() == content
        assert Path(file_path).read_bytes() == content
    finally:
        old_reader.close()


@pytest.mark.asyncio
async def test_save_failure_keeps_previous_file_and_logs_marker_without_temp_file(
    tmp_path, monkeypatch,
):
    """写临时文件中途失败时保留旧目标，移除临时文件并记录可检索标记。"""
    import tempfile

    monkeypatch.setattr(config.server, "upload_dir", str(tmp_path))
    old_content = b"previous complete content"
    file_path, _ = await file_utils.save_uploaded_file(old_content, "audio.wav")
    original_named_tempfile = tempfile.NamedTemporaryFile
    injected_error = OSError("injected mid-write failure")

    class FailingWriter:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.wrapped.__exit__(exc_type, exc_value, traceback)

        def write(self, data):
            self.wrapped.write(data[:5])
            raise injected_error

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    def fail_during_write(*args, **kwargs):
        return FailingWriter(original_named_tempfile(*args, **kwargs))

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_during_write)
    with patch.object(file_utils.logger, "error") as log_error:
        with pytest.raises(OSError, match="injected mid-write failure") as raised:
            await file_utils.save_uploaded_file(old_content, "audio.wav")

    assert raised.value is injected_error
    assert Path(file_path).read_bytes() == old_content
    assert list(tmp_path.iterdir()) == [
        tmp_path / f"{hashlib.md5(old_content).hexdigest()}.wav"
    ]
    assert any(
        "UPLOAD-ATOMIC-PUBLISH-FAILED" in " ".join(map(str, call.args))
        for call in log_error.call_args_list
    )


@pytest.mark.asyncio
async def test_chunked_finalize_atomically_replaces_shared_file(tmp_path, monkeypatch):
    """分片 finalize 入口替换同名目标时，已有 reader 继续读取完整旧文件。"""
    content = b"chunked upload payload" * 1024
    file_hash = hashlib.md5(content).hexdigest()
    monkeypatch.setattr(config.server, "upload_dir", str(tmp_path))
    temp_path = tmp_path / "received-chunks.tmp"
    temp_path.write_bytes(content)
    final_path = tmp_path / f"{file_hash}.wav"
    final_path.write_bytes(content)
    old_reader = open(final_path, "rb")
    old_inode = os.fstat(old_reader.fileno()).st_ino

    handler = WebSocketHandler()
    task_id = "chunked-atomic-finalize"
    handler.upload_sessions[task_id] = {
        "task_id": task_id,
        "file_name": "audio.wav",
        "file_size": len(content),
        "file_hash": file_hash,
        "chunk_size": len(content),
        "total_chunks": 1,
        "received_chunks": 1,
        "temp_file_path": str(temp_path),
        "chunks_received": {0},
        "output_format": "json",
        "force_refresh": True,
        "connection_id": "conn-upload",
        "engine": None,
        "language": None,
        "diarize": True,
        "word_align": None,
        "terms": [],
        "state": "ready",
        "created_at": 0,
        "finalized_file_path": None,
    }

    with patch("src.core.task_manager.task_manager") as task_manager, patch.object(
        handler, "_send_message", new=AsyncMock()
    ), patch.object(handler, "_send_error", new=AsyncMock()):
        task_manager.create_task = AsyncMock(return_value=SimpleNamespace(task_id=task_id))
        task_manager.submit_task = AsyncMock(return_value=None)
        await handler._handle_finalize_upload(SimpleNamespace(), task_id)

    try:
        assert final_path.stat().st_ino != old_inode
        assert old_reader.read() == content
        assert final_path.read_bytes() == content
        assert not temp_path.exists()
        assert not handler.upload_sessions
        assert list(tmp_path.iterdir()) == [final_path]
    finally:
        old_reader.close()
