"""单一请求 engine 准入判定及其两个调用边界。"""
import pytest

import src.core.task_manager as task_manager_module
from src.api.websocket_handler import WebSocketHandler
from src.core.config import config
from src.models.schemas import FileUploadRequest


def test_check_requested_engine_handles_empty_whitespace_match_and_mismatch():
    assert hasattr(task_manager_module, "check_requested_engine"), (
        "task_manager 缺少共享 engine 判定函数"
    )
    check = task_manager_module.check_requested_engine

    assert check(None, "qwen3") is None
    assert check("   ", "qwen3") is None
    assert check(" qwen3 ", "qwen3") is None
    assert check(" funasr ", "qwen3") == (
        "Server configured with engine='qwen3', "
        "cannot accept engine='funasr'. "
        "Please omit the engine field or set it to 'qwen3'."
    )


def test_websocket_early_rejection_uses_shared_engine_check(monkeypatch):
    calls = []

    def shared_check(requested, server_engine):
        calls.append((requested, server_engine))
        return "shared rejection"

    monkeypatch.setattr(task_manager_module, "check_requested_engine", shared_check)
    monkeypatch.setattr(config.transcription, "default_engine", "qwen3")
    request = FileUploadRequest(
        file_name="audio.wav", file_size=1, file_hash="abc", engine=" funasr "
    )

    assert WebSocketHandler()._upload_engine_error(request) == "shared rejection"
    assert calls == [(" funasr ", "qwen3")]


@pytest.mark.asyncio
async def test_create_task_final_guard_uses_shared_engine_check(monkeypatch):
    calls = []

    def shared_check(requested, server_engine):
        calls.append((requested, server_engine))
        return "shared rejection"

    monkeypatch.setattr(task_manager_module, "check_requested_engine", shared_check)
    monkeypatch.setattr(config.transcription, "default_engine", "qwen3")
    request = FileUploadRequest(
        file_name="audio.wav", file_size=1, file_hash="abc", engine=" funasr "
    )
    manager = task_manager_module.TaskManager()
    try:
        with pytest.raises(ValueError, match="shared rejection"):
            await manager.create_task(request, task_id="engine-check")
    finally:
        manager.executor.shutdown(wait=True)

    assert calls == [(" funasr ", "qwen3")]
