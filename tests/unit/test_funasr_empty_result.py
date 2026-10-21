"""FunASR 顶层空列表按合法空转录处理。"""
from __future__ import annotations

import sys
import threading
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def transcriber(monkeypatch):
    """加载转录器并用 object.__new__ 跳过模型初始化。"""
    for name, attrs in (
        ("funasr", {"AutoModel": MagicMock()}),
        ("torch", {}),
        ("torch.cuda", {}),
        ("ffmpeg", {}),
    ):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    for name in (
        "src.core.funasr_transcriber",
        "src.utils.file_utils",
        "src.utils.torch_utils",
        "src.core.device_manager",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    from src.core.funasr_transcriber import FunASRTranscriber

    yield object.__new__(FunASRTranscriber)
    monkeypatch.delitem(sys.modules, "src.core.funasr_transcriber", raising=False)


def _prepare_transcriber(instance, monkeypatch, module, file_utils, result):
    """让公开 transcribe 入口跳过真实模型、音频和文件系统依赖。"""
    instance.is_initialized = True
    instance.concurrency_mode = "lock"
    instance.config = {"funasr": {"batch_size_s": 20}, "transcription": {}}
    instance._model_lock = threading.Lock()
    instance.model = MagicMock()
    instance.model.generate.return_value = result
    monkeypatch.setattr(module, "get_audio_duration", lambda _path: 1.0)
    monkeypatch.setattr(module, "release_accelerator_memory", lambda **_kwargs: None)
    monkeypatch.setattr(file_utils, "calculate_file_hash", AsyncMock(return_value="empty-hash"))


@pytest.mark.asyncio
async def test_top_level_empty_list_is_success_and_logs_marker(transcriber, monkeypatch):
    """真实转录入口把 FunASR 的 [] 作为空成功，并记录可检索警告。"""
    import src.core.funasr_transcriber as module
    import src.utils.file_utils as file_utils

    logger = MagicMock()
    monkeypatch.setattr(module, "logger", logger)
    _prepare_transcriber(transcriber, monkeypatch, module, file_utils, [])

    result, raw_result = await transcriber.transcribe(
        audio_path="/tmp/empty.wav", task_id="empty-task", output_format="json"
    )

    assert raw_result == []
    assert result.segments == []
    assert any(
        call.args[:2] == ("FUNASR-EMPTY-RESULT: task_id={}", "empty-task")
        for call in logger.warning.call_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [[{"text": 1}], [{"text": "recognized"}]])
async def test_invalid_top_level_payload_still_fails(transcriber, monkeypatch, result):
    """空列表豁免不放宽其它非法结果形状。"""
    import src.core.funasr_transcriber as module
    import src.utils.file_utils as file_utils

    _prepare_transcriber(transcriber, monkeypatch, module, file_utils, result)

    with pytest.raises(Exception, match="FunASR result invalid-shape"):
        await transcriber.transcribe(
            audio_path="/tmp/invalid.wav", task_id="invalid-task", output_format="json"
        )
