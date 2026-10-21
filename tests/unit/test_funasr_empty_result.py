"""FunASR 顶层空列表按合法空转录处理。"""
from __future__ import annotations

import sys
import threading
import wave
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def transcriber(monkeypatch):
    """加载转录器并用 object.__new__ 跳过模型初始化。"""
    for name, attrs in (
        ("funasr", {"AutoModel": MagicMock()}),
        ("torch", {}),
        ("torch.cuda", {}),
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


def _prepare_transcriber(instance, result):
    """只替换最外层模型调用，让公开 transcribe 走真实音频与文件路径。"""
    instance.is_initialized = True
    instance.concurrency_mode = "lock"
    instance.config = {"funasr": {"batch_size_s": 20}, "transcription": {}}
    instance._model_lock = threading.Lock()
    instance.model = MagicMock()
    instance.model.generate.return_value = result


def _write_audio(path):
    """生成一秒真实 WAV，供公开转录路径探测时长并计算文件哈希。"""
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000)


@pytest.mark.asyncio
async def test_top_level_empty_list_is_success_and_logs_marker(transcriber, monkeypatch, tmp_path):
    """真实转录入口把 FunASR 的 [] 作为空成功，并记录可检索警告。"""
    import src.core.funasr_transcriber as module

    logger = MagicMock()
    monkeypatch.setattr(module, "logger", logger)
    _prepare_transcriber(transcriber, [])
    audio = tmp_path / "empty.wav"
    _write_audio(audio)

    result, raw_result = await transcriber.transcribe(
        audio_path=str(audio), task_id="empty-task", output_format="json"
    )

    assert raw_result == []
    assert result.segments == []
    assert any(
        call.args[:2] == ("FUNASR-EMPTY-RESULT: task_id={}", "empty-task")
        for call in logger.warning.call_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [[{"text": 1}], [{"text": "recognized"}]])
async def test_invalid_top_level_payload_still_fails(transcriber, result, tmp_path):
    """空列表豁免不放宽其它非法结果形状。"""
    _prepare_transcriber(transcriber, result)
    audio = tmp_path / "invalid.wav"
    _write_audio(audio)

    with pytest.raises(Exception, match="FunASR result invalid-shape"):
        await transcriber.transcribe(
            audio_path=str(audio), task_id="invalid-task", output_format="json"
        )
