"""毫秒输入渲染 SRT 时先四舍五入总毫秒，再拆分时间字段。"""
from __future__ import annotations

import sys
import threading
import wave
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def funasr_transcriber(monkeypatch):
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


@pytest.fixture
def database(tmp_path):
    from src.core.database import DatabaseManager

    return DatabaseManager(db_path=str(tmp_path / "srt.db"))


def _write_audio(path):
    """写一秒真实 WAV，避免 mock 音频探测和文件哈希。"""
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000)


@pytest.mark.asyncio
@pytest.mark.parametrize("formatter", ["funasr", "database"])
async def test_srt_millisecond_rounding_and_second_carry(
    formatter, funasr_transcriber, database, tmp_path
):
    """通过转录或缓存命中入口验证 raw 毫秒取整与秒进位。"""
    sentences = [
        {"start": 1234.9, "end": 999.6, "text": "fractional", "spk": 0},
        {"start": 0, "end": 1234, "text": "integer", "spk": 0},
    ]
    raw_result = [{"sentence_info": sentences}]
    if formatter == "funasr":
        audio = tmp_path / "srt-rounding.wav"
        _write_audio(audio)
        funasr_transcriber.is_initialized = True
        funasr_transcriber.concurrency_mode = "lock"
        funasr_transcriber.config = {"funasr": {"batch_size_s": 20}, "transcription": {}}
        funasr_transcriber._model_lock = threading.Lock()
        funasr_transcriber.model = MagicMock()
        funasr_transcriber.model.generate.return_value = raw_result
        result = await funasr_transcriber.transcribe(
            audio_path=str(audio), task_id="srt-task", output_format="srt"
        )
        srt = result["content"]
    else:
        from src.models.schemas import TranscriptionResult, TranscriptionSegment

        await database.init_db()
        cached_result = TranscriptionResult(
            task_id="srt-task",
            file_name="srt-rounding.wav",
            file_hash="srt-hash",
            duration=1.0,
            segments=[
                TranscriptionSegment(
                    start_time=1.2349, end_time=0.9996, text="fractional", speaker="Speaker1"
                ),
                TranscriptionSegment(
                    start_time=0, end_time=1.234, text="integer", speaker="Speaker1"
                ),
            ],
            speakers=["Speaker1"],
            processing_time=0.1,
        )
        await database.save_result(cached_result, raw_result=raw_result)
        result = await database.get_cached_result("srt-hash", output_format="srt")
        assert result is not None
        srt = result["content"]

    assert "00:00:01,235 --> 00:00:01,000" in srt
    assert "00:00:00,000 --> 00:00:01,234" in srt
