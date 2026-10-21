"""毫秒输入渲染 SRT 时先四舍五入总毫秒，再拆分时间字段。"""
from __future__ import annotations

import sys
import threading
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def funasr_transcriber(monkeypatch):
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


@pytest.fixture
def database(tmp_path):
    from src.core.database import DatabaseManager

    return DatabaseManager(db_path=str(tmp_path / "srt.db"))


@pytest.mark.asyncio
@pytest.mark.parametrize("formatter", ["funasr", "database"])
async def test_srt_millisecond_rounding_and_second_carry(
    formatter, funasr_transcriber, database, monkeypatch
):
    """通过转录或缓存命中入口验证 raw 毫秒取整与秒进位。"""
    sentences = [
        {"start": 1234.9, "end": 999.6, "text": "fractional", "spk": 0},
        {"start": 0, "end": 1234, "text": "integer", "spk": 0},
    ]
    raw_result = [{"sentence_info": sentences}]
    if formatter == "funasr":
        import src.core.funasr_transcriber as module
        import src.utils.file_utils as file_utils

        funasr_transcriber.is_initialized = True
        funasr_transcriber.concurrency_mode = "lock"
        funasr_transcriber.config = {"funasr": {"batch_size_s": 20}, "transcription": {}}
        funasr_transcriber._model_lock = threading.Lock()
        funasr_transcriber.model = MagicMock()
        funasr_transcriber.model.generate.return_value = raw_result
        monkeypatch.setattr(module, "get_audio_duration", lambda _path: 1.0)
        monkeypatch.setattr(module, "release_accelerator_memory", lambda **_kwargs: None)
        monkeypatch.setattr(file_utils, "calculate_file_hash", AsyncMock(return_value="srt-hash"))
        result = await funasr_transcriber.transcribe(
            audio_path="/tmp/srt-rounding.wav", task_id="srt-task", output_format="srt"
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
