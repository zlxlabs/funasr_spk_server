"""FunASR 热词走每任务 .txt，避免拼接串被 funasr 当 url / 路径解析。"""
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from src.models.schemas import TranscribeOptions

# 含 funasr 会误判的形态：.txt 后缀、http 前缀、行内空格
TRICKY_TERMS = ["config.txt", "http://x", "hello world"]


class _RecordingLock:
    def __init__(self):
        self.active = False

    def __enter__(self):
        self.active = True

    def __exit__(self, *_):
        self.active = False


def _ready_transcriber(mode, on_hotword):
    """公开入口 FunASRTranscriber.transcribe；只 mock 最外层 generate。"""
    from src.core.funasr_transcriber import FunASRTranscriber

    tx = FunASRTranscriber.__new__(FunASRTranscriber)
    tx.is_initialized = True
    tx.concurrency_mode = mode
    tx.config = {"funasr": {"batch_size_s": 300}, "transcription": {}}
    result = [{"sentence_info": [{"start": 0, "end": 1000, "text": "ok", "spk": 0}]}]

    def _observe(hotword):
        on_hotword(hotword)
        return result

    if mode == "pool":
        tx.model_pool = MagicMock()

        async def generate_with_pool(**kwargs):
            return _observe(kwargs["hotword"])

        tx.model_pool.generate_with_pool = generate_with_pool
    else:
        tx.model = MagicMock()
        tx._model_lock = _RecordingLock()

        def generate(**kwargs):
            return _observe(kwargs["hotword"])

        tx.model.generate.side_effect = generate
    return tx


def _patch_io():
    return (
        patch("src.core.funasr_transcriber.get_audio_duration", return_value=1.0),
        patch("src.utils.file_utils.calculate_file_hash", new=AsyncMock(return_value="h")),
        patch("src.core.funasr_transcriber.release_accelerator_memory"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["lock", "pool"])
async def test_transcribe_passes_existing_txt_with_verbatim_lines_then_deletes(mode):
    observed = {}

    def on_hotword(hotword):
        observed["hotword"] = hotword
        exists = os.path.isfile(hotword)
        observed["exists_during"] = exists
        observed["lines"] = (
            Path(hotword).read_text(encoding="utf-8").splitlines() if exists else None
        )

    tx = _ready_transcriber(mode, on_hotword)
    duration, file_hash, release = _patch_io()
    with duration, file_hash, release:
        await tx.transcribe(
            "/fake/a.wav",
            "task-hw",
            options=TranscribeOptions(terms=list(TRICKY_TERMS)),
        )

    path = observed["hotword"]
    assert path.endswith(".txt"), path
    assert observed["exists_during"] is True
    assert observed["lines"] == TRICKY_TERMS
    assert os.path.isfile(path) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["lock", "pool"])
async def test_transcribe_deletes_hotword_file_when_generate_raises(mode):
    observed = {}

    def on_hotword(hotword):
        observed["hotword"] = hotword
        observed["exists_during"] = os.path.isfile(hotword)
        raise RuntimeError("model boom")

    tx = _ready_transcriber(mode, on_hotword)
    duration, file_hash, release = _patch_io()
    with duration, file_hash, release:
        with pytest.raises(Exception, match="转录失败"):
            await tx.transcribe(
                "/fake/a.wav",
                "task-hw-fail",
                options=TranscribeOptions(terms=list(TRICKY_TERMS)),
            )

    path = observed["hotword"]
    assert path.endswith(".txt"), path
    assert observed["exists_during"] is True
    assert os.path.isfile(path) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["lock", "pool"])
@pytest.mark.parametrize("options", [None, TranscribeOptions(terms=[])])
async def test_transcribe_without_terms_keeps_empty_hotword(mode, options):
    seen = []
    tx = _ready_transcriber(mode, seen.append)
    duration, file_hash, release = _patch_io()
    with duration, file_hash, release, patch("tempfile.mkstemp") as mkstemp:
        await tx.transcribe("/fake/a.wav", "task-hw-empty", options=options)

    assert seen == [""]
    mkstemp.assert_not_called()


@pytest.mark.asyncio
async def test_hotword_file_write_failure_logs_marker():
    records = []
    sink = logger.add(records.append, level="ERROR")
    tx = _ready_transcriber("lock", lambda hotword: None)
    duration, file_hash, release = _patch_io()
    try:
        with duration, file_hash, release, \
             patch("tempfile.mkstemp", side_effect=OSError("disk full")), \
             pytest.raises(OSError, match="disk full"):
            await tx.transcribe(
                "/fake/a.wav",
                "task-hw-write",
                options=TranscribeOptions(terms=list(TRICKY_TERMS)),
            )
    finally:
        logger.remove(sink)

    text = "".join(map(str, records))
    assert "FUNASR-HOTWORD-FILE" in text
