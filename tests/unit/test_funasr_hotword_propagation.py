"""I1-T3 FunASR hotword 参数链与 worker 日志边界。"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.schemas import TranscribeOptions


class _RecordingLock:
    def __init__(self):
        self.active = False

    def __enter__(self):
        self.active = True

    def __exit__(self, *_):
        self.active = False


def _build_transcriber(mode, hotword_seen):
    from src.core.funasr_transcriber import FunASRTranscriber

    tx = FunASRTranscriber.__new__(FunASRTranscriber)
    tx.is_initialized = True
    tx.concurrency_mode = mode
    tx.config = {"funasr": {"batch_size_s": 300}, "transcription": {}}
    result = [{"sentence_info": [{"start": 0, "end": 1000, "text": "ok", "spk": 0}]}]
    model = MagicMock()

    def generate(**kwargs):
        hotword_seen.append(kwargs["hotword"])
        if mode == "lock":
            assert tx._model_lock.active is True
        return result

    model.generate.side_effect = generate
    if mode == "pool":
        tx.model_pool = MagicMock()
        tx.model_pool.generate_with_pool = AsyncMock(return_value=result)
    else:
        tx.model = model
        tx._model_lock = _RecordingLock()
    return tx


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "terms", "expected"),
    [("pool", ["Alpha", "Beta"], "Alpha Beta"), ("pool", [], ""), ("lock", ["Alpha", "Beta"], "Alpha Beta")],
)
async def test_funasr_hotword_reaches_pool_or_lock(mode, terms, expected):
    seen = []
    tx = _build_transcriber(mode, seen)
    with patch("src.core.funasr_transcriber.get_audio_duration", return_value=1.0), \
         patch("src.utils.file_utils.calculate_file_hash", new=AsyncMock(return_value="h")), \
         patch("src.core.funasr_transcriber.release_accelerator_memory"):
        await tx.transcribe("/fake/a.wav", "t", options=TranscribeOptions(terms=terms))

    if mode == "pool":
        call = tx.model_pool.generate_with_pool.call_args.kwargs
        assert call["hotword"] == expected
        assert "extra_task_fields" not in call
    else:
        assert seen == [expected]


@pytest.mark.asyncio
async def test_funasr_pool_task_json_keeps_hotword_top_level(tmp_path):
    from src.core.file_based_process_pool import FileBasedProcessPool

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    task_dir = tmp_path / "tasks"
    pool = FileBasedProcessPool(pool_size=1, task_dir=str(task_dir))
    pool.is_initialized = True
    process = MagicMock()
    process.poll.return_value = None
    pool.worker_processes = [process]
    pool._ensure_workers_alive = AsyncMock()
    pool._calculate_timeout = MagicMock(return_value=1.0)

    with patch("src.core.file_based_process_pool.asyncio.sleep", side_effect=RuntimeError("stop")), \
         pytest.raises(RuntimeError, match="stop"):
        await pool.generate_with_pool(str(audio), hotword="Alpha Beta")

    task_file = next(task_dir.glob("*.task"))
    payload = json.loads(task_file.read_text(encoding="utf-8"))
    assert payload["hotword"] == "Alpha Beta"
    assert "options" not in payload


def test_funasr_worker_passes_hotword_and_redacts_log(tmp_path, capsys):
    from src.core import worker_process as wp

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    task_file = tmp_path / "worker_0_hotword.task"
    task_file.write_text(json.dumps({
        "task_id": "hotword", "audio_path": str(audio), "source_audio_path": str(audio),
        "batch_size_s": 300, "hotword": "Alpha Beta", "use_pickle": True,
    }), encoding="utf-8")
    model = MagicMock()
    model.generate.return_value = []

    with patch.object(wp, "release_accelerator_memory"):
        wp.process_task(0, model, str(task_file), str(tmp_path))

    model.generate.assert_called_once_with(input=str(audio), batch_size_s=300, hotword="Alpha Beta")
    output = capsys.readouterr().out
    assert "Alpha Beta" not in output
    assert "hotword_length=10" in output
