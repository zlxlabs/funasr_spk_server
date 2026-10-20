"""音频时长探测缺少容器时长及探测失败时的公开行为。"""
import shutil
import subprocess
import threading
from unittest.mock import MagicMock

import pytest
from loguru import logger

from src.utils import file_utils


def _mock_decode(monkeypatch, stderr=b"frame=  75 fps=0.0 time=00:00:03.00 bitrate=N/A\n"):
    """模拟 ffmpeg null 输出中的最终解码时间。"""
    ffmpeg_input = MagicMock()
    ffmpeg_output = MagicMock()
    ffmpeg_input.output.return_value = ffmpeg_output
    ffmpeg_output.run.return_value = (b"", stderr)
    monkeypatch.setattr(file_utils.ffmpeg, "input", lambda _path: ffmpeg_input)
    return ffmpeg_input, ffmpeg_output


def test_webm_without_container_duration_uses_decoded_duration(tmp_path, monkeypatch):
    """无 ffmpeg CLI 时 mock 解码边界；有 CLI 时生成真实无 Duration WebM。"""
    ffmpeg_binary = shutil.which("ffmpeg")
    ffprobe_binary = shutil.which("ffprobe")
    audio_path = tmp_path / "durationless.webm"

    if ffmpeg_binary and ffprobe_binary:
        subprocess.run(
            [
                ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:duration=3",
                "-c:a",
                "libopus",
                "-f",
                "webm",
                "-live",
                "1",
                "-y",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        probe = file_utils.ffmpeg.probe(str(audio_path))
        assert "duration" not in probe.get("format", {})
        assert all("duration" not in stream for stream in probe.get("streams", []))
    else:
        audio_path.touch()
        monkeypatch.setattr(
            file_utils.ffmpeg,
            "probe",
            lambda _path: {
                "format": {"format_name": "matroska,webm"},
                "streams": [{"codec_type": "audio", "codec_name": "opus"}],
            },
        )
        _mock_decode(monkeypatch)

    assert file_utils.get_audio_duration(str(audio_path)) == pytest.approx(3.0, abs=0.1)


def test_audio_stream_duration_is_used_before_decoding(monkeypatch):
    monkeypatch.setattr(
        file_utils.ffmpeg,
        "probe",
        lambda _path: {
            "format": {"format_name": "webm"},
            "streams": [
                {"codec_type": "video", "duration": "99.0"},
                {"codec_type": "audio", "duration": "3.25"},
            ],
        },
    )
    decode_input, _ = _mock_decode(monkeypatch)

    assert file_utils.get_audio_duration("audio.webm") == pytest.approx(3.25)
    decode_input.output.assert_not_called()


def test_probe_failure_raises_and_logs_audio_duration_marker(monkeypatch):
    records = []
    sink_id = logger.add(records.append, level="ERROR")
    monkeypatch.setattr(
        file_utils.ffmpeg,
        "probe",
        MagicMock(side_effect=OSError("ffprobe executable missing")),
    )

    try:
        with pytest.raises(RuntimeError, match="时长探测失败") as exc_info:
            file_utils.get_audio_duration("audio.webm")
    finally:
        logger.remove(sink_id)

    assert "ffprobe executable missing" in str(exc_info.value)
    assert "AUDIO-DURATION-PROBE-FAILED" in "".join(map(str, records))


def test_decode_failure_raises_and_logs_audio_duration_marker(monkeypatch):
    monkeypatch.setattr(
        file_utils.ffmpeg,
        "probe",
        lambda _path: {"format": {}, "streams": [{"codec_type": "audio"}]},
    )
    _, ffmpeg_output = _mock_decode(monkeypatch)
    ffmpeg_output.run.side_effect = OSError("ffmpeg executable missing")

    with pytest.raises(RuntimeError, match="时长探测失败") as exc_info:
        file_utils.get_audio_duration("audio.webm")

    assert "ffmpeg executable missing" in str(exc_info.value)


@pytest.mark.asyncio
async def test_transcriber_probe_failure_is_not_reported_as_short_audio(monkeypatch):
    from src.core.funasr_transcriber import FunASRTranscriber

    tx = FunASRTranscriber.__new__(FunASRTranscriber)
    tx.is_initialized = True
    tx.concurrency_mode = "lock"
    tx.config = {"funasr": {"batch_size_s": 300}, "transcription": {}}
    tx.model = MagicMock()
    tx._model_lock = threading.Lock()
    monkeypatch.setattr(
        file_utils.ffmpeg,
        "probe",
        MagicMock(side_effect=OSError("ffprobe unavailable")),
    )

    with pytest.raises(Exception) as exc_info:
        await tx.transcribe("audio.webm", "duration-probe-failure")

    assert "时长探测失败" in str(exc_info.value)
    assert "音频时长过短" not in str(exc_info.value)
    tx.model.generate.assert_not_called()
