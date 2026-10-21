"""GBK 元数据不能阻断 ffmpeg 静音区间解析的回归测试。"""
from __future__ import annotations

import subprocess
from unittest.mock import patch

from src.utils.silence_detect import ffmpeg_speech_regions


def test_ffmpeg_speech_regions_replaces_gbk_metadata_and_parses_silence():
    """从公开入口经过 subprocess 的文本解码，GBK 元数据应不影响数值解析。"""
    stderr_bytes = (
        b"Input metadata title: "
        + "中文标题".encode("gbk")
        + b"\n[Parsed_silencedetect_0 @ 0x1] silence_start: 1.000\n"
        + b"[Parsed_silencedetect_0 @ 0x1] silence_end: 2.000 | silence_duration: 1.000\n"
    )

    def decode_ffmpeg_output(command, **kwargs):
        """模拟 subprocess.run(text=True) 按其 errors 参数解码 stderr 字节。"""
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs.get("errors") == "replace"
        stderr_text = stderr_bytes.decode("utf-8", errors=kwargs["errors"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=stderr_text)

    with patch("src.utils.silence_detect.subprocess.run", side_effect=decode_ffmpeg_output):
        regions = ffmpeg_speech_regions("audio.wav", audio_duration=4.0)

    assert regions == [
        {"start": 0.0, "end": 1.0},
        {"start": 2.0, "end": 4.0},
    ]
