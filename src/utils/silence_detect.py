"""ffmpeg silencedetect 包装 — 轻量 VAD 边界检测.

用途: Qwen3 silence-aware 段切点对齐 (snap_segments_to_silence) 的输入,
也可供其他需要静音/语音区段切分的模块复用.

设计要点:
  - 纯函数, 无类无 side effect, 易测试
  - 仅依赖 ffmpeg 二进制和标准库, 不引入其他包
  - 返回 "speech_regions" (非 silence), 与 snap-to-silence 算法接口对齐

参数 sweet spot (见 spikes/qwen3_silence_align/SUMMARY.md):
  - podcast 类音频: noise_db="-25dB", min_silence_sec=0.2
  - FunASR 历史默认: noise_db="-35dB", min_silence_sec=0.8 (podcast 0 检出, 不推荐)

PoC 来源: tests/manual/server/qwen3_long_audio_poc.py:ffmpeg_speech_regions
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def ffmpeg_speech_regions(
    audio_path: str | Path,
    audio_duration: float,
    noise_db: str = "-25dB",
    min_silence_sec: float = 0.20,
) -> list[dict]:
    """跑 ffmpeg silencedetect, 把 silence 反向解析成 speech_regions.

    Args:
        audio_path: 音频文件路径 (任意 ffmpeg 可读格式).
        audio_duration: 音频总时长 (秒). 用于合成尾部 region.
        noise_db: silencedetect 噪声门限 (e.g. "-25dB"). 越敏感 (绝对值越小) 检出 silence 越多.
        min_silence_sec: 最短 silence 时长门限 (秒). 短于此值不算 silence.

    Returns:
        list[dict]: 每项含 {"start": float, "end": float}, 按 start 升序, 已过滤 < 0.1s 的极短 region.
        speech_regions = audio_duration 区间内的所有 "非 silence" 区段.

    Raises:
        RuntimeError: ffmpeg 调用失败 (返回 stderr 后 2000 字符方便排错).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_db}:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise RuntimeError(stderr[-2000:])

    silence_starts: list[float] = []
    silences: list[tuple[float, float]] = []
    for line in stderr.splitlines():
        m_start = re.search(r"silence_start: ([0-9.]+)", line)
        if m_start:
            silence_starts.append(float(m_start.group(1)))
            continue
        m_end = re.search(r"silence_end: ([0-9.]+)", line)
        if m_end and silence_starts:
            start = silence_starts.pop(0)
            end = float(m_end.group(1))
            silences.append((start, end))
    # 文件尾未闭合的 silence (silencedetect 不输出尾部 silence_end), 补到 audio 末尾
    for start in silence_starts:
        silences.append((start, audio_duration))

    regions: list[dict] = []
    cursor = 0.0
    for start, end in silences:
        if start > cursor:
            regions.append({"start": cursor, "end": start})
        cursor = max(cursor, end)
    if cursor < audio_duration:
        regions.append({"start": cursor, "end": audio_duration})

    return [r for r in regions if r["end"] - r["start"] > 0.1]
