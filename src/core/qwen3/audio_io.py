"""轻量音频加载 — 16kHz 单声道 float32.

从 diarize.py 抽出 (DRY): diarize / diarize_ort / word_align sidecar 共用. 只依赖
soundfile + 惰性 librosa, **不碰 sherpa / ASR / diarize 机器**, 让 word_align
sidecar 能瘦入口 import 它 (codex #9, 不拖 sherpa).
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000


def load_audio_mono_16k(audio_path: str) -> tuple[np.ndarray, int]:
    """读音频并归一为 16kHz 单声道 float32.

    优先 soundfile (wav/flac/ogg); soundfile 不支持的格式 (m4a/aac 等)
    fallback 到 librosa.load (会自动调 ffmpeg/audioread 解码). 重采样惰性 import librosa.
    """
    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # 多声道取各声道平均混为单声道 — 与 librosa mono=True 及 Qwen3 ASR 侧
        # (vendor audio.py load_audio_numpy mean(axis=1)) 的混音方式一致;
        # 取单声道 [:, 0] 会丢掉双声道录音的右声道说话人。
        audio = audio.mean(axis=1)
    except Exception:
        # m4a/aac/mp3 等 soundfile 不支持的格式
        import librosa

        audio, _sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        return audio.astype(np.float32), TARGET_SAMPLE_RATE

    if sample_rate != TARGET_SAMPLE_RATE:
        import librosa

        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        )
        sample_rate = TARGET_SAMPLE_RATE
    return audio, sample_rate
