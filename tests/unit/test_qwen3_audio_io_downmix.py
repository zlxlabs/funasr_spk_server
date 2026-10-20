"""Qwen3 audio_io: 多声道读取必须平均混为单声道 (全仓审查 261020 卡 4b).

修复前: soundfile 路径 audio[:, 0] 只取第一声道, 双声道通话录音的右声道说话人
在 diarize 输入里不存在 → 切不出 turn, 叠加 merge 缺陷整段转写为空.
修复后: audio.mean(axis=1), 与 librosa mono=True 路径及 Qwen3 ASR 侧
vendor audio.py load_audio (mean(axis=1)) 的混音方式一致.
"""
import numpy as np
import soundfile as sf

from src.core.qwen3.audio_io import TARGET_SAMPLE_RATE, load_audio_mono_16k


def test_stereo_left_silent_right_signal_downmixed(tmp_path):
    """左静音+右有信号的双声道 wav: 混音后 RMS>0 且样本等于两声道平均.

    修复前 (audio[:, 0] 取第一声道): 输出全是静音 → RMS==0, 此处红.
    用 FLOAT wav 免量化误差, 「等于两声道平均」可断言到样本级.
    """
    sr = TARGET_SAMPLE_RATE  # 16k, 免重采样, 直接命中 soundfile 路径
    t = np.arange(sr, dtype=np.float64) / sr
    left = np.zeros(sr, dtype=np.float32)
    right = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo_left_silent.wav"
    sf.write(str(path), stereo, sr, subtype="FLOAT")

    audio, out_sr = load_audio_mono_16k(str(path))

    assert out_sr == TARGET_SAMPLE_RATE
    np.testing.assert_allclose(audio, (left + right) / 2, atol=1e-6)
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms > 0, "右声道信号丢失: 混音后仍是静音"


def test_mono_passthrough_not_regression(tmp_path):
    """单声道回归保护: (N,1) mean(axis=1) 与原 [:, 0] 等价, 输出不变."""
    sr = TARGET_SAMPLE_RATE
    t = np.arange(sr, dtype=np.float64) / sr
    mono = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    path = tmp_path / "mono.wav"
    sf.write(str(path), mono.reshape(-1, 1), sr, subtype="FLOAT")

    audio, out_sr = load_audio_mono_16k(str(path))

    assert out_sr == TARGET_SAMPLE_RATE
    np.testing.assert_allclose(audio, mono, atol=1e-6)
