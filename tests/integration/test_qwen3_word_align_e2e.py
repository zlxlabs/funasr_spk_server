"""
词级时间戳 — Qwen3 word_align 端到端集成测试

默认 skip, 需 FUNASR_RUN_INTEGRATION=1 (加载 Qwen3-ASR ~2GB + sherpa diarize +
MMS CTC-FA ~1.2GB).

覆盖:
1. parity (flag 关): word_align off → 段 words 全 None, ASR/diarize 输出不变.
2. fallback: word_align on 但 aligner 抛错 → 段照常出 (words=None), 不崩.
3. 真 MMS: word_align on + 真 MMS → podcast 60s 出 words, 词时间落在所属段窗内.
4. 轻量精度基准: 词覆盖率 (有词段占比) 不低于阈值 (AAS 思路, 复用 PoC compare).

MMS 模型路径: 优先 config 默认 (models/qwen3_diarize/ctc_forced_aligner/model.onnx),
回退 deskpai 缓存 ~/ctc_forced_aligner/model.onnx; 都缺则 skip 真 MMS 用例.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="设 FUNASR_RUN_INTEGRATION=1 启用 (默认 skip, 加载 Qwen3 + MMS 大模型)",
)


def _resolve_mms_model() -> str | None:
    """找可用的 MMS ONNX 路径, 都缺返回 None (跳过真 MMS 用例)."""
    candidates = [
        Path("models/qwen3_diarize/ctc_forced_aligner/model.onnx"),
        Path.home() / "ctc_forced_aligner" / "model.onnx",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _build_transcriber(word_align_enabled: bool, model_path: str | None = None):
    from src.core.config import config
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    q = config.qwen3
    return Qwen3DiarizeTranscriber(
        asr_model_dir=q.asr_model_dir,
        segmentation_model=q.segmentation_model,
        embedding_model=q.embedding_model,
        num_speakers=q.num_speakers,
        cluster_threshold=q.cluster_threshold,
        num_threads=q.num_threads,
        provider=q.provider,
        language=q.language,
        temperature=q.temperature,
        word_align_enabled=word_align_enabled,
        word_align_language="chi",
        word_align_model_path=model_path or q.word_align_model_path,
        word_align_provider="cpu",
        word_align_batch_size=16,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_word_align_off_segments_have_no_words(podcast_audio: Path):
    """flag 关: 段 words 全 None (parity — 老路不变)."""
    tx = _build_transcriber(word_align_enabled=False)
    await tx.initialize()
    result, raw = await tx.transcribe(
        audio_path=str(podcast_audio), task_id="wa-off", output_format="json"
    )
    assert len(result.segments) > 0
    assert all(s.words is None for s in result.segments)
    assert raw["word_align"]["enabled"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_word_align_fallback_on_align_error(podcast_audio: Path):
    """word_align on 但对齐抛错 → 段照常出, 不崩, words=None."""
    from src.models.schemas import TranscribeOptions

    tx = _build_transcriber(word_align_enabled=True, model_path="/nonexistent/mms.onnx")
    await tx.initialize()
    # _ensure_word_aligner 会 build 失败 (模型缺) → 进 except, 段仍出
    # 决策 1A: transcribe 读 options.word_align (effective 值), 不读 config;
    # 直调 transcribe 绕过 resolve_word_align, 故必须显式传 word_align=True.
    result, raw = await tx.transcribe(
        audio_path=str(podcast_audio), task_id="wa-fallback", output_format="json",
        options=TranscribeOptions(word_align=True),
    )
    assert len(result.segments) > 0
    assert all(s.words is None for s in result.segments)
    assert raw["word_align"]["enabled"] is True
    assert "error" in raw["word_align"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_word_align_real_mms_produces_words(podcast_audio: Path):
    """真 MMS: podcast 60s 出 words, 词时间落在所属段窗内, 覆盖率达阈值."""
    model_path = _resolve_mms_model()
    if model_path is None:
        pytest.skip("MMS 模型缺失, 跑 scripts/download_qwen3_models.sh --word-align 后再测")

    from src.models.schemas import TranscribeOptions

    tx = _build_transcriber(word_align_enabled=True, model_path=model_path)
    await tx.initialize()
    result, raw = await tx.transcribe(
        audio_path=str(podcast_audio), task_id="wa-real", output_format="json",
        # 决策 1A: word_align 是 per-request effective 值, config 的 word_align_enabled
        # 只在 resolve_word_align 里当兜底; 直调 transcribe 必须显式传.
        options=TranscribeOptions(language="chi", word_align=True),
    )

    assert raw["word_align"]["enabled"] is True
    assert "error" not in raw["word_align"], f"word_align 异常: {raw['word_align']}"
    assert raw["word_align"]["total_words"] > 0, "真 MMS 应出词"

    # 词时间落在所属段窗内 (容差 50ms, snap/relabel 不动词)
    segs_with_words = [s for s in result.segments if s.words]
    assert segs_with_words, "至少一段挂到词"
    for s in segs_with_words:
        for w in s.words:
            assert w.start >= s.start_time - 0.05, f"词 {w.text} start={w.start} < 段 {s.start_time}"
            assert w.end <= s.end_time + 0.05, f"词 {w.text} end={w.end} > 段 {s.end_time}"
            assert w.end >= w.start

    # 轻量精度基准: 有词段占比 (AAS 思路 — 覆盖率不退化)
    coverage = len(segs_with_words) / len(result.segments)
    assert coverage >= 0.5, f"词覆盖率过低: {coverage:.2f} ({len(segs_with_words)}/{len(result.segments)})"
