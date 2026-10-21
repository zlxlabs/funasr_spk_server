"""第三批 Qwen3 转写器回归测试：sidecar 显存预检、language 规范化和池工厂参数。"""
from __future__ import annotations

import asyncio
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.qwen3.asr import ASRChunkItem, ASRResult
from src.models.schemas import TranscribeOptions


def _asr_result() -> ASRResult:
    """构造不依赖真实模型的公开转写入口响应。"""
    return ASRResult(
        text="你好",
        items=[],
        chunks=[ASRChunkItem(text="你好", start=0.0, end=2.0, index=0)],
        duration=2.0,
        elapsed=0.1,
        rtf=0.05,
        peak_rss_mb=100.0,
        rss_delta_mb=1.0,
    )


def _make_transcriber(**overrides):
    """创建关闭非目标后处理的轻量转写器。"""
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    kwargs = {
        "asr_model_dir": "/fake/asr",
        "segmentation_model": "/fake/seg.onnx",
        "embedding_model": "/fake/emb.onnx",
        "cluster_merge_enabled": False,
        "silence_align_enabled": False,
        "short_segment_guard_enabled": False,
        "word_align_enabled": True,
        "word_align_sidecar_enabled": True,
        "word_align_provider": "cuda",
    }
    kwargs.update(overrides)
    return Qwen3DiarizeTranscriber(**kwargs)


def _sidecar_client(words=None):
    """构造 sidecar 网络边界替身。"""
    client = MagicMock()
    client.align.return_value = (
        words or [{"text": "你好", "start": 0.2, "end": 1.0, "score": -0.1}],
        {"total_windows": 1, "failed_windows": 0, "total_words": 1, "failures": []},
    )
    return client


@pytest.mark.asyncio
async def test_preflight_counts_free_vram_after_excluding_live_sidecar_usage(tmp_path):
    """sidecar 已占用的显存应从全卡占用中扣除，热复用仍可过预检。"""
    tx = _make_transcriber(word_align_preflight_free_mib=4608)
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 32000)
    client = _sidecar_client()
    client.running_pid = 27182

    with patch("src.core.qwen3_transcriber.free_vram_mib", return_value=1200), \
         patch(
             "src.core.qwen3_transcriber.process_vram_mib", return_value=4096
         ) as process_probe, \
         patch.object(
             tx, "_ensure_word_aligner_cpu",
             return_value=SimpleNamespace(
                 align_chunks=MagicMock(
                     return_value=(
                         [],
                         {
                             "total_windows": 0,
                             "failed_windows": 0,
                             "total_words": 0,
                             "failures": [],
                         },
                     )
                 )
             ),
         ), \
         patch("src.core.qwen3_transcriber.get_word_align_sidecar_client", return_value=client):
        with patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.run_asr", return_value=_asr_result()), \
             patch(
                 "src.core.qwen3_transcriber.run_diarization_dispatched",
                 return_value=[{"start": 0.0, "end": 2.0, "speaker": 0}],
             ), \
             patch(
                 "src.core.qwen3_transcriber.calculate_file_hash",
                 new=AsyncMock(return_value="audio-hash"),
             ):
            await tx.transcribe(
                str(audio), "task-sidecar-vram", output_format="json",
                options=TranscribeOptions(word_align=True, language="zh"),
            )

    client.align.assert_called_once()
    process_probe.assert_called_once_with(27182)


@pytest.mark.asyncio
async def test_language_whitespace_matches_cache_tag_and_alignment_language(tmp_path):
    """带空白和规范 language 的缓存维度一致，送给对齐器的值也一致。"""
    from src.core.database import cache_params

    tx = _make_transcriber(word_align_preflight_enabled=False)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    client = _sidecar_client()

    with patch("src.core.qwen3_transcriber.get_word_align_sidecar_client", return_value=client):
        with patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
             patch("src.core.qwen3_transcriber.run_asr", return_value=_asr_result()), \
             patch(
                 "src.core.qwen3_transcriber.run_diarization_dispatched",
                 return_value=[{"start": 0.0, "end": 2.0, "speaker": 0}],
             ), \
             patch(
                 "src.core.qwen3_transcriber.calculate_file_hash",
                 new=AsyncMock(return_value="audio-hash"),
             ):
            for language in ("  zh ", "zh"):
                await tx.transcribe(
                    str(audio), f"task-{language!r}", output_format="json",
                    options=TranscribeOptions(word_align=True, language=language),
                )

    spaced_tag = cache_params(
        "qwen3", TranscribeOptions(word_align=True, language="  zh "), "json",
    )[0]
    normalized_tag = cache_params(
        "qwen3", TranscribeOptions(word_align=True, language="zh"), "json",
    )[0]
    assert spaced_tag == normalized_tag == "qwen3+wa:zh"
    assert [call.args[2] for call in client.align.call_args_list] == ["zh", "zh"]


def test_inproc_pool_factory_matches_singleton_constructor_parameters(monkeypatch):
    """池初始化走公开入口时，实例构造参数应与单例工厂完整一致。"""
    from src.core.config import config
    from src.core.qwen3_inproc_pool import Qwen3InProcPool
    from src.core.qwen3_transcriber import (
        get_qwen3_transcriber,
        reset_qwen3_transcriber_singleton,
    )

    class FakeTranscriber:
        def __init__(self, **kwargs):
            self.constructor_kwargs = kwargs
            self.embedding_model = None

        async def initialize(self):
            return None

    reset_qwen3_transcriber_singleton()
    monkeypatch.setattr("src.core.qwen3_transcriber.Qwen3DiarizeTranscriber", FakeTranscriber)
    try:
        singleton = get_qwen3_transcriber()
        pool = Qwen3InProcPool(pool_size=1)
        asyncio.run(pool.initialize())
        pooled = pool._txs[0]

        assert singleton.constructor_kwargs == pooled.constructor_kwargs
        assert (
            pooled.constructor_kwargs["word_align_cuda_batch_size"]
            == config.qwen3.word_align_cuda_batch_size
        )
        assert (
            pooled.constructor_kwargs["word_align_preflight_enabled"]
            == config.qwen3.word_align_preflight_enabled
        )
        assert (
            pooled.constructor_kwargs["word_align_preflight_free_mib"]
            == config.qwen3.word_align_preflight_free_mib
        )
        assert (
            pooled.constructor_kwargs["word_align_sidecar_enabled"]
            == config.qwen3.word_align_sidecar_enabled
        )
    finally:
        reset_qwen3_transcriber_singleton()
