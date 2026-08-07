"""I2 Qwen context 契约与两套 transport 的 terms 穿透。"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.qwen3.asr import ASRResult, _run_asr_loaded_audio, build_qwen_context
from src.core.result_projection import build_result_metadata
from src.models.schemas import TranscribeOptions


def _asr_result() -> ASRResult:
    return ASRResult(
        text="测试文本", items=[], chunks=[], duration=1.0, elapsed=0.1,
        rtf=0.1, peak_rss_mb=1.0, rss_delta_mb=0.0,
    )


def test_qwen_context_template_is_exact_and_ordered():
    assert build_qwen_context([]) is None
    assert build_qwen_context(["术语 A", "术语 B"]) == (
        "You are a helpful assistant.\nKnown terms:\n术语 A\n术语 B"
    )


def test_qwen_asr_passes_context_to_existing_vendor_argument():
    result = SimpleNamespace(
        text="ok", alignment=SimpleNamespace(items=[]), chunks=[],
    )
    engine = MagicMock(config=SimpleNamespace(chunk_size=40.0, memory_num=1))
    engine.asr.return_value = result

    _run_asr_loaded_audio(
        [0] * 16000, engine, language="Chinese",
        context=build_qwen_context(["Alpha"]),
    )

    assert engine.asr.call_args.kwargs["context"] == (
        "You are a helpful assistant.\nKnown terms:\nAlpha"
    )


@pytest.fixture
def qwen_transcriber():
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    return Qwen3DiarizeTranscriber(
        asr_model_dir="/fake/asr", segmentation_model="/fake/seg",
        embedding_model="/fake/emb", cluster_merge_enabled=False,
        silence_align_enabled=False, nospk_split_enabled=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("terms", [[], ["Alpha", "Beta"]])
@pytest.mark.parametrize("diarize", [True, False])
@pytest.mark.parametrize("output_format", ["json", "srt"])
@pytest.mark.parametrize("word_align", [False, True])
async def test_qwen_transcriber_context_matrix(
    qwen_transcriber, tmp_path, terms, diarize, output_format, word_align,
):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"\0")
    captured = {}

    def fake_run_asr(*args, **kwargs):
        captured["context"] = kwargs["context"]
        return _asr_result()

    def fake_word_align(_path, _chunks, segments, _language, _task_id):
        return segments, {"enabled": True, "total_words": 0}

    with patch("src.core.qwen3_transcriber.run_asr", side_effect=fake_run_asr), \
         patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
         patch("src.core.qwen3_transcriber.calculate_file_hash", new=AsyncMock(return_value="h")), \
         patch("src.core.qwen3_transcriber.run_diarization_dispatched", return_value=[
             {"start": 0.0, "end": 1.0, "speaker": 0},
         ]), \
         patch.object(qwen_transcriber, "_word_align_segments", side_effect=fake_word_align):
        result = await qwen_transcriber.transcribe(
            str(audio), "t-matrix", output_format=output_format,
            options=TranscribeOptions(
                terms=terms, diarize=diarize, word_align=word_align,
            ),
        )

    assert captured["context"] == build_qwen_context(terms)
    if output_format == "json":
        transcription, _raw = result
        if diarize:
            assert transcription.segments[0].speaker == "Speaker1"
        else:
            assert transcription.speakers == []
            assert transcription.segments[0].speaker is None
    else:
        assert result["format"] == "srt"


@pytest.mark.asyncio
async def test_qwen_file_pool_serializes_nonempty_terms():
    from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

    pool = MagicMock()
    pool.generate_with_pool = AsyncMock(return_value={"format": "srt", "content": "x"})
    wrapper = Qwen3PoolTranscriber(pool_size=1, pool=pool)
    options = TranscribeOptions(terms=["Alpha", "Beta"])
    await wrapper.transcribe("a.wav", "t", output_format="srt", options=options)
    fields = pool.generate_with_pool.call_args.kwargs["extra_task_fields"]
    assert fields["options"]["terms"] == ["Alpha", "Beta"]
    json.dumps(fields)


@pytest.mark.asyncio
async def test_qwen_inproc_pool_forwards_same_options_object():
    from src.core.qwen3_inproc_pool import Qwen3InProcPool

    tx = MagicMock(initialize=AsyncMock(), transcribe=AsyncMock(return_value="ok"))
    pool = Qwen3InProcPool(pool_size=1, transcriber_factory=lambda: tx)
    options = TranscribeOptions(terms=["Alpha"])
    await pool.transcribe("a.wav", "t", options=options)
    assert tx.transcribe.call_args.kwargs["options"] is options
    assert tx.transcribe.call_args.kwargs["options"].terms == ["Alpha"]


def test_qwen_context_metadata_is_success_only_and_engine_specific():
    qwen = build_result_metadata(engine="qwen3", options=TranscribeOptions(terms=["Alpha"]))
    empty = build_result_metadata(engine="qwen3", options=TranscribeOptions())
    funasr = build_result_metadata(engine="funasr", options=TranscribeOptions(terms=["Alpha"]))
    assert qwen["context_applied"] is True and qwen["terms_count"] == 1
    assert "context_applied" not in empty
    assert "context_applied" in funasr
