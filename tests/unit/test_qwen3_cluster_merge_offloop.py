"""cluster_merge 必须在线程池执行，不能阻塞 asyncio 事件循环。"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from src.core.qwen3.asr import ASRResult
from src.models.schemas import TranscribeOptions


def _asr_result() -> ASRResult:
    return ASRResult(
        text="测试语音",
        items=[],
        chunks=[],
        duration=10.0,
        elapsed=0.1,
        rtf=0.01,
        peak_rss_mb=100.0,
        rss_delta_mb=0.0,
    )


def _transcriber():
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    return Qwen3DiarizeTranscriber(
        asr_model_dir="/fake/asr",
        segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx",
        cluster_merge_enabled=True,
        short_segment_guard_enabled=False,
        silence_align_enabled=False,
    )


def _install_outer_stubs(monkeypatch, transcriber) -> None:
    import src.core.qwen3_transcriber as module

    monkeypatch.setattr(module, "build_engine", lambda _path: object())
    monkeypatch.setattr(module, "run_asr", lambda *args, **kwargs: _asr_result())
    monkeypatch.setattr(
        module,
        "run_diarization_dispatched",
        lambda *args, **kwargs: [{"speaker": 0, "start": 0.0, "end": 10.0}],
    )
    monkeypatch.setattr(transcriber, "_ensure_embedding_extractor_fn", lambda: object())
    monkeypatch.setattr(module, "calculate_file_hash", AsyncMock(return_value="test-hash"))


@pytest.mark.asyncio
async def test_transcribe_runs_cluster_merge_off_loop_and_keeps_loop_scheduling(
    monkeypatch, tmp_path
):
    import src.core.qwen3_transcriber as module

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    transcriber = _transcriber()
    _install_outer_stubs(monkeypatch, transcriber)

    loop_thread_id = threading.get_ident()
    merge_started = threading.Event()
    merge_finished = threading.Event()
    observed_thread_ids = []
    loop_ticks_during_merge = 0

    def slow_cluster_merge(turns, _audio_path, _cfg, *, extractor_fn=None):
        observed_thread_ids.append(threading.get_ident())
        merge_started.set()
        try:
            time.sleep(0.08)
        finally:
            merge_finished.set()
        return turns, []

    monkeypatch.setattr(module, "apply_cluster_centroid_merge_to_turns", slow_cluster_merge)

    async def count_loop_ticks_during_merge():
        nonlocal loop_ticks_during_merge
        while not merge_started.is_set():
            await asyncio.sleep(0)
        while not merge_finished.is_set():
            loop_ticks_during_merge += 1
            await asyncio.sleep(0.001)

    ticker = asyncio.create_task(count_loop_ticks_during_merge())
    await transcriber.transcribe(
        str(audio),
        "cluster-offloop",
        options=TranscribeOptions(diarize=True, word_align=False),
    )
    await ticker

    assert observed_thread_ids, "transcribe 必须实际调用 cluster_merge"
    assert observed_thread_ids[0] != loop_thread_id
    assert loop_ticks_during_merge > 0, "cluster_merge 执行期间事件循环应继续调度协程"


@pytest.mark.asyncio
async def test_transcribe_logs_cluster_merge_failure_marker(monkeypatch, tmp_path):
    import src.core.qwen3_transcriber as module

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    transcriber = _transcriber()
    _install_outer_stubs(monkeypatch, transcriber)
    messages = []

    def failed_cluster_merge(*args, **kwargs):
        raise RuntimeError("synthetic cluster merge error")

    monkeypatch.setattr(module, "apply_cluster_centroid_merge_to_turns", failed_cluster_merge)
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        await transcriber.transcribe(
            str(audio),
            "cluster-failure-marker",
            options=TranscribeOptions(diarize=True, word_align=False),
        )
    finally:
        logger.remove(sink_id)

    assert any("QWEN3-CLUSTER-MERGE-FAILED" in message for message in messages)
