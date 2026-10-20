"""并行 ASR+diarize 任一失败时必须等两路结束再抛。

缺陷：asyncio.gather 默认在 diarize 抛错时立即抛出，ASR 的 executor 线程仍占用
engine；Qwen3InProcPool.finally 随即归还 tx，下一任务与孤儿线程并发使用同一
llama 上下文。本测试从公开入口 transcribe() 锁死：抛出原异常时 in-flight=0。
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest
from loguru import logger

from src.core.qwen3.asr import ASRResult
from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
from src.models.schemas import TranscribeOptions


class BoomDiarize(RuntimeError):
    """diarize 分支专用异常，钉死类型不被包装/改分类。"""


class BoomAsr(RuntimeError):
    """ASR 分支专用异常，钉死类型不被包装/改分类。"""


def _asr_ok() -> ASRResult:
    return ASRResult(
        text="hello",
        items=[],
        chunks=[],
        duration=1.0,
        elapsed=0.1,
        rtf=0.1,
        peak_rss_mb=1.0,
        rss_delta_mb=0.1,
    )


@pytest.fixture
def transcriber():
    return Qwen3DiarizeTranscriber(
        asr_model_dir="/fake/qwen3-asr",
        segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx",
        cluster_merge_enabled=False,
        silence_align_enabled=False,
        short_segment_guard_enabled=False,
        nospk_split_enabled=False,
        word_align_enabled=False,
    )


def _log_text(records) -> str:
    parts = []
    for rec in records:
        if hasattr(rec, "record"):
            parts.append(str(rec.record.get("message", rec)))
        else:
            parts.append(str(rec))
    return "".join(parts)


@pytest.mark.asyncio
async def test_diarize_failure_waits_for_slow_asr_then_reraises(transcriber, tmp_path):
    """diarize 立即抛错时，transcribe 抛出原异常前 ASR 线程必须已经结束。"""
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    inflight = {"asr": 0}
    asr_started = threading.Event()
    records = []
    sink = logger.add(records.append, level="ERROR")

    def slow_asr(*_a, **_k):
        inflight["asr"] += 1
        asr_started.set()
        try:
            time.sleep(0.5)
            return _asr_ok()
        finally:
            inflight["asr"] -= 1

    def boom_diarize(*_a, **_k):
        # 等 ASR 线程真正入场，再抛错，避免 diarize 先结束导致 in-flight 竞态假绿
        assert asr_started.wait(timeout=2.0), "ASR 线程未启动"
        raise BoomDiarize("diarize exploded")

    try:
        with patch("src.core.qwen3_transcriber.run_asr", side_effect=slow_asr), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", side_effect=boom_diarize), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()):
            with pytest.raises(BoomDiarize, match="diarize exploded"):
                await transcriber.transcribe(
                    audio_path=str(audio),
                    task_id="t-orphan-asr",
                    output_format="json",
                    options=TranscribeOptions(diarize=True),
                )
    finally:
        logger.remove(sink)

    assert inflight["asr"] == 0, "transcribe 抛出时 ASR executor 线程必须已经结束"
    assert "QWEN3-PARALLEL-BRANCH-FAILED" in _log_text(records)


@pytest.mark.asyncio
async def test_asr_failure_waits_for_slow_diarize_then_reraises(transcriber, tmp_path):
    """对称路径：ASR 立即抛错时，也必须等 diarize 结束再抛原异常。"""
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    inflight = {"diarize": 0}
    diarize_started = threading.Event()
    records = []
    sink = logger.add(records.append, level="ERROR")

    def boom_asr(*_a, **_k):
        assert diarize_started.wait(timeout=2.0), "diarize 线程未启动"
        raise BoomAsr("asr exploded")

    def slow_diarize(*_a, **_k):
        inflight["diarize"] += 1
        diarize_started.set()
        try:
            time.sleep(0.5)
            return [{"start": 0.0, "end": 1.0, "speaker": 0}]
        finally:
            inflight["diarize"] -= 1

    try:
        with patch("src.core.qwen3_transcriber.run_asr", side_effect=boom_asr), \
             patch("src.core.qwen3_transcriber.run_diarization_dispatched", side_effect=slow_diarize), \
             patch("src.core.qwen3_transcriber.build_engine", return_value=object()):
            with pytest.raises(BoomAsr, match="asr exploded"):
                await transcriber.transcribe(
                    audio_path=str(audio),
                    task_id="t-orphan-diarize",
                    output_format="json",
                    options=TranscribeOptions(diarize=True),
                )
    finally:
        logger.remove(sink)

    assert inflight["diarize"] == 0, "transcribe 抛出时 diarize executor 线程必须已经结束"
    assert "QWEN3-PARALLEL-BRANCH-FAILED" in _log_text(records)
