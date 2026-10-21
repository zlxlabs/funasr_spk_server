"""回归锁定 Qwen3 分段、词挂段及 ORT 聚类参数缺陷。"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from src.core.qwen3.merge import Segment


def _old_attach_reference(segments: list[Segment], words: list[dict]) -> list[list[dict] | None]:
    """复刻原逐词逐段算法，作为非零时长词语义 oracle。"""
    buckets: list[list[dict] | None] = [None] * len(segments)
    for word in words:
        word_start = float(word["start"])
        word_end = float(word["end"])
        best_idx = -1
        best_overlap = 0.0
        for index, segment in enumerate(segments):
            overlap = min(segment.end, word_end) - max(segment.start, word_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = index
        if best_idx >= 0:
            if buckets[best_idx] is None:
                buckets[best_idx] = []
            buckets[best_idx].append({
                "text": word["text"], "start": word_start,
                "end": word_end, "score": word.get("score"),
            })
    return buckets


def test_default_same_speaker_merge_has_a_span_cap() -> None:
    """公开 guard 默认不得把连续独白合成超过已有 120 秒上限的巨段。"""
    from src.core.qwen3.postprocess import apply_short_segment_guard

    segments = [
        {"start": 0.0, "end": 60.0, "speaker": "0", "text": "一"},
        {"start": 60.0, "end": 120.0, "speaker": "0", "text": "二"},
        {"start": 120.0, "end": 180.0, "speaker": "0", "text": "三"},
    ]
    out, stats = apply_short_segment_guard(segments)

    assert [(s["start"], s["end"]) for s in out] == [(0.0, 120.0), (120.0, 180.0)]
    assert stats["merge"]["merged"] == 1


def test_merge_chunks_output_is_stably_sorted_by_start_and_end() -> None:
    """公开融合入口应对同起点的 turn 按 end 排序，而非沿用输入乱序。"""
    from src.core.qwen3.merge import merge_asr_and_diarize, merge_asr_chunks_and_diarize

    @dataclass
    class Chunk:
        """为 merge 公开入口提供的 ASR chunk 样本。"""
        text: str
        start: float
        end: float

    segments = merge_asr_chunks_and_diarize(
        [Chunk("abcdefghij", 0.0, 10.0)],
        [
            {"start": 0.0, "end": 8.0, "speaker": 0},
            {"start": 0.0, "end": 5.0, "speaker": 1},
        ],
    )

    assert [(s.start, s.end) for s in segments] == [(0.0, 5.0), (0.0, 8.0)]

    text_segments = merge_asr_and_diarize(
        "abcdefghij",
        [
            {"start": 0.0, "end": 8.0, "speaker": 0},
            {"start": 0.0, "end": 5.0, "speaker": 1},
        ],
    )
    assert [(s.start, s.end) for s in text_segments] == [(0.0, 5.0), (0.0, 8.0)]

    out_of_order = merge_asr_chunks_and_diarize(
        [Chunk("第一", 0.0, 0.8), Chunk("第二", 0.2, 0.4)],
        [{"start": 0.5, "end": 0.8, "speaker": 0}],
    )
    assert [(s.start, s.end) for s in out_of_order] == [(0.2, 0.4), (0.5, 0.8)]


def test_split_long_segments_never_chooses_cut_beyond_max_duration() -> None:
    """公开切段入口不能因优先选择静音候选而生成超 max_dur 的片段。"""
    from src.core.qwen3.segment_split import split_long_segments

    segment = Segment(start=0.0, end=12.0, speaker=0, text="甲乙丙丁戊己庚辛壬癸子丑")
    speech_regions = [
        {"start": 0.0, "end": 4.8},
        {"start": 5.2, "end": 7.8},
        {"start": 8.2, "end": 12.0},
    ]
    out, _stats = split_long_segments(
        [segment], speech_regions=speech_regions, audio_duration=12.0,
        max_dur=4.0, min_dur=1.0,
    )

    assert len(out) >= 3
    assert all(piece.end - piece.start <= 4.0 + 1e-9 for piece in out)
    assert "".join(piece.text for piece in out) == segment.text


def test_non_positive_num_speakers_uses_automatic_clustering(tmp_path, monkeypatch) -> None:
    """公开 ORT 入口把非正人数作为自动聚类，而非 maxclust 固定簇数。"""
    import soundfile as sf

    from src.core.qwen3 import diarize_ort

    audio_path = tmp_path / "automatic-clustering.wav"
    sf.write(str(audio_path), np.ones(16000 * 12, dtype=np.float32) * 0.1, 16000)
    chunk0 = np.zeros((589, 3), dtype=np.int8)
    chunk0[:300, 0] = 1
    chunk1 = np.zeros((589, 3), dtype=np.int8)
    chunk1[200:, 1] = 1
    monkeypatch.setattr(
        diarize_ort,
        "run_segmentation_chunks",
        lambda _audio, _session: ([0, 16000], [chunk0, chunk1]),
    )
    monkeypatch.setattr(
        diarize_ort, "compute_titanet_embedding",
        lambda _audio, _session: np.eye(2, 192, dtype=np.float32)[0],
    )
    real_fast_clustering = diarize_ort.fast_clustering
    seen: list[int | None] = []

    def observe_num_clusters(embeddings, num_clusters=None, threshold=0.9, method="complete"):
        """记录生产函数传给聚类器的人数参数，并继续执行真实聚类。"""
        seen.append(num_clusters)
        return real_fast_clustering(
            embeddings, num_clusters=num_clusters, threshold=threshold, method=method
        )

    monkeypatch.setattr(diarize_ort, "fast_clustering", observe_num_clusters)
    monkeypatch.setattr(diarize_ort, "_get_pyannote_session", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(diarize_ort, "_get_titanet_session", lambda *_args, **_kwargs: MagicMock())

    diarize_ort.run_diarization_ort_cuda(
        str(audio_path), "seg.onnx", "emb.onnx", num_speakers=0
    )

    assert seen == [None]


def test_zero_duration_word_is_assigned_by_its_time_point() -> None:
    """零时长词应按时间点归入包含它的段，不丢失文字。"""
    from src.core.qwen3.merge import attach_words_to_segments

    segments = [
        Segment(start=0.0, end=2.0, speaker=0, text="左"),
        Segment(start=2.0, end=4.0, speaker=1, text="右"),
    ]
    words = [
        {"text": "零时长", "start": 1.5, "end": 1.5, "score": 0.8},
        {"text": "边界", "start": 2.0, "end": 2.0, "score": 0.7},
    ]

    out = attach_words_to_segments(segments, words)

    assert [word["text"] for word in out[0].words or []] == ["零时长", "边界"]
    assert out[1].words is None


def test_time_sorted_word_assignment_matches_original_for_nonzero_words() -> None:
    """公开词挂段结果与原始逐词逐段规则一致，覆盖重叠段及乱序输入词。"""
    from src.core.qwen3.merge import attach_words_to_segments

    segments = [
        Segment(start=4.0, end=8.0, speaker=1, text="后"),
        Segment(start=0.0, end=5.0, speaker=0, text="前"),
        Segment(start=7.0, end=10.0, speaker=2, text="尾"),
    ]
    words = [
        {"text": "w3", "start": 7.5, "end": 8.5},
        {"text": "w1", "start": 1.0, "end": 2.0, "score": 0.1},
        {"text": "w2", "start": 4.5, "end": 7.5},
        {"text": "tie", "start": 4.0, "end": 5.0},
        {"text": "w4", "start": 9.0, "end": 10.0},
        {"text": "outside", "start": 11.0, "end": 12.0},
    ]

    actual = attach_words_to_segments(segments, words)

    assert [segment.words for segment in actual] == _old_attach_reference(segments, words)


def test_missing_asr_chunks_uses_diarization_time_range_not_zero() -> None:
    """无 chunk 时间戳的全文入口应以有效 turn 范围构造时长。"""
    from src.core.qwen3.merge import merge_asr_and_diarize

    segments = merge_asr_and_diarize(
        "缺少内部 chunks 但保留全文。",
        [{"start": 7.0, "end": 13.0, "speaker": 2}],
    )

    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (7.0, 13.0)
    assert segments[0].end > segments[0].start
