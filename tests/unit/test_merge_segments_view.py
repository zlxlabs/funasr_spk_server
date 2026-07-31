"""D1: merge_segments_view 纯函数单测

同说话人相邻句合并视图（serve 投影层）:
- 同 speaker + gap < gap_sec + 累计跨度 ≤ max_span_sec → 合并
- 任一条件不满足断开；单句自身超 cap 原样保留（只并不切）
- 纯函数、不 mutate 输入；max_span_sec<=0 无上限
"""
from __future__ import annotations

from src.core.result_projection import merge_segments_view
from src.models.schemas import TranscriptionSegment


def _seg(start: float, end: float, text: str, speaker: str = "Speaker1") -> TranscriptionSegment:
    return TranscriptionSegment(
        start_time=start, end_time=end, text=text, speaker=speaker,
    )


class TestMergeSegmentsView:
    def test_empty_list(self):
        assert merge_segments_view([], gap_sec=3.0, max_span_sec=120.0) == []

    def test_gap_breaks_merge(self):
        """同 speaker 但 gap >= gap_sec → 断开."""
        segs = [
            _seg(0.0, 1.0, "A"),
            _seg(5.0, 6.0, "B"),  # gap=4s >= 3
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 2
        assert out[0].text == "A"
        assert out[1].text == "B"

    def test_speaker_change_breaks_merge(self):
        """换 speaker 即使 gap=0 也不合并."""
        segs = [
            _seg(0.0, 1.0, "我", "Speaker1"),
            _seg(1.0, 2.0, "你", "Speaker2"),
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 2
        assert out[0].speaker == "Speaker1"
        assert out[1].speaker == "Speaker2"

    def test_cap_breaks_merge(self):
        """累计跨度超 max_span_sec → 在句边界断开."""
        # 三句同 speaker, gap 小; 0-50 + 50-100 = span 100 ok; +100-150 span 150 > 120 → 断开
        segs = [
            _seg(0.0, 50.0, "一"),
            _seg(50.5, 100.0, "二"),
            _seg(100.5, 150.0, "三"),
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 2
        assert out[0].text == "一二"
        assert out[0].start_time == 0.0
        assert out[0].end_time == 100.0
        assert out[1].text == "三"
        assert out[1].start_time == 100.5
        assert out[1].end_time == 150.0
        # 每段跨度 ≤ 120
        for s in out:
            assert s.end_time - s.start_time <= 120.0

    def test_single_sentence_over_cap_kept(self):
        """单句自身超 cap → 原样保留，绝不切句内."""
        segs = [
            _seg(0.0, 200.0, "超长单句"),
            _seg(200.5, 210.0, "下一句"),
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        # 第一句保留；下一句若并上 span=210 > 120 断开 → 两段
        assert len(out) == 2
        assert out[0].text == "超长单句"
        assert out[0].end_time - out[0].start_time == 200.0
        assert out[1].text == "下一句"

    def test_max_span_le_zero_no_cap(self):
        """max_span_sec <= 0 ⇒ 不设上限（等价旧行为，仅 gap/speaker）."""
        segs = [
            _seg(0.0, 50.0, "一"),
            _seg(50.5, 100.0, "二"),
            _seg(100.5, 200.0, "三"),
            _seg(200.5, 400.0, "四"),
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=0.0)
        assert len(out) == 1
        assert out[0].text == "一二三四"
        assert out[0].end_time == 400.0

        out_neg = merge_segments_view(segs, gap_sec=3.0, max_span_sec=-1.0)
        assert len(out_neg) == 1

    def test_merges_same_speaker_within_gap_and_cap(self):
        """同 speaker + 小 gap + 未超 cap → 合并，文本直接拼接保留标点."""
        segs = [
            _seg(0.0, 1.0, "你好。"),
            _seg(1.5, 2.0, "世界！"),
            _seg(2.2, 3.0, "继续"),
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 1
        assert out[0].text == "你好。世界！继续"
        assert out[0].start_time == 0.0
        assert out[0].end_time == 3.0

    def test_does_not_mutate_input(self):
        segs = [_seg(0.0, 1.0, "A"), _seg(1.0, 2.0, "B")]
        originals = [(s.start_time, s.end_time, s.text, s.speaker) for s in segs]
        merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert [(s.start_time, s.end_time, s.text, s.speaker) for s in segs] == originals

    def test_idempotent_on_already_merged_input(self):
        """已合并形态过投影幂等: 可合并输入先 merge 得 len==1, 再过一遍字段完全相等."""
        segs = [
            _seg(0.0, 1.0, "A"),
            _seg(1.2, 2.0, "B"),  # gap=0.2 < 3, span 不超 cap
            _seg(2.1, 3.0, "C"),
        ]
        out1 = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out1) == 1
        assert out1[0].text == "ABC"
        out2 = merge_segments_view(out1, gap_sec=3.0, max_span_sec=120.0)
        assert len(out2) == 1
        assert out1[0].model_dump() == out2[0].model_dump()

    def test_span_equal_to_cap_still_merges(self):
        """next.end - cur.start == max_span_sec 仍可合并（<= 条件）."""
        # span=120, gap=0.5 < 3 → 同 speaker 可并
        segs = [
            _seg(0.0, 60.0, "一"),
            _seg(60.5, 120.0, "二"),
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 1
        assert out[0].end_time - out[0].start_time == 120.0

    def test_nested_segment_merge_end_not_shrink(self):
        """嵌套 B 完全落在 A 内: 合并后 end=max, 不收缩丢失时间."""
        segs = [
            _seg(0.0, 5.0, "A"),
            _seg(4.0, 4.5, "B"),  # 嵌套, gap=-1 < 3
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 1
        assert out[0].start_time == 0.0
        assert out[0].end_time == 5.0  # max(5, 4.5), 非 next.end=4.5
        assert out[0].text == "AB"

    def test_out_of_order_segment_not_merged(self):
        """倒序段: next.start < 上一输入 start → 不合并, 按输入顺序原样输出."""
        segs = [
            _seg(10.0, 11.0, "后"),
            _seg(0.0, 1.0, "前"),
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 2
        assert out[0].start_time == 10.0 and out[0].end_time == 11.0 and out[0].text == "后"
        assert out[1].start_time == 0.0 and out[1].end_time == 1.0 and out[1].text == "前"

    def test_partial_overlap_merge_extends_end(self):
        """部分重叠 A=[0,5], B=[4,8] → 合并为 [0,8]."""
        segs = [
            _seg(0.0, 5.0, "A"),
            _seg(4.0, 8.0, "B"),
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 1
        assert out[0].start_time == 0.0
        assert out[0].end_time == 8.0
        assert out[0].text == "AB"

    def test_three_segment_regression_breaks_on_adjacent_non_monotonic(self):
        """三段回退: A=[0,5] B=[4,4.5] C=[3,6] — AB 可并, C 相对 B 起点回退断开.
        守卫必须比对相邻输入段 start, 不能只比合并组首段 start (3>=0 会误并).
        """
        segs = [
            _seg(0.0, 5.0, "A"),
            _seg(4.0, 4.5, "B"),
            _seg(3.0, 6.0, "C"),  # 3 < 4 (上一输入 B.start) → 断开
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 2
        assert out[0].start_time == 0.0
        assert out[0].end_time == 5.0
        assert out[0].text == "AB"
        assert out[1].start_time == 3.0
        assert out[1].end_time == 6.0
        assert out[1].text == "C"
