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
        """旧缓存已合并形态: 相邻同 speaker 必 gap>=3s 或不同 speaker → 过一遍不产生新巨段."""
        segs = [
            _seg(0.0, 100.0, "已合并大段A"),
            _seg(105.0, 200.0, "已合并大段B"),  # gap=5s > 3
        ]
        out1 = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        out2 = merge_segments_view(out1, gap_sec=3.0, max_span_sec=120.0)
        assert len(out1) == 2
        assert len(out2) == 2
        assert out1[0].text == out2[0].text
        assert out1[1].text == out2[1].text

    def test_span_equal_to_cap_still_merges(self):
        """next.end - cur.start == max_span_sec 仍可合并（<= 条件）."""
        segs = [
            _seg(0.0, 50.0, "一"),
            _seg(60.0, 120.0, "二"),  # span = 120, gap = 10 but gap_sec would need be >10
        ]
        # gap=10 >= 3 → actually gap breaks. Use small gap:
        segs = [
            _seg(0.0, 60.0, "一"),
            _seg(60.5, 120.0, "二"),  # span=120, gap=0.5
        ]
        out = merge_segments_view(segs, gap_sec=3.0, max_span_sec=120.0)
        assert len(out) == 1
        assert out[0].end_time - out[0].start_time == 120.0
