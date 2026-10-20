"""Qwen3 merge: diarize 无 turn 时不得丢弃 ASR 文本 (全仓审查 261020 卡 4a).

修复前: merge_asr_and_diarize / merge_asr_chunks_and_diariize 在 turns==[] 时
直接 return [], ASR 已识别的全文被静默丢弃, 空 segments 被当成功结果缓存 30 天.
修复后: turns 为空且有文本时, 全部文本归 speaker 0 (与单说话人正常路径经
relabel_segments_by_duration_desc 后的 label 一致), 时间范围取 ASR chunk 起止,
并打含 DIARIZE-NO-TURNS 标记的 WARNING; ASR 文本为空仍返回 [].
"""
import logging

from src.core.qwen3.asr import ASRChunkItem
from src.core.qwen3.merge import merge_asr_and_diarize, merge_asr_chunks_and_diarize


def _chunk(text: str, start: float, end: float, index: int) -> ASRChunkItem:
    """构造 ASRChunkItem 测试样本 (公开入口数据的形状, 不 mock 内部)."""
    return ASRChunkItem(text=text, start=start, end=end, index=index)


class TestMergeChunksEmptyTurns:
    """merge_asr_chunks_and_diarize — chunks 版本的无 turn 行为."""

    def test_chunks_with_text_empty_turns_keeps_all_text(self, caplog):
        """turns 为空但 chunks 有文本: segments 非空、文本完整拼回、speaker 全 0."""
        chunks = [
            _chunk("你好世界。", 0.0, 4.0, 0),
            _chunk("今天天气不错。", 40.0, 44.0, 1),
        ]
        with caplog.at_level(logging.WARNING):
            segments = merge_asr_chunks_and_diarize(chunks, [])

        assert segments, "turns 为空但 ASR 有文本时不得返回空段 (静默丢转写)"
        assert "".join(s.text for s in segments) == "你好世界。今天天气不错。"
        assert all(s.speaker == 0 for s in segments)
        # 时间范围取 ASR chunk 起止
        assert [(s.start, s.end) for s in segments] == [(0.0, 4.0), (40.0, 44.0)]
        assert any("DIARIZE-NO-TURNS" in r.message for r in caplog.records)

    def test_no_chunks_returns_empty_without_warning(self, caplog):
        """ASR 文本为空仍返回 [], 且不打 DIARIZE-NO-TURNS (文本空不算降级)."""
        with caplog.at_level(logging.WARNING):
            assert merge_asr_chunks_and_diarize([], []) == []
            assert merge_asr_chunks_and_diarize(
                [_chunk("", 0.0, 4.0, 0)], []
            ) == []
        assert not any("DIARIZE-NO-TURNS" in r.message for r in caplog.records)

    def test_with_turns_speaker_from_turn_unchanged(self):
        """回归保护: 有 turns 时 speaker 仍来自 turn, 修复不写死 0."""
        turns = [{"start": 0.0, "end": 4.0, "speaker": 1}]
        segments = merge_asr_chunks_and_diarize(
            [_chunk("你好世界。", 0.0, 4.0, 0)], turns
        )
        assert len(segments) == 1
        assert segments[0].speaker == 1


class TestMergeAsrEmptyTurns:
    """merge_asr_and_diarize — 纯文本版本的无 turn 行为."""

    def test_text_with_empty_turns_keeps_text_single_speaker(self, caplog):
        """turns 为空但 asr_text 非空: 单段、speaker 0、全文保留、打 WARNING."""
        with caplog.at_level(logging.WARNING):
            segments = merge_asr_and_diarize("一整段识别出来的文本。", [])

        assert len(segments) == 1
        assert segments[0].speaker == 0
        assert segments[0].text == "一整段识别出来的文本。"
        assert any("DIARIZE-NO-TURNS" in r.message for r in caplog.records)

    def test_empty_text_returns_empty(self):
        """asr_text 为空仍返回 [] (原行为不变)."""
        assert merge_asr_and_diarize("", []) == []

    def test_with_turns_unchanged(self):
        """回归保护: 有 turns 时按比例切分、speaker 来自 turn."""
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": 0},
            {"start": 2.0, "end": 4.0, "speaker": 3},
        ]
        segments = merge_asr_and_diarize("abcd", turns)
        assert [s.speaker for s in segments] == [0, 3]
        assert "".join(s.text for s in segments) == "abcd"
