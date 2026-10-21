"""
时间轴融合 — 把 ASR 全文按 speaker turn 时长比例线性切到各 turn

输入:
  - asr_text: ASR 输出的完整文本 (无 timestamp,因为 production aligner 模型不存在)
  - turns: [{start, end, speaker}, ...] 由 sherpa diarization 给出

策略 (PoC 第一版,粗放线性切分):
  1. 总有效说话时间 D_total = sum(turn.end - turn.start)
  2. 每秒文本字符数 c_per_sec = len(asr_text) / D_total
  3. 按 turn 顺序累计,turn_i 的文本起止字符位置 = round(累积秒数 × c_per_sec)

精度局限:
  - 假设 ASR 文本时间线性 — 实际语速不均时会偏
  - 不能处理静音段 / 插话 / overlap
  - 后续优化: 接 forced aligner 拿词级 timestamp 再融合,精度可以到秒以内

输出:
  - segments: [{start, end, speaker, text}, ...]

本文件移植自 spikes/qwen3_diarize/src/merge.py, 纯函数, 无侧依赖。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """融合后的转录片段(内部表示, 不直接暴露到 API 层)"""
    start: float
    end: float
    speaker: int
    text: str
    # 词级时间戳(可选): word_align 挂词后填充 [{text,start,end,score}, ...] 绝对秒;
    # 未挂词时 None (向后兼容). silence_align / relabel 的 dataclasses.replace 自动透传.
    words: Optional[List[dict]] = None


def attach_words_to_segments(
    segments: List[Segment], words: List[dict]
) -> List[Segment]:
    """把 MMS 词级时间戳按时间窗增量挂进现有段 (纯函数, 不动入参).

    词按时间落在哪个段的 [start, end] 窗就挂哪个段; 跨段边界用最大重叠时长优先;
    零时长词按时间点归入覆盖段. 词不落任何段 (与所有段重叠均 ≤ 0) → 丢弃;
    段未收到词 → words 保持 None.

    不替换 merge_asr_chunks_and_diarize 的段边界, 只增量挂 words.

    Args:
        segments: 已 merge + silence_align 的 Segment 列表 (干净段).
        words: [{text, start, end, score?}, ...] 绝对秒 (word_align.align_chunks 输出).

    Returns:
        新 List[Segment], 收到词的段 words=[...] (保持词输入顺序), 其余 words=None.
    """
    if not segments:
        return []
    # 段和词分别按起点排序, 用扫描指针跳过已结束的早期段; 只检查与当前词时间窗
    # 相交的候选段. original_index 保留旧算法的并列优先级和每段内的词输入顺序.
    sorted_segments = sorted(
        enumerate(segments),
        key=lambda item: (item[1].start, item[1].end, item[0]),
    )
    sorted_words = sorted(
        enumerate(words or []),
        key=lambda item: (float(item[1]["start"]), float(item[1]["end"]), item[0]),
    )
    buckets: List[Optional[List[tuple[int, dict]]]] = [None] * len(segments)
    first_candidate = 0
    for word_index, w in sorted_words:
        w_start = float(w["start"])
        w_end = float(w["end"])
        is_point = w_start == w_end
        while first_candidate < len(sorted_segments):
            seg_end = sorted_segments[first_candidate][1].end
            expired = seg_end < w_start if is_point else seg_end <= w_start
            if not expired:
                break
            first_candidate += 1

        best_idx = -1
        best_overlap = 0.0
        for segment_position in range(first_candidate, len(sorted_segments)):
            seg_idx, seg = sorted_segments[segment_position]
            if seg.start > w_end or (not is_point and seg.start >= w_end):
                break
            if is_point:
                if seg.start <= w_start <= seg.end and (best_idx < 0 or seg_idx < best_idx):
                    best_idx = seg_idx
                continue
            overlap = min(seg.end, w_end) - max(seg.start, w_start)
            if overlap > best_overlap or (
                overlap == best_overlap and overlap > 0 and seg_idx < best_idx
            ):
                best_overlap = overlap
                best_idx = seg_idx
        if best_idx < 0:
            continue  # 词不落任何段, 丢弃
        if buckets[best_idx] is None:
            buckets[best_idx] = []
        buckets[best_idx].append(
            (
                word_index,
                {
                    "text": w["text"],
                    "start": w_start,
                    "end": w_end,
                    "score": w.get("score"),
                },
            )
        )

    attached: List[Optional[List[dict]]] = []
    for bucket in buckets:
        if bucket is None:
            attached.append(None)
        else:
            attached.append([word for _index, word in sorted(bucket, key=lambda item: item[0])])
    return [replace(seg, words=attached[i]) for i, seg in enumerate(segments)]


def relabel_segments_by_duration_desc(segments: List[Segment]) -> List[Segment]:
    """按 speaker 总时长降序重新映射 int label, 让 Speaker ID 跨 backend/平台稳定.

    raw cluster id 由底层算法决定 (sherpa/ort_cuda 互不兼容, 同 audio 跨 backend
    出来的 int 不同), 输出层直接 +1 转字符串会让客户端看到 "Speaker55" / "Speaker41"
    这种不稳定的编号. 这个 helper 在输出前把内部 int 按"主→次→第三"的总时长
    顺序重新映射成 0/1/2/..., 配合下游 `f"Speaker{i+1}"` 渲染出稳定的
    Speaker1 (说最多) / Speaker2 / ... .

    平局 (两个 speaker 总时长相同) 时按原 int 升序作为 tie breaker, 保证
    deterministic — 同一份 input 跑多次输出一致.

    Args:
        segments: List[Segment], speaker 字段是 raw int (cluster id).
    Returns:
        新 List[Segment], speaker 重映射为 0..N-1, 内容 (start/end/text) 不变.
    """
    if not segments:
        return segments
    durations: dict[int, float] = {}
    for s in segments:
        durations[s.speaker] = durations.get(s.speaker, 0.0) + (s.end - s.start)
    # 按 (-时长, 原int) 排序: 时长降序 + 平局时原 int 升序
    sorted_ids = sorted(durations.keys(), key=lambda k: (-durations[k], k))
    remap = {old: new for new, old in enumerate(sorted_ids)}
    return [replace(s, speaker=remap[s.speaker]) for s in segments]


def filter_spurious_speakers(
    turns: List[dict],
    min_speaker_total: float = 2.0,
    min_speaker_share: float = 0.01,
    audio_duration: float = 0.0,
) -> List[dict]:
    """合并"假说话人" — 某 speaker 总时长太小则归到时间最近的另一 speaker.

    阈值组合(满足任一即视为 spurious):
      - 绝对时长 < min_speaker_total (默认 2s)
      - 占音频时长比例 < min_speaker_share (默认 1%; 仅当 audio_duration > 0 才启用)

    场景:
      - 2 人 5min(300s) 音频: 1% = 3s,刚好滤掉<3s 的噪声碎片
      - 4 人 8min(480s) 音频: 1% = 4.8s,滤掉 NeMo+0.9 产生的 5 个 <3% spurious cluster
    """
    if not turns:
        return turns
    totals: dict = {}
    for t in turns:
        totals[t["speaker"]] = totals.get(t["speaker"], 0) + (t["end"] - t["start"])

    abs_threshold = min_speaker_total
    pct_threshold = min_speaker_share * audio_duration if audio_duration > 0 else 0.0
    threshold = max(abs_threshold, pct_threshold)
    spurious = {sp for sp, dur in totals.items() if dur < threshold}
    if not spurious:
        return turns

    valid_speakers = sorted(totals.keys() - spurious, key=lambda s: -totals[s])
    if not valid_speakers:
        return turns  # 全是噪声,什么都不改

    out = []
    for t in turns:
        if t["speaker"] in spurious:
            # 找时间最近的 valid speaker 邻居
            mid = (t["start"] + t["end"]) / 2
            best = valid_speakers[0]
            best_dist = float("inf")
            for ot in turns:
                if ot["speaker"] not in spurious and ot is not t:
                    omid = (ot["start"] + ot["end"]) / 2
                    d = abs(omid - mid)
                    if d < best_dist:
                        best_dist = d
                        best = ot["speaker"]
            out.append({**t, "speaker": best})
        else:
            out.append(t)
    return out


def merge_asr_and_diarize(asr_text: str, turns: List[dict]) -> List[Segment]:
    """按 turn 时长比例线性切分 ASR 文本.

    边界条件:
      - asr_text 为空 → 返回 []
      - turns 为空但 asr_text 有文本 → 全文归 speaker 0 单段返回 (不丢转写),
        打 DIARIZE-NO-TURNS WARNING。本函数无时间输入, start/end 置 0。
      - 总时长 <= 0 → 返回 []
      - 最后一个 turn 直接吃完剩余文本, 避免浮点截断丢字
    """
    if not asr_text:
        return []

    if not turns:
        # diarize 无 turn (例如右声道说话人丢失/聚类失败) 但 ASR 有文本:
        # 把全文归给单一 speaker 0 (与单说话人正常路径经 relabel 后的 label 一致),
        # 不能静默丢弃已识别的转写 — 否则空 segments 被当成功结果缓存。
        logger.warning(
            "DIARIZE-NO-TURNS: turns 为空但 asr_text 有 %d 字, "
            "全文归 speaker 0 单段返回",
            len(asr_text),
        )
        return [Segment(start=0.0, end=0.0, speaker=0, text=asr_text)]

    # 按 start 时间排序 (sherpa 已经排好,这里防御)
    turns_sorted = sorted(turns, key=lambda t: t["start"])

    durations = [(t["end"] - t["start"]) for t in turns_sorted]
    total_dur = sum(durations)
    if total_dur <= 0:
        return []

    total_chars = len(asr_text)
    c_per_sec = total_chars / total_dur

    segments: List[Segment] = []
    cumulative = 0.0
    prev_char_pos = 0

    for i, t in enumerate(turns_sorted):
        cumulative += durations[i]
        if i == len(turns_sorted) - 1:
            char_end = total_chars
        else:
            char_end = int(round(cumulative * c_per_sec))
            char_end = min(max(char_end, prev_char_pos), total_chars)

        chunk = asr_text[prev_char_pos:char_end]
        segments.append(
            Segment(
                start=float(t["start"]),
                end=float(t["end"]),
                speaker=int(t["speaker"]),
                text=chunk,
            )
        )
        prev_char_pos = char_end

    return sorted(segments, key=lambda seg: (seg.start, seg.end))


def _choose_text_boundary(text: str, target: int, search: int = 18) -> int:
    """Choose a split point near target, preferring sentence punctuation."""
    if target <= 0:
        return 0
    if target >= len(text):
        return len(text)

    punct = "。？！；!?;"
    lo = max(1, target - search)
    hi = min(len(text) - 1, target + search)
    candidates = []
    for idx in range(lo, hi + 1):
        if text[idx - 1] in punct:
            candidates.append(idx)
    if candidates:
        return min(candidates, key=lambda idx: abs(idx - target))
    return target


def _split_text_by_weights(text: str, weights: List[float]) -> List[str]:
    """Split text into len(weights) chunks while keeping punctuation intact."""
    if not weights:
        return []
    if len(weights) == 1:
        return [text]

    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        weights = [1.0] * len(weights)
        total = float(len(weights))

    chunks: List[str] = []
    cursor = 0
    acc = 0.0
    for i, weight in enumerate(weights[:-1]):
        acc += max(0.0, weight)
        target = int(round(len(text) * acc / total))
        target = max(cursor, target)
        expected_piece_chars = max(0, target - cursor)
        # A fixed wide punctuation search lets very short diarization turns
        # steal a whole sentence.  Scale the search window with the piece size.
        boundary_search = min(18, max(2, int(round(expected_piece_chars * 0.5))))
        boundary = _choose_text_boundary(text, target, search=boundary_search)
        chunks.append(text[cursor:boundary])
        cursor = boundary
    chunks.append(text[cursor:])
    return chunks


def merge_asr_chunks_and_diarize(chunks: Iterable, turns: List[dict]) -> List[Segment]:
    """Merge ASR chunk text with diarization turns using chunk time windows.

    This is a stricter version of ``merge_asr_and_diarize`` for Qwen3 long
    audio.  Qwen3-ASR exposes rough 40s internal chunk boundaries even without
    a forced aligner.  Splitting text inside each 40s chunk bounds drift much
    better than linearly cutting a whole 12min macro window.
    """
    chunks_sorted = sorted(
        [c for c in chunks if getattr(c, "text", "")],
        key=lambda c: (float(getattr(c, "start", 0.0)), float(getattr(c, "end", 0.0))),
    )
    if not chunks_sorted:
        return []

    if not turns:
        # diarize 无 turn 但 ASR chunks 有文本: 每个 chunk 归 speaker 0, 时间取
        # chunk 起止, 不丢转写 (与 merge_asr_and_diarize 的无 turn 分支同一策略)。
        logger.warning(
            "DIARIZE-NO-TURNS: turns 为空, %d 个非空 ASR chunk 全归 speaker 0",
            len(chunks_sorted),
        )
        return [
            Segment(
                start=round(float(c.start), 2),
                end=round(float(c.end), 2),
                speaker=0,
                text=str(c.text),
            )
            for c in chunks_sorted
        ]

    turns_sorted = sorted(turns, key=lambda t: t["start"])
    segments: List[Segment] = []

    for chunk in chunks_sorted:
        chunk_start = float(chunk.start)
        chunk_end = float(chunk.end)
        text = str(chunk.text)
        if chunk_end <= chunk_start or not text:
            continue

        overlaps = []
        for turn in turns_sorted:
            start = max(chunk_start, float(turn["start"]))
            end = min(chunk_end, float(turn["end"]))
            if end - start > 0.03:
                overlaps.append((start, end, int(turn["speaker"])))

        if not overlaps:
            mid = (chunk_start + chunk_end) / 2.0
            nearest = min(
                turns_sorted,
                key=lambda t: min(abs(float(t["start"]) - mid), abs(float(t["end"]) - mid)),
            )
            overlaps = [(chunk_start, chunk_end, int(nearest["speaker"]))]

        pieces = _split_text_by_weights(text, [end - start for start, end, _sp in overlaps])
        for (start, end, speaker), piece in zip(overlaps, pieces):
            if not piece:
                continue
            segments.append(
                Segment(
                    start=round(start, 2),
                    end=round(end, 2),
                    speaker=speaker,
                    text=piece,
                )
            )

    return sorted(segments, key=lambda seg: (seg.start, seg.end))


# ==================== silence-aware 段切点对齐 ====================
#
# 携进自 spikes/qwen3_silence_align (spike commit 405abf6).
#
# 目标: 把 merge_asr_chunks_and_diarize 输出的段切点吸附到最近静音中点,
# 解决 Qwen3 chunk 硬边界 (40s 内部 chunk + 12min macro window) 引入的
# "切在话中间" 问题. 60s podcast: align_ratio 54.55% → 73.68% (+19pp),
# 60min long audio: 27.41% → 60.73% (+33pp).
#
# 设计:
#   - 独立 snap 每个 segment 的 start / end (不强求段连续 — baseline 自带
#     diarize overlap 插话场景, 强行同步会触发 0 时长段)
#   - 容忍窗口 tolerance 内才动, 否则不变
#   - snap 后段时长 < min_segment_dur 则回退该 snap, 保护段不退化


@dataclass
class SilenceInterval:
    """静音区间 (start ~ end), 由 speech_regions 反向算出."""
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2

    def contains(self, ts: float) -> bool:
        return self.start <= ts <= self.end


def silence_intervals_from_speech(
    speech_regions: List[dict],
    audio_duration: float,
) -> List[SilenceInterval]:
    """把 speech_regions 反向解析为 silence_intervals.

    Args:
        speech_regions: ffmpeg_speech_regions 输出, list[{'start', 'end'}].
        audio_duration: 音频总时长 (秒).

    Returns:
        相邻 speech 区段之间的 silence + 末尾 silence.
    """
    sorted_regions = sorted(
        [{"start": float(r["start"]), "end": float(r["end"])} for r in speech_regions],
        key=lambda r: r["start"],
    )
    silences: List[SilenceInterval] = []
    cursor = 0.0
    for r in sorted_regions:
        if r["start"] > cursor:
            silences.append(SilenceInterval(start=cursor, end=r["start"]))
        cursor = max(cursor, r["end"])
    if cursor < audio_duration:
        silences.append(SilenceInterval(start=cursor, end=audio_duration))
    return silences


def _find_snap_target(
    ts: float,
    silences: List[SilenceInterval],
    tolerance: float,
) -> float | None:
    """找 ts 应该吸附到的目标时间戳, None 表示距离过远不动.

    规则:
      1. ts 已在某个 silence 内 → 不动 (返回 ts 本身, caller 用 != 跳过)
      2. ts 不在 silence 内 → 找最近 silence 中点, 距离 <= tolerance 才返回
    """
    for s in silences:
        if s.contains(ts):
            return ts
    best_target: float | None = None
    best_dist = tolerance
    for s in silences:
        d = abs(ts - s.mid)
        if d < best_dist:
            best_dist = d
            best_target = s.mid
    return best_target


def snap_segments_to_silence(
    segments: List[Segment],
    speech_regions: List[dict],
    audio_duration: float,
    tolerance: float = 2.0,
    min_segment_dur: float = 0.1,
) -> tuple[List[Segment], dict]:
    """对 segments 做 snap-to-silence (独立吸附 start 和 end).

    设计:
      - 每个 segment 的 start / end 独立 snap 到最近 silence (容忍 tolerance 内)
      - 不强求 end[i] == start[i+1]: baseline 的 diarize overlap (插话场景)
        必须保留, 否则会触发 0 时长段
      - snap 后段时长 < min_segment_dur 时回退该 snap, 保护段时长
      - 第一段 start 与最后一段 end 不动 (音频起止天然对齐)

    Args:
        segments: merge_asr_chunks_and_diarize 输出 (List[Segment]).
        speech_regions: ffmpeg_speech_regions 输出.
        audio_duration: 音频总时长 (秒).
        tolerance: 吸附容忍窗口 (秒). 距离 > tolerance 不动.
        min_segment_dur: snap 后段时长 < 此值则回退, 保护段不退化.

    Returns:
        (new_segments, stats). new_segments 是 dataclasses.replace 出的新 Segment 列表,
        原 segments 不被修改.
    """
    if not segments:
        return [], {
            "snapped_starts": 0,
            "snapped_ends": 0,
            "total_starts": 0,
            "total_ends": 0,
            "skipped_zero_dur": 0,
            "tolerance_sec": tolerance,
            "min_segment_dur_sec": min_segment_dur,
            "silence_intervals_count": 0,
        }

    silences = silence_intervals_from_speech(speech_regions, audio_duration)
    new_segs = [replace(s) for s in segments]
    snapped_starts = 0
    snapped_ends = 0
    skipped_zero_dur = 0
    n = len(new_segs)
    total_starts = max(0, n - 1)  # 跳过 segments[0].start
    total_ends = max(0, n - 1)    # 跳过 segments[-1].end

    for i, s in enumerate(new_segs):
        if i > 0:
            target = _find_snap_target(s.start, silences, tolerance)
            if target is not None and target != s.start:
                new_dur = s.end - target
                if new_dur >= min_segment_dur:
                    s.start = round(target, 3)
                    snapped_starts += 1
                else:
                    skipped_zero_dur += 1
        if i < n - 1:
            target = _find_snap_target(s.end, silences, tolerance)
            if target is not None and target != s.end:
                new_dur = target - s.start
                if new_dur >= min_segment_dur:
                    s.end = round(target, 3)
                    snapped_ends += 1
                else:
                    skipped_zero_dur += 1

    return new_segs, {
        "snapped_starts": snapped_starts,
        "snapped_ends": snapped_ends,
        "total_starts": total_starts,
        "total_ends": total_ends,
        "skipped_zero_dur": skipped_zero_dur,
        "tolerance_sec": tolerance,
        "min_segment_dur_sec": min_segment_dur,
        "silence_intervals_count": len(silences),
    }


def segments_to_srt(segments: List[Segment], include_speaker: bool = True) -> str:
    """把 segments 转 SRT 字幕格式 (跳过空文本片段, 索引按非空 segment 重新编号).

    include_speaker=False (diarize=false, D8): 不带 "SpeakerN:" 前缀 —
    字幕场景正是要纯文本行.
    """
    def fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: List[str] = []
    idx = 0
    for seg in segments:
        if not seg.text.strip():
            continue
        idx += 1
        lines.append(str(idx))
        lines.append(f"{fmt(seg.start)} --> {fmt(seg.end)}")
        if include_speaker:
            # 与 FunASR SRT 字节级对齐: 冒号后无空格(funasr_transcriber.py:516)
            lines.append(f"Speaker{seg.speaker + 1}:{seg.text.strip()}")
        else:
            lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)
