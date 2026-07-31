"""
Qwen3-Diarize 后处理 (PR2 short-segment guard).

按 TDD 红绿循环逐步实现, 每次只为当前红测试加最少代码.
"""
from __future__ import annotations

import re


_BACKCHANNEL_TOKENS = {"对", "嗯嗯", "好的", "是的"}
_PURE_PUNCT_RE = re.compile(r"^[，。！？!?,.\s]+$")
_QUESTION_TAIL_RE = re.compile(r"(对吗|是吧|是不是|有没有|可以吗|好吗|是吗)$")


def is_backchannel(text: str) -> bool:
    """判断文本是否为 backchannel (空/单字短语气词/多字 backchannel/纯标点).

    Args:
        text: segment 的 ASR 文本.
    """
    if not text:
        return True
    stripped = text.strip()
    if stripped in _BACKCHANNEL_TOKENS:
        return True
    if _PURE_PUNCT_RE.match(text):
        return True
    return False


_QUESTION_TAIL_MAX_LEN = 14


def is_question_tail(text: str) -> bool:
    """判断文本是否为短问句尾巴 (含 '对吗' / '是吧' 等 marker 且 ≤14 字).

    Args:
        text: segment 的 ASR 文本.
    """
    stripped = (text or "").strip()
    if len(stripped) > _QUESTION_TAIL_MAX_LEN:
        return False
    return bool(_QUESTION_TAIL_RE.search(stripped))


def drop_tiny_segments(
    segments: list[dict], min_sec: float
) -> tuple[list[dict], dict]:
    """合并 dur < min_sec 的微短段到时间最近邻段.

    Args:
        segments: ordered list of {start, end, speaker, text}.
        min_sec: 短段阈值, dur 小于此值且 text 非空才会被合并.

    Returns:
        (new_segments, stats) — stats 含 dropped_total / merged_into_prev / merged_into_next.
    """
    out: list[dict] = []
    dropped = 0
    merged_into_prev = 0
    merged_into_next = 0
    skip: set[int] = set()
    for i, seg in enumerate(segments):
        if i in skip:
            continue
        dur = float(seg["end"]) - float(seg["start"])
        text = (seg.get("text") or "").strip()
        if dur >= min_sec or not text:
            out.append(seg)
            continue
        # 微短且 text 非空: 选 gap 更近一侧合并
        prev_seg = out[-1] if out else None
        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        target = None
        if next_seg is not None and prev_seg is not None:
            gap_prev = float(seg["start"]) - float(prev_seg["end"])
            gap_next = float(next_seg["start"]) - float(seg["end"])
            target = "prev" if gap_prev <= gap_next else "next"
        elif next_seg is not None:
            target = "next"
        elif prev_seg is not None:
            target = "prev"
        if target == "prev":
            prev_seg["text"] = (prev_seg.get("text") or "") + (seg.get("text") or "")
            prev_seg["end"] = max(float(prev_seg["end"]), float(seg["end"]))
            merged_into_prev += 1
            dropped += 1
        elif target == "next":
            new_next = dict(next_seg)
            new_next["text"] = (seg.get("text") or "") + (next_seg.get("text") or "")
            new_next["start"] = min(float(seg["start"]), float(next_seg["start"]))
            skip.add(i + 1)
            out.append(new_next)
            merged_into_next += 1
            dropped += 1
        else:
            out.append(seg)
    return out, {
        "dropped_total": dropped,
        "merged_into_prev": merged_into_prev,
        "merged_into_next": merged_into_next,
    }


def aba_smoothing(
    segments: list[dict], max_mid_sec: float
) -> tuple[list[dict], dict]:
    """A-B-A 抖动平滑: 短中间段强制改回 A speaker (不动 text).

    Args:
        segments: ordered list of {start, end, speaker, text}.
        max_mid_sec: 中间段最大时长, 超过此值不平滑.

    Returns:
        (new_segments, stats) — stats 含 changed (改了多少个中间段).
    """
    out = [dict(s) for s in segments]
    changed = 0
    for i in range(1, len(out) - 1):
        a, b, c = out[i - 1], out[i], out[i + 1]
        if str(a.get("speaker")) != str(c.get("speaker")):
            continue
        if str(a.get("speaker")) == str(b.get("speaker")):
            continue
        dur = float(b["end"]) - float(b["start"])
        if dur > max_mid_sec:
            continue
        text = (b.get("text") or "").strip()
        cps = (len(text) / dur) if dur > 0 else 0
        # 接受: backchannel / 极短高密度切碎 (<=1.0s 且 cps>=5)
        if is_backchannel(text) or (dur <= 1.0 and cps >= 5):
            b["speaker"] = a["speaker"]
            changed += 1
    return out, {"changed": changed}


def apply_short_segment_guard(
    segments: list[dict],
    enabled: bool = True,
    short_drop_sec: float = 1.5,
    aba_max_mid_sec: float = 1.5,
    merge_same: bool = True,
    merge_gap_sec: float = 0.05,
    max_span_sec: float = 0.0,
) -> tuple[list[dict], dict]:
    """short-segment guard 完整 pipeline 入口.

    依次跑: drop_tiny_segments -> aba_smoothing -> merge_consecutive_same_speaker.
    enabled=False 时直接返回原 segments.

    Args:
        segments: ordered list of {start, end, speaker, text}.
        enabled: 是否启用 guard. False 时直接返回输入.
        short_drop_sec: drop_tiny_segments 阈值.
        aba_max_mid_sec: aba_smoothing 中间段最大时长.
        merge_same: 是否在 ABA 后合并同 speaker.
        merge_gap_sec: merge_consecutive_same_speaker 的 gap 阈值.
        max_span_sec: 合并累计跨度上限 (秒); <=0 无上限. 透传给 merge_consecutive_same_speaker.

    Returns:
        (new_segments, stats) — stats 含各阶段子 stats + enabled flag.
    """
    if not enabled:
        return list(segments), {"enabled": False}
    cur = list(segments)
    stats: dict = {"enabled": True}
    cur, drop_stats = drop_tiny_segments(cur, min_sec=short_drop_sec)
    stats["drop"] = drop_stats
    cur, aba_stats = aba_smoothing(cur, max_mid_sec=aba_max_mid_sec)
    stats["aba"] = aba_stats
    if merge_same:
        cur, merged = merge_consecutive_same_speaker(
            cur, merge_gap_sec=merge_gap_sec, max_span_sec=max_span_sec,
        )
        stats["merge"] = {"merged": merged}
    return cur, stats


def merge_consecutive_same_speaker(
    segments: list[dict],
    merge_gap_sec: float = 0.05,
    max_span_sec: float = 0.0,
) -> tuple[list[dict], int]:
    """合并相邻同 speaker + gap ≤ merge_gap_sec 的连续段, 减少碎片.

    Args:
        segments: ordered list of {start, end, speaker, text}.
        merge_gap_sec: 允许合并的最大 gap, 超过则视为独立段.
        max_span_sec: 合并后累计跨度上限 (秒). <=0 不设上限 (默认, 保持历史行为);
            >0 时 next.end - prev.start 超限则在段边界断开, 防单人独白滚成巨段.

    Returns:
        (new_segments, merged_count).
    """
    if not segments:
        return list(segments), 0
    out = [dict(segments[0])]
    merged = 0
    for s in segments[1:]:
        prev = out[-1]
        same_speaker = str(prev.get("speaker")) == str(s.get("speaker"))
        small_gap = float(s["start"]) - float(prev["end"]) <= merge_gap_sec
        span = float(s["end"]) - float(prev["start"])
        span_ok = max_span_sec <= 0 or span <= max_span_sec
        if same_speaker and small_gap and span_ok:
            prev["text"] = (prev.get("text") or "") + (s.get("text") or "")
            prev["end"] = max(float(prev["end"]), float(s["end"]))
            merged += 1
        else:
            out.append(dict(s))
    return out, merged
