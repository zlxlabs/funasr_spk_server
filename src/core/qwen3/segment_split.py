"""nospk 分层切段 — diarize=false 时把超长 ASR chunk 段切成字幕可用的小段

背景 (设计定案 D5+T1): diarize=false 跳过 diarize 后没有 turn 边界, 段 = ASR
~40s 内部 chunk, SRT 不可用. silence_align 只做切点吸附不做重切, 必须独立切段.

两层 fallback 策略 (按段逐一选层):
  1. 段带 words (word_align 词级时间戳, CUDA 主力路径默认开) → 词隙中点精确切
  2. 无 words → 静音表切 (speech_regions 反算 silence midpoints)
两层都拿不到候选切点时硬切在目标等分点 (保证 max_dur 上界, 计入 stats.hard_cuts).

文本归属沿用现役 char-ratio 机制 (_split_text_by_weights, 标点优先), 切层只决定
切点"时间"精度. 内置最小段时长阈值 min_dur (吸收 short_segment_guard 的通用清理
职责: 不产出微短片).

SRT 不变量 (T-D #5): word_align 是 JSON-only, SRT 路径段上永远无 words →
SRT 的 nospk 切段永远走静音 fallback. 这是设计内行为, 不是缺陷.

纯函数, 无副依赖; caller (qwen3_transcriber.apply_nospk_split_to_segments)
负责 ffmpeg speech_regions 获取与 (关/空/异常)→fallback to input 的容错形状.
"""
from __future__ import annotations

import math
from typing import List, Optional

from src.core.qwen3.merge import (
    Segment,
    SilenceInterval,
    _split_text_by_weights,
    silence_intervals_from_speech,
)


def _pick_cuts(
    seg: Segment,
    candidates: List[float],
    max_dur: float,
    min_dur: float,
) -> tuple[List[float], int]:
    """在目标等分点附近选最近候选切点; 无可用候选时硬切在目标点.

    Returns:
        (cuts, hard_cuts) — cuts 升序, 相邻间隔 >= min_dur; hard_cuts 是硬切次数.
    """
    dur = seg.end - seg.start
    n = math.ceil(dur / max_dur)
    if n <= 1:
        return [], 0
    sorted_cands = sorted(candidates)
    cuts: List[float] = []
    hard_cuts = 0
    prev = seg.start
    for i in range(1, n):
        remaining_pieces = n - i
        # 保证当前片不超过 max_dur，同时给剩余片保留足够容量，并满足 min_dur。
        lower = max(prev + min_dur, seg.end - max_dur * remaining_pieces)
        upper = min(prev + max_dur, seg.end - min_dur * remaining_pieces)
        if lower > upper + 1e-9:
            break
        target = seg.start + dur * i / n
        viable = [c for c in sorted_cands if lower - 1e-9 <= c <= upper + 1e-9]
        if viable:
            cut = min(viable, key=lambda c: abs(c - target))
        else:
            # 候选静音/词隙都越过 max_dur 安全窗时，硬切到可行边界。
            cut = min(max(target, lower), upper)
            hard_cuts += 1
        if cut - prev < min_dur - 1e-9:
            continue
        cuts.append(cut)
        prev = cut
    return cuts, hard_cuts


def _word_gap_candidates(seg: Segment, min_dur: float) -> List[float]:
    """词隙候选切点: 相邻词之间的中点 (不切进任何词内部)."""
    words = seg.words or []
    out: List[float] = []
    for a, b in zip(words, words[1:]):
        gap_mid = (float(a["end"]) + float(b["start"])) / 2
        if seg.start + min_dur <= gap_mid <= seg.end - min_dur:
            out.append(gap_mid)
    return out


def _silence_candidates(
    seg: Segment, silences: List[SilenceInterval], min_dur: float
) -> List[float]:
    """静音候选切点: 落在段内的 silence 中点."""
    return [
        s.mid
        for s in silences
        if seg.start + min_dur <= s.mid <= seg.end - min_dur
    ]


def _build_pieces(seg: Segment, cuts: List[float]) -> List[Segment]:
    """按切点构造子段: 文本 char-ratio 归属 (标点优先), words 按词中点落窗重分."""
    bounds = [seg.start] + cuts + [seg.end]
    weights = [b - a for a, b in zip(bounds, bounds[1:])]
    texts = _split_text_by_weights(seg.text, weights)
    pieces: List[Segment] = []
    n = len(weights)
    for i, ((a, b), t) in enumerate(zip(zip(bounds, bounds[1:]), texts)):
        piece_words: Optional[List[dict]] = None
        if seg.words:
            piece_words = [
                w
                for w in seg.words
                if a <= (float(w["start"]) + float(w["end"])) / 2
                and (
                    (float(w["start"]) + float(w["end"])) / 2 < b
                    or (i == n - 1 and (float(w["start"]) + float(w["end"])) / 2 <= b)
                )
            ] or None
        pieces.append(
            Segment(
                start=round(a, 3),
                end=round(b, 3),
                speaker=seg.speaker,
                text=t,
                words=piece_words,
            )
        )
    return pieces


def split_long_segments(
    segments: List[Segment],
    *,
    speech_regions: List[dict],
    audio_duration: float,
    max_dur: float,
    min_dur: float,
) -> tuple[List[Segment], dict]:
    """对超长段做两层 fallback 切分 (词隙优先, 静音兜底, 硬切保底).

    Args:
        segments: nospk 路径的 Segment 列表 (来自 ASR chunk 出段).
        speech_regions: ffmpeg_speech_regions 输出 (与 silence_intervals_from_speech
            同型, 不造模糊 silences 参数).
        audio_duration: 音频总时长 (秒).
        max_dur: 超过此时长的段触发切分, 也是切出片的近似上界.
        min_dur: 切出片的最小时长 (内置清理阈值).

    Returns:
        (new_segments, stats). 原 segments 不被修改; 不需要切的段原样透传.
        stats: split_segments / produced_segments / word_split / silence_split /
               hard_cuts. 文本逐字无损 (char-ratio 切分覆盖完整原文).
    """
    stats = {
        "split_segments": 0,
        "produced_segments": 0,
        "word_split": 0,
        "silence_split": 0,
        "hard_cuts": 0,
    }
    if not segments:
        return [], stats

    silences = silence_intervals_from_speech(speech_regions or [], audio_duration)
    out: List[Segment] = []
    for seg in segments:
        if seg.end - seg.start <= max_dur or not seg.text.strip():
            out.append(seg)
            continue
        if seg.words:
            candidates = _word_gap_candidates(seg, min_dur)
            method = "word_split"
        else:
            candidates = _silence_candidates(seg, silences, min_dur)
            method = "silence_split"
        cuts, hard_cuts = _pick_cuts(seg, candidates, max_dur, min_dur)
        if not cuts:
            out.append(seg)
            continue
        pieces = _build_pieces(seg, cuts)
        out.extend(pieces)
        stats["split_segments"] += 1
        stats["produced_segments"] += len(pieces)
        stats[method] += 1
        stats["hard_cuts"] += hard_cuts
    return out, stats
