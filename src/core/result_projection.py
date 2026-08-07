"""serve 层输出投影 — nospk 抹 speaker + funasr segment 合并视图

设计定案 D3+T-A (nospk) + issue #1 (segment merge):
纯函数, 缓存永远只存真算结果 (句级 / diarized), 有损视图在出口做.

应用点两个 (双出口):
1. db_manager.get_cached_result 出口 — 覆盖 websocket_handler 早返回
   (upload_request / chunked finalize) + task_manager 缓存读;
   E1: qwen3 nospk 请求 exact tag miss → 同引擎同 wa-tag diarized 行现场投影
   (标 projected:true, **不回写** — 缓存里永远只有真算结果, T2);
   funasr 免折维行 (D4) 本身 diarized, 出口投影一行通吃.
   funasr JSON 行: 先 merge_segments_view 再 nospk (顺序铁律).
2. task_manager fresh 结果出口 — funasr 先存句级再 merge 视图 (+ 可选 nospk).

对外契约 (D8): diarize=false ⇒ segments[].speaker=null + speakers=[] +
SRT 无 "SpeakerN:" 前缀. words / 时间 / 文本不动.
"""
from __future__ import annotations

from typing import Iterable, List

from src.models.schemas import TranscriptionResult, TranscriptionSegment


def build_result_metadata(
    *,
    engine: str,
    options,
    output_format: str = "json",
    projected: bool = False,
    has_words: bool = None,
    word_align_error: str = None,
    segment_merge_applied: bool = None,
) -> dict:
    """E2 effective options 回显块 (serve 层组装, 不入库).

    字段: engine / diarize / word_align / language / projected
    (+ 可选 word_align_error / segment_merge_max_span_sec).
    合并优先级 (E2 定义): request > 分片 session 回填 > config > 引擎默认 —
    request 与 session 回填已在 TranscribeOptions 收拢 (resolve_word_align 在构造 options
    时解析 effective word_align), 本函数只做 options 与引擎默认的合并:

    - word_align (决策 1A + 2A + codex #12): 反映本响应**实际交付**的词级时间戳, 而非"请求想要".
      delivered = qwen3 引擎 AND options.word_align (effective) AND output_format=='json'
                  AND words 实际挂上. funasr 无此能力恒 False; SRT 不挂词恒 False.
      · has_words=None (缓存命中等无法判时): 按 requested 推 (决策 B 保证 +wa 缓存行必有词).
      · has_words=False (fresh 对齐失败): delivered=False, 另附 word_align_error 说明原因.
    - language: request 值 strip 规范化优先; word_align 请求时空值回退
      config.qwen3.word_align_language (与缓存折维同一规范化规则)
    - projected: 请求级属性 — 本响应是否由 diarized 结果投影而来
    - word_align_error: 仅当请求词级但失败时附上 (fresh 出口; 缓存命中无此键)
    - segment_merge_max_span_sec: **实际过了 merge 视图** 的 JSON 出口才回显 cap.
      · segment_merge_applied is None (fresh 默认): 按 engine==funasr 推断
        (fresh 时 task.engine 与是否应用一致)
      · 显式 bool (缓存命中): 跟 get_cached_result 通道事实, 不跟请求 engine
        (跨引擎回退时请求 engine 与行 engine 可分离)
    """
    from src.core.config import config

    requested = engine == "qwen3" and bool(options.word_align) and output_format == "json"
    delivered = requested if has_words is None else (requested and bool(has_words))
    language = (options.language or "").strip() or None
    if requested and language is None:
        language = config.qwen3.word_align_language
    md = {
        "engine": engine,
        "diarize": options.diarize,
        "word_align": delivered,
        "language": language,
        "projected": projected,
    }
    if word_align_error:
        md["word_align_error"] = word_align_error
    if engine in ("funasr", "qwen3") and options.terms:
        md["context_applied"] = True
        md["terms_count"] = len(options.terms)
    # cap 键存在 == 本响应 segments 实际过了 merge 视图
    if segment_merge_applied is None:
        show_cap = engine == "funasr" and output_format == "json"
    else:
        show_cap = bool(segment_merge_applied) and output_format == "json"
    if show_cap:
        md["segment_merge_max_span_sec"] = config.transcription.segment_merge_max_span_sec
    return md


def cache_hit_metadata(cached_result, *, engine, options, output_format):
    """缓存命中出口共享纯逻辑 (3 处去重: ws 整文件 / ws 分片 / task_manager.submit).

    把 "projected / segment_merge_applied 提取 + metadata 构建 + SRT-dict 有效性判断"
    这段被抄 3 份的逻辑收拢成一个**无副作用**纯函数. 返回 (metadata, projected, srt_ok):
    - metadata: build_result_metadata 的结果 (srt_ok=False 时为 None)
    - projected: 该缓存是否由 diarized 投影而来 (回显用)
    - srt_ok: SRT 请求时缓存是否为合法 srt-dict; False ⇒ 调用方应跳过缓存继续处理

    codex #7/#8 定的边界: 统一用 get 不 pop (绝不改 cached_result); 控制流 (set task /
    计数 / 发消息 / 排除 projected key 不泄漏给客户端) 仍由各出口自理, 本函数只组装数据.
    segment_merge_applied 与 projected 同通道 (JSON: result.metadata; SRT 不应用 merge).
    """
    if output_format == "srt":
        srt_ok = isinstance(cached_result, dict) and cached_result.get("format") == "srt"
        if not srt_ok:
            return None, False, False
        projected = bool(cached_result.get("projected", False))  # get 不 pop
        merge_applied = False  # SRT 出口不过 merge 视图
    else:
        meta = cached_result.metadata or {}
        projected = bool(meta.get("projected"))  # get 不 pop
        merge_applied = bool(meta.get("segment_merge_applied"))
    md = build_result_metadata(
        engine=engine, options=options, output_format=output_format, projected=projected,
        segment_merge_applied=merge_applied,
    )
    return md, projected, True


def merge_segments_view(
    segments: List[TranscriptionSegment],
    *,
    gap_sec: float,
    max_span_sec: float,
) -> List[TranscriptionSegment]:
    """同说话人相邻句合并视图 (纯函数, 不 mutate 输入).

    合并条件 (全满足才并):
      同 speaker AND 相邻输入 start 单调 (next.start >= 上一输入段 start)
      AND next.start - cur.end < gap_sec
      AND next.end - cur.start <= max_span_sec (max_span_sec<=0 不设上限).
    合并 end_time = max(cur.end, next.end) — 嵌套/重叠不收缩丢失时间.
    乱序守卫比对**相邻输入段**起点 (非合并组首段 start): 三段回退
    A=[0,5] B=[4,4.5] C=[3,6] 中 C.start=3 < B.start=4 → 断开, 不误并入组首 0.
    单句自身超 cap 原样保留, 绝不切句内 (投影只并不切).
    文本拼接保留标点: cur.text + next.text (与历史 FunASR 引擎层合并一致).
    对旧缓存已合并行天然幂等: 无「同 speaker 且 gap 小」相邻对则不产生新巨段.
    """
    if not segments:
        return []

    merged: List[TranscriptionSegment] = []
    current = segments[0].model_copy(deep=True)
    # 上一输入段起点 (不论是否已并入 current); 乱序守卫按相邻输入单调, 非组首
    prev_input_start = segments[0].start_time

    for next_seg in segments[1:]:
        # 乱序守卫: 相对上一输入段 start 回退 → 不合并, 直接断开
        if next_seg.start_time < prev_input_start:
            merged.append(current)
            current = next_seg.model_copy(deep=True)
            prev_input_start = next_seg.start_time
            continue

        time_gap = next_seg.start_time - current.end_time
        span = next_seg.end_time - current.start_time
        same_speaker = current.speaker == next_seg.speaker
        gap_ok = time_gap < gap_sec
        span_ok = max_span_sec <= 0 or span <= max_span_sec

        if same_speaker and gap_ok and span_ok:
            current = TranscriptionSegment(
                start_time=current.start_time,
                end_time=max(current.end_time, next_seg.end_time),
                text=current.text + next_seg.text,
                speaker=current.speaker,
                # funasr 路径无 words; 不跨句发明词级时间戳
                words=None,
            )
        else:
            merged.append(current)
            current = next_seg.model_copy(deep=True)

        prev_input_start = next_seg.start_time

    merged.append(current)
    return merged


def project_result_nospk(result: TranscriptionResult) -> TranscriptionResult:
    """把 diarized 结果投影成 nospk 形态 (纯函数, 不动入参, 幂等).

    speaker → None (null = 未区分, 与"真只有一人"可区分), speakers → [].
    其余字段 (segments 时间/文本/words, duration, metadata 等) 原样保留.
    """
    projected = result.model_copy(deep=True)
    projected.segments = [
        seg.model_copy(update={"speaker": None}) for seg in projected.segments
    ]
    projected.speakers = []
    return projected


def segments_to_srt_text(segments: Iterable[TranscriptionSegment]) -> str:
    """schema 层 segments → SRT 字符串 (引擎中立渲染点).

    speaker 非空 → "SpeakerN:文本" (与 FunASR 字节级对齐, 冒号后无空格);
    speaker=None (nospk) → 纯文本行, 无前缀.
    空文本片段跳过, 索引按非空 segment 重新编号.
    """
    def fmt(ms: int) -> str:
        seconds = ms / 1000
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        msec = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"

    lines: List[str] = []
    idx = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        idx += 1
        start_ms = int(round(seg.start_time * 1000))
        end_ms = int(round(seg.end_time * 1000))
        lines.append(str(idx))
        lines.append(f"{fmt(start_ms)} --> {fmt(end_ms)}")
        lines.append(f"{seg.speaker}:{text}" if seg.speaker else text)
        lines.append("")
    return "\n".join(lines)
