"""Qwen3-Diarize 转录器

调用形状与 FunASRTranscriber 对齐:
    async def transcribe(audio_path, task_id, progress_callback, output_format)
        JSON 模式 -> (TranscriptionResult, raw_result_dict)
        SRT  模式 -> {format, content, file_name, file_hash, duration, processing_time, raw_result}

内部组装 (diarize=True, 默认):
    1. ASR  (src.core.qwen3.asr.run_asr) — 拿全文 text + duration (与 2 并行)
    2. Diarize (src.core.qwen3.diarize.run_diarization_dispatched) — 拿 turns
    3. filter_spurious_speakers — 假说话人碎片归并
    4. apply_cluster_centroid_merge — 过聚 cluster 合并
    5. merge_asr_chunks_and_diarize — ASR chunk 文本切到 turn
    6. apply_short_segment_guard — 微短段 drop / ABA 平滑 / 同 spk 合并
    7. apply_silence_align — 切点吸附最近静音中点
    8. word_align (可选, JSON-only) — MMS CTC-FA 词级时间戳增量挂 segment.words
    9. relabel_segments_by_duration_desc — Speaker ID 按时长降序稳定化

diarize=False (per-request options.diarize, 设计定案 D2/D5/D8):
    1. 仅 ASR (跳 Diarize + 上面 3/4/6/9 speaker 后处理层, 省算力)
    2. 段直接来自 ASR ~40s chunk 时间窗 (无 chunks 时单段全文兜底)
    7/8 照常 (silence_align / word_align 与 speaker 正交)
    出口: TranscriptionSegment.speaker=None, speakers=[], SRT 无 SpeakerN: 前缀

ASR 引擎(libllama.dylib + Metal context) 实例级 lazy 单例, 进程内复用.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Optional, Tuple, Union

from loguru import logger

from src.core.qwen3.asr import build_engine, build_qwen_context, run_asr
from src.core.qwen3.cluster_merge import apply_cluster_centroid_merge
from src.core.qwen3.diarize import (
    _load_audio_mono_16k,
    run_diarization,
    run_diarization_dispatched,
)
from src.core.qwen3.merge import (
    Segment,
    attach_words_to_segments,
    filter_spurious_speakers,
    merge_asr_chunks_and_diarize,
    merge_asr_and_diarize,
    relabel_segments_by_duration_desc,
    segments_to_srt,
    snap_segments_to_silence,
)
from src.core.qwen3.postprocess import apply_short_segment_guard
from src.models.schemas import (
    TranscribeOptions,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)
from src.core.gpu_mem import free_vram_mib, has_headroom, used_vram_mib
from src.core.qwen3.word_align_sidecar import get_word_align_sidecar_client
from src.utils.file_utils import calculate_file_hash
from src.utils.silence_detect import ffmpeg_speech_regions


# TitaNet ONNX 导出图的 Where mask 维硬编码 12288 帧 (x10ms hop = 122.88s),
# >=123s 的单段输入直接 RuntimeError (sherpa wrapper 与 ORT 直跑同边界, 见
# docs/开发/2026-06-10-cluster_merge-extractor-122s崩溃-新session-prompt.md).
# 120s 留安全余量; centroid 语义上分块平均无损 (192 维声纹 120s 已绰绰有余).
MAX_EXTRACTOR_SEGMENT_SEC = 120.0




def _cap_extractor_fn(raw_fn, max_segment_sec: float = MAX_EXTRACTOR_SEGMENT_SEC):
    """把 raw extractor 包成段长受限版, 底层永远收到 <= max_segment_sec 的段.

    超限段切成 n = ceil(dur / max) 个等宽连续窗 (每窗 >= max/2, 不会出现碎窗),
    逐窗 embedding 后平均再 L2 归一化; 窗级 None (太短/越界) 跳过, 全 None 返回 None.

    Args:
        raw_fn: callable (audio_16k, start, end) -> np.ndarray or None.
        max_segment_sec: 单次底层调用允许的最大段长 (秒).
    """
    import math

    import numpy as np

    def capped_fn(audio_16k, start, end):
        sr = 16000
        start = max(0.0, float(start))
        end = min(len(audio_16k) / sr, float(end))
        dur = end - start
        if dur <= max_segment_sec:
            return raw_fn(audio_16k, start, end)
        n = math.ceil(dur / max_segment_sec)
        win = dur / n
        embs = []
        for i in range(n):
            emb = raw_fn(audio_16k, start + i * win, start + (i + 1) * win)
            if emb is not None:
                embs.append(emb)
        if not embs:
            return None
        c = np.mean(np.stack(embs), axis=0)
        norm = float(np.linalg.norm(c))
        return c / norm if norm > 0 else c

    return capped_fn


def build_embedding_extractor_fn(cfg_like):
    """构造 sherpa SpeakerEmbeddingExtractor 包装成 callable.

    返回 callable (audio, start, end) -> np.ndarray or None, 已带段长上限
    (_cap_extractor_fn): >120s 的段自动切窗平均, 避开 TitaNet 122.88s 崩溃.
    extractor 真正 build 是惰性的, 调一次创建后 closure 复用.

    cfg_like: 鸭子类型, 需要 embedding_model / provider 属性 (Qwen3Config 或 self).
    """
    import sherpa_onnx
    import numpy as np

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=cfg_like.embedding_model,
        # D5: 从 cfg 读 num_threads (Qwen3Config Pydantic validator 已解析 "auto" → int)
        # 兜底 4 是 mac default (老 hardcode 值), 万一传入对象没 num_threads 属性时
        num_threads=getattr(cfg_like, "num_threads", 4),
        provider=getattr(cfg_like, "provider", "cpu"),
        debug=False,
    )
    if not cfg.validate():
        raise RuntimeError("embedding extractor config invalid")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def extractor_fn(audio_16k, start, end):
        sr = 16000
        a = int(max(0.0, start) * sr)
        b = int(min(len(audio_16k) / sr, end) * sr)
        if b - a < int(0.3 * sr):
            return None
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sr, waveform=audio_16k[a:b])
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        emb = np.asarray(extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb

    return _cap_extractor_fn(extractor_fn)


def apply_cluster_centroid_merge_to_turns(
    turns: list,
    audio_path: str,
    cfg_like,
    *,
    extractor_fn=None,
) -> tuple[list, list]:
    """加载 audio + 调 apply_cluster_centroid_merge.

    enabled=False 时直接返回原 turns (零开销, 不加载 audio).

    Args:
        turns: List[dict] of {speaker, start, end} (sherpa diarize 输出).
        audio_path: 音频文件路径.
        cfg_like: 鸭子类型, 需要 cluster_merge_* 字段 + embedding_model + provider.
        extractor_fn: 预 build 好的 extractor callable. None 时按 cfg_like 现 build
            (老路径, 每次 task ~5-10s 浪费). 推荐 caller 用 lazy singleton 复用.

    Returns:
        (new_turns, log) — turns 的 speaker 字段被重写, log 是 merge events.
    """
    if not getattr(cfg_like, "cluster_merge_enabled", True):
        return list(turns), []
    audio_16k, _sr = _load_audio_mono_16k(audio_path)
    if extractor_fn is None:
        extractor_fn = build_embedding_extractor_fn(cfg_like)
    # turns 是 List[dict], speaker 字段是 int (sherpa 输出), apply_cluster_centroid_merge
    # 内部用 str(speaker), 我们最后转回 int.
    out, log = apply_cluster_centroid_merge(
        turns,
        extractor_fn,
        audio_16k,
        min_main_share=cfg_like.cluster_merge_min_main_share,
        relabel_threshold=cfg_like.cluster_merge_relabel_threshold,
        main_threshold=cfg_like.cluster_merge_main_threshold,
        dominant_share=cfg_like.cluster_merge_dominant_share,
        dominant_threshold=cfg_like.cluster_merge_dominant_threshold,
        dominant_minor_threshold=getattr(
            cfg_like, "cluster_merge_dominant_minor_threshold", 0.5
        ),
    )
    # speaker 字段转回 int (cluster_merge 内部按 str 比较)
    out = [dict(t, speaker=int(t["speaker"])) for t in out]
    return out, log


def apply_silence_align_to_segments(
    segments: list[Segment],
    audio_path: str,
    audio_duration: float,
    qwen3_config,
) -> tuple[list[Segment], dict]:
    """对 Segment 列表做 silence-aware 切点吸附 (snap-to-silence).

    enabled=False 时直接返回原列表 (零开销, 不跑 ffmpeg).
    ffmpeg silencedetect 在 60min 上 ~2s, 整体 RTF 影响 <1%.

    Args:
        segments: merge_asr_chunks_and_diarize (+short_guard) 后的 Segment 列表.
        audio_path: 已转好的 wav 路径 (worker 内 actual_audio_path).
        audio_duration: 音频总时长 (秒).
        qwen3_config: Qwen3Config 实例 (读 silence_* 字段).

    Returns:
        (new_segments, stats). stats 含 enabled / snapped_starts / snapped_ends 等.
        ffmpeg 失败时返回原列表 + stats 含 error 字段, 不阻塞主流程.
    """
    if not getattr(qwen3_config, "silence_align_enabled", True):
        return segments, {"enabled": False}
    if not segments:
        return segments, {"enabled": True, "skipped": "empty_segments"}
    try:
        speech_regions = ffmpeg_speech_regions(
            audio_path,
            audio_duration,
            noise_db=qwen3_config.silence_vad_noise_db,
            min_silence_sec=qwen3_config.silence_vad_min_silence_sec,
        )
    except Exception as exc:
        # ffmpeg 失败不应阻塞转录, 退化为不做 snap
        return segments, {"enabled": True, "error": str(exc)}
    new_segs, stats = snap_segments_to_silence(
        segments,
        speech_regions,
        audio_duration,
        tolerance=qwen3_config.silence_align_tolerance_sec,
        min_segment_dur=qwen3_config.silence_align_min_segment_dur_sec,
    )
    stats["enabled"] = True
    stats["speech_regions_count"] = len(speech_regions)
    return new_segs, stats


def apply_nospk_split_to_segments(
    segments: list[Segment],
    audio_path: str,
    audio_duration: float,
    qwen3_config,
) -> tuple[list[Segment], dict]:
    """diarize=false 的超长段分层切分 wrapper (照 apply_silence_align_to_segments 形状).

    enabled=False / 空输入 / ffmpeg 异常 → fallback to input, 不阻塞主流程.
    切分策略 (词隙切→静音切→硬切) 见 src/core/qwen3/segment_split.py.
    与 silence_align 各自调一次 ffmpeg silencedetect (60min ~2s, RTF <1%),
    不共享 speech_regions 以保持两层独立可关.

    Args:
        segments: nospk 路径的 Segment 列表 (ASR chunk 出段, 可能已挂 words).
        audio_path: 已转好的 wav 路径 (worker 内 actual_audio_path).
        audio_duration: 音频总时长 (秒).
        qwen3_config: Qwen3Config 实例 (读 nospk_split_* + silence_vad_* 字段).

    Returns:
        (new_segments, stats). stats 含 enabled / split_segments / word_split /
        silence_split / hard_cuts; ffmpeg 失败时含 error 字段.
    """
    if not getattr(qwen3_config, "nospk_split_enabled", True):
        return segments, {"enabled": False}
    if not segments:
        return segments, {"enabled": True, "skipped": "empty_segments"}
    try:
        speech_regions = ffmpeg_speech_regions(
            audio_path,
            audio_duration,
            noise_db=qwen3_config.silence_vad_noise_db,
            min_silence_sec=qwen3_config.silence_vad_min_silence_sec,
        )
    except Exception as exc:
        return segments, {"enabled": True, "error": str(exc)}
    from src.core.qwen3.segment_split import split_long_segments

    new_segs, stats = split_long_segments(
        segments,
        speech_regions=speech_regions,
        audio_duration=audio_duration,
        max_dur=qwen3_config.nospk_split_max_segment_sec,
        min_dur=qwen3_config.nospk_split_min_segment_sec,
    )
    stats["enabled"] = True
    return new_segs, stats


def apply_short_segment_guard_to_segments(
    merged_segments: list[Segment],
    qwen3_config,
) -> tuple[list[Segment], dict]:
    """把 List[Segment] 经 apply_short_segment_guard 后转回 List[Segment].

    enabled=False 时直接返回原列表 (零拷贝).

    Args:
        merged_segments: 来自 merge_asr_chunks_and_diarize 的 List[Segment].
        qwen3_config: Qwen3Config 实例 (读 short_segment_* 字段).

    Returns:
        (new_segments, stats).
    """
    if not qwen3_config.short_segment_guard_enabled:
        return merged_segments, {"enabled": False}
    # 转 dict (speaker int -> str, postprocess 用字符串比较).
    # words 一并透传 (codex #6): postprocess 的 drop/aba/merge 都用 dict(s)/append 保留 dict,
    # pass-through 段的 words 不丢. 若 word_align 在 guard 之后跑则此处 words 恒 None, 无副作用.
    seg_dicts = [
        {
            "start": float(s.start),
            "end": float(s.end),
            "speaker": str(s.speaker),
            "text": s.text,
            "words": s.words,
        }
        for s in merged_segments
    ]
    # max_span_sec: 防单人独白滚成巨段 (issue #1 防御收口); 与 funasr serve 投影共用
    # config.transcription.segment_merge_max_span_sec. 默认 120s; 函数签名默认 0=无上限
    # 保持单元测试/旧调用行为不变.
    from src.core.config import config as _cfg
    out_dicts, stats = apply_short_segment_guard(
        seg_dicts,
        enabled=True,
        short_drop_sec=qwen3_config.short_segment_drop_sec,
        aba_max_mid_sec=qwen3_config.short_segment_aba_max_mid_sec,
        merge_same=qwen3_config.short_segment_merge_same,
        max_span_sec=_cfg.transcription.segment_merge_max_span_sec,
    )
    # 转回 Segment (speaker str -> int, words 透传)
    out_segments = [
        Segment(
            start=float(d["start"]),
            end=float(d["end"]),
            speaker=int(d["speaker"]),
            text=d["text"],
            words=d.get("words"),
        )
        for d in out_dicts
    ]
    return out_segments, stats


class Qwen3DiarizeTranscriber:
    """Qwen3-ASR + sherpa Speaker Diarization 离线转录器

    参数注入式: 模型路径 / preset 由调用方传入(典型从 config 读), 类本身不绑 config.
    """

    # CUDA word_align poison flag — **class attr (进程/pool 级共享, codex #8)**:
    # 第一次 CUDA word_align 资源错误 (OOM/CUBLAS) 后置 True, 该进程内所有实例的 word_align
    # 余生走 CPU, 不再赌 CUDA (避免反复 thrash + 不健康 CUDA 上下文). in-proc CUDA pool 的
    # N 个实例共享同一进程 → 共享此 flag = pool 级 poison. 重启服务恢复 CUDA.
    # (Mac file-based pool 每 task 一进程, poison 无意义, codex #9 — class attr 各进程独立.)
    _cuda_word_align_poisoned: bool = False

    def __init__(
        self,
        asr_model_dir: str,
        segmentation_model: str,
        embedding_model: str,
        num_speakers: Optional[int] = None,
        cluster_threshold: float = 0.9,
        # 默认 4 跟 Qwen3Config 默认 "auto" 在 Mac 解析后一致 (pre-existing 8 是 inconsistency, A2 修)
        num_threads: int = 4,
        provider: str = "cpu",
        language: str = "Chinese",
        temperature: float = 0.4,
        # filter_spurious 参数
        spurious_min_total: float = 2.0,
        spurious_min_share: float = 0.01,
        # PR2 short-segment guard 参数 (与 Qwen3Config 字段对齐, 鸭子类型供 helper 读)
        short_segment_guard_enabled: bool = True,
        short_segment_drop_sec: float = 1.5,
        short_segment_aba_max_mid_sec: float = 1.5,
        short_segment_merge_same: bool = True,
        # PR3 cluster centroid merge 参数
        cluster_merge_enabled: bool = True,
        cluster_merge_min_main_share: float = 0.03,
        cluster_merge_relabel_threshold: float = 0.55,
        cluster_merge_main_threshold: float = 0.78,
        cluster_merge_dominant_share: float = 0.6,
        cluster_merge_dominant_threshold: float = 0.6,
        cluster_merge_dominant_minor_threshold: float = 0.5,
        # silence-aware 切点对齐 (spike 405abf6)
        silence_align_enabled: bool = True,
        silence_align_tolerance_sec: float = 2.0,
        silence_align_min_segment_dur_sec: float = 0.1,
        silence_vad_noise_db: str = "-25dB",
        silence_vad_min_silence_sec: float = 0.20,
        # nospk 分层切段 (diarize 开关 D5)
        nospk_split_enabled: bool = True,
        nospk_split_max_segment_sec: float = 12.0,
        nospk_split_min_segment_sec: float = 1.5,
        # 词级时间戳 word_align (MMS CTC-FA, 默认关)
        word_align_enabled: bool = False,
        word_align_language: str = "chi",
        word_align_model_path: str = "./models/qwen3_diarize/ctc_forced_aligner/model.onnx",
        word_align_provider: str = "auto",
        word_align_batch_size: int = 16,
        word_align_cuda_batch_size: int = 1,
        word_align_preflight_enabled: bool = True,
        word_align_preflight_free_mib: int = 4608,
        word_align_sidecar_enabled: bool = True,
    ):
        self.asr_model_dir = asr_model_dir
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.num_speakers = num_speakers
        self.cluster_threshold = cluster_threshold
        self.num_threads = num_threads
        self.provider = provider
        self.language = language
        self.temperature = temperature
        self.spurious_min_total = spurious_min_total
        self.spurious_min_share = spurious_min_share
        self.short_segment_guard_enabled = short_segment_guard_enabled
        self.short_segment_drop_sec = short_segment_drop_sec
        self.short_segment_aba_max_mid_sec = short_segment_aba_max_mid_sec
        self.short_segment_merge_same = short_segment_merge_same
        self.cluster_merge_enabled = cluster_merge_enabled
        self.cluster_merge_min_main_share = cluster_merge_min_main_share
        self.cluster_merge_relabel_threshold = cluster_merge_relabel_threshold
        self.cluster_merge_main_threshold = cluster_merge_main_threshold
        self.cluster_merge_dominant_share = cluster_merge_dominant_share
        self.cluster_merge_dominant_threshold = cluster_merge_dominant_threshold
        self.cluster_merge_dominant_minor_threshold = cluster_merge_dominant_minor_threshold
        self.silence_align_enabled = silence_align_enabled
        self.silence_align_tolerance_sec = silence_align_tolerance_sec
        self.silence_align_min_segment_dur_sec = silence_align_min_segment_dur_sec
        self.silence_vad_noise_db = silence_vad_noise_db
        self.silence_vad_min_silence_sec = silence_vad_min_silence_sec
        self.nospk_split_enabled = nospk_split_enabled
        self.nospk_split_max_segment_sec = nospk_split_max_segment_sec
        self.nospk_split_min_segment_sec = nospk_split_min_segment_sec
        self.word_align_enabled = word_align_enabled
        self.word_align_language = word_align_language
        self.word_align_model_path = word_align_model_path
        self.word_align_provider = word_align_provider
        self.word_align_batch_size = word_align_batch_size
        self.word_align_cuda_batch_size = word_align_cuda_batch_size
        self.word_align_preflight_enabled = word_align_preflight_enabled
        self.word_align_preflight_free_mib = word_align_preflight_free_mib
        self.word_align_sidecar_enabled = word_align_sidecar_enabled

        # 引擎单例 — 第一次 transcribe 时构造, 后续复用
        self._asr_engine = None
        # PR3 sherpa embedding extractor 单例 — 同 worker 多 task 复用, 省 ~5-10s/task
        self._embedding_extractor_fn = None
        # word_align MMS aligner 单例 — 同 worker 多 task 复用 MMS ONNX session
        self._word_aligner = None
        # CPU fallback aligner (lazy) — CUDA poison 后或资源不足时走它
        self._word_aligner_cpu = None

    # ==================== 引擎管理 ====================

    def _ensure_embedding_extractor_fn(self):
        """惰性构造 sherpa embedding extractor fn (per-worker singleton).

        首次调用 ~3-5s (build sherpa SpeakerEmbeddingExtractor + onnx warm),
        后续 task 复用同一 callable, 省每次 build 的开销.
        """
        if self._embedding_extractor_fn is None:
            logger.info(
                f"首次构造 sherpa embedding extractor(embedding_model={self.embedding_model})"
            )
            self._embedding_extractor_fn = build_embedding_extractor_fn(self)
        return self._embedding_extractor_fn

    def _ensure_word_aligner(self):
        """惰性构造 MMS word_align aligner (per-worker singleton).

        首次调用 build MMS ONNX session (~1-2s warm), 后续 task 复用. provider="auto"
        在 WordAligner 内按 runtime 解析 (Mac/Cpu→CPU EP, Cuda→CUDA EP).
        """
        if self._word_aligner is None:
            from src.core.qwen3.word_align import WordAligner

            logger.info(
                f"首次构造 word_align MMS aligner(model_path={self.word_align_model_path})"
            )
            self._word_aligner = WordAligner(
                model_path=self.word_align_model_path,
                provider=self.word_align_provider,
                language=self.word_align_language,
                batch_size=self.word_align_batch_size,
                cuda_batch_size=self.word_align_cuda_batch_size,
            )
        return self._word_aligner

    def _ensure_word_aligner_cpu(self):
        """惰性构造 CPU word_align aligner (per-worker 单例) — CUDA poison / fallback 用.

        显式 provider="cpu" + batch_size=word_align_batch_size (16), 与 primary (可能 CUDA)
        互不影响. CPU session 独立于 CUDA 上下文, OOM 后仍可用.
        """
        if self._word_aligner_cpu is None:
            from src.core.qwen3.word_align import WordAligner

            logger.info("构造 CPU word_align aligner (CUDA fallback/poison 路径)")
            self._word_aligner_cpu = WordAligner(
                model_path=self.word_align_model_path,
                provider="cpu",
                language=self.word_align_language,
                batch_size=self.word_align_batch_size,
                cuda_batch_size=self.word_align_cuda_batch_size,
            )
        return self._word_aligner_cpu

    def _poison_cuda_word_align(self, task_id: str):
        """OOM 后 poison pool 级 CUDA word_align (决策 2A-CQ + 4A).

        - 置 class flag (进程/pool 内所有实例后续走 CPU)
        - dispose CUDA aligner session 尝试回收显存 (决策 4A, ORT 未必真还, 打 nvidia-smi delta 观测)
        """
        Qwen3DiarizeTranscriber._cuda_word_align_poisoned = True
        before = used_vram_mib()
        try:
            if self._word_aligner is not None:
                self._word_aligner.close()
                self._word_aligner = None
        except Exception as exc:  # dispose 失败不致命
            logger.warning(f"[{task_id}] dispose CUDA word_align session 失败: {exc}")
        after = used_vram_mib()
        if before is not None and after is not None:
            logger.warning(
                f"[{task_id}] CUDA word_align POISONED — 该 worker 余生走 CPU. "
                f"dispose 显存 delta: {before}→{after} MiB (回收 {before - after} MiB; "
                f"ORT BFCArena 未必真还, 仅观测)"
            )
        else:
            logger.warning(
                f"[{task_id}] CUDA word_align POISONED — 该 worker 余生走 CPU "
                f"(已 dispose session; nvidia-smi 不可用, 显存回收量未测)"
            )

    def _word_align_segments(self, audio_path, asr_chunks, merged_segments, effective_lang, task_id):
        """词级时间戳挂段 + CUDA→CPU fallback + OOM poison. 同步, 在 executor 跑.

        返回 (merged_segments, stats). 流程:
          1. 已 poison → 直接 CPU aligner, 不赌 CUDA
          2. 否则试 primary aligner (provider=auto, 可能 CUDA)
          3. primary 抛 CUDA 资源错误 (is_resource_error) 且 primary 是 CUDA → poison + dispose + CPU 重试
          4. CPU 也失败 / 非资源错误 → 段不带词 + error, 整请求不挂 (decisions A/2A-CQ)
        """
        from src.core.qwen3.word_align import is_resource_error

        cls = Qwen3DiarizeTranscriber
        # 音频惰性加载: sidecar 命中路径主进程不需要 (sidecar 自读文件), 仅进程内对齐
        # (CPU 兜底 / 非 CUDA / sidecar 关) 才加载, 避免 sidecar happy path 白读一遍.
        _audio_cache: dict = {}

        def _audio():
            if "a" not in _audio_cache:
                _audio_cache["a"], _ = _load_audio_mono_16k(audio_path)
            return _audio_cache["a"]

        def _run(aligner, provider_label, extra=None):
            words, stats = aligner.align_chunks(_audio(), asr_chunks, language=effective_lang)
            segs = attach_words_to_segments(merged_segments, words)
            stats = {
                **stats, "enabled": True, "language": effective_lang,
                "provider": provider_label, **(extra or {}),
            }
            if stats.get("failed_windows"):
                logger.warning(
                    f"[{task_id}] word_align 部分 window 失败 "
                    f"({stats['failed_windows']}/{stats.get('total_windows', 0)}), 该段不带词"
                )
            logger.info(
                f"[{task_id}] word_align: provider={provider_label} "
                f"windows={stats.get('total_windows', 0)} failed={stats.get('failed_windows', 0)} "
                f"words={stats.get('total_words', 0)} lang={effective_lang}"
            )
            return segs, stats

        # 0) Lane 2 (#18): cuda runtime + sidecar 启用 → CUDA word_align 走独立 sidecar 进程
        #    (进程内永不建 CUDA session, codex #7 硬切; sidecar 用完 idle TTL 自杀释放 VRAM).
        #    非 CUDA (Mac/CPU) 落到下面进程内路径 (A4 作用域). 主进程 in-proc poison 在此分支
        #    无意义 (CUDA 不在主进程跑), sidecar 自己退休 (codex #8).
        if self.word_align_sidecar_enabled and self._word_align_primary_is_cuda():
            return self._word_align_via_sidecar(
                audio_path, asr_chunks, merged_segments, effective_lang, task_id, _run,
            )

        # 1) 已 poison → 直走 CPU
        if cls._cuda_word_align_poisoned:
            try:
                return _run(self._ensure_word_aligner_cpu(), "cpu", {"cuda_poisoned": True})
            except Exception as exc:
                logger.warning(f"[{task_id}] word_align CPU 失败 (CUDA 已 poison), 段不带词: {exc}")
                return merged_segments, {
                    "enabled": True, "error": f"cpu_fail: {exc}",
                    "language": effective_lang, "provider": "cpu",
                }

        # 2) 试 primary (可能 CUDA)
        aligner = None
        try:
            aligner = self._ensure_word_aligner()  # 构造便宜, session 在 align 时才 build
            # 2.5) VRAM preflight (#17): primary 解析成 CUDA 且显存不足 → 直走 CPU,
            #      不等 CUDA OOM (OOM 后 CUDA 上下文可能已不健康). 探不到/preflight 关
            #      → 不误杀, 照走 CUDA 交给 OOM fallback (codex #11, TOCTOU).
            #      primary 非 CUDA (Mac/CPU) 跳过, 省一次 nvidia-smi.
            if self.word_align_preflight_enabled and aligner.is_cuda:
                free = free_vram_mib()
                if not has_headroom(free, self.word_align_preflight_free_mib):
                    logger.warning(
                        f"[{task_id}] word_align CUDA preflight 显存不足 "
                        f"(free={free} MiB < {self.word_align_preflight_free_mib} MiB), "
                        f"走 CPU (不等 OOM)"
                    )
                    return _run(
                        self._ensure_word_aligner_cpu(), "cpu",
                        {"preflight_skipped_cuda": True, "free_vram_mib": free},
                    )
                logger.info(
                    f"[{task_id}] word_align CUDA preflight OK "
                    f"(free={free} MiB ≥ {self.word_align_preflight_free_mib} MiB)"
                )
            return _run(aligner, aligner.effective_provider)
        except Exception as exc:
            # 3) CUDA 资源错误 → poison + 转 CPU (用返回的 aligner 引用判 is_cuda,
            #    不靠 self._word_aligner: 测试可能 patch _ensure_word_aligner 不写实例字段)
            if is_resource_error(exc) and aligner is not None and aligner.is_cuda:
                logger.warning(f"[{task_id}] word_align CUDA 资源错误, poison + 转 CPU: {exc}")
                self._poison_cuda_word_align(task_id)
                try:
                    return _run(self._ensure_word_aligner_cpu(), "cpu", {"cuda_oom_fallback": True})
                except Exception as cpu_exc:
                    logger.warning(f"[{task_id}] word_align CPU fallback 也失败, 段不带词: {cpu_exc}")
                    return merged_segments, {
                        "enabled": True, "error": f"cuda_oom+cpu_fail: {cpu_exc}",
                        "language": effective_lang, "provider": "cpu",
                    }
            # 4) 非资源错误 / primary 非 CUDA → 段不带词
            logger.warning(f"[{task_id}] word_align 失败, 段不带词: {exc}")
            return merged_segments, {"enabled": True, "error": str(exc), "language": effective_lang}

    def _word_align_primary_is_cuda(self) -> bool:
        """primary word_align provider 是否解析成 CUDA (决定走 sidecar 还是进程内).

        不构造 WordAligner (省), 直接解析 provider="auto" → runtime EP. cuda runtime
        → True (走 sidecar); Mac/Cpu → False (进程内 CPU, A4 作用域).
        """
        from src.core.qwen3.word_align import resolve_word_align_providers

        providers = resolve_word_align_providers(self.word_align_provider)
        return any("CUDA" in p for p in providers)

    @staticmethod
    def _chunk_to_dict(c) -> dict:
        """ASRChunkItem / dict → {start, end, text} (sidecar JSON 传输, 不传对象)."""
        if isinstance(c, dict):
            return {"start": c.get("start"), "end": c.get("end"), "text": c.get("text")}
        return {"start": getattr(c, "start", None), "end": getattr(c, "end", None),
                "text": getattr(c, "text", None)}

    def _word_align_via_sidecar(
        self, audio_path, asr_chunks, merged_segments, effective_lang, task_id, _run,
    ):
        """Lane 2: CUDA word_align 经独立 sidecar 进程出词 + 降级链 (A3).

        preflight 不足 → 主进程 CPU (连 sidecar 都不拉, 省启动). sidecar
        超时/资源错误/不可用 → 主进程 CPU 兜底; 普通对齐错误 → 无词 (sidecar 健康).
        进程内永不建 CUDA session (codex #7). _run 用于 CPU 兜底 (复用进程内 CPU aligner).
        """
        from src.core.qwen3.word_align_sidecar import (
            SidecarAlignError,
            SidecarResourceError,
            SidecarTimeout,
            SidecarUnavailable,
        )

        # preflight (#17 复用): 显存不足 → 主进程 CPU, 不拉 CUDA sidecar (codex #11 TOCTOU:
        # 仍非保证, sidecar 内 OOM 退休 + CPU 兜底兜住).
        if self.word_align_preflight_enabled:
            free = free_vram_mib()
            if not has_headroom(free, self.word_align_preflight_free_mib):
                logger.warning(
                    f"[{task_id}] word_align CUDA preflight 显存不足 "
                    f"(free={free} MiB < {self.word_align_preflight_free_mib} MiB), "
                    f"走主进程 CPU (不拉 sidecar)"
                )
                return _run(
                    self._ensure_word_aligner_cpu(), "cpu",
                    {"preflight_skipped_cuda": True, "free_vram_mib": free},
                )

        client = get_word_align_sidecar_client()
        chunks_json = [self._chunk_to_dict(c) for c in asr_chunks]
        try:
            words, stats = client.align(audio_path, chunks_json, effective_lang)
        except (SidecarTimeout, SidecarResourceError, SidecarUnavailable) as exc:
            # sidecar 不可用/超时/退休 → 主进程 CPU 兜底 (A3)
            logger.warning(
                f"[{task_id}] word_align sidecar 失败 ({type(exc).__name__}), 转主进程 CPU: {exc}"
            )
            try:
                return _run(
                    self._ensure_word_aligner_cpu(), "cpu",
                    {"sidecar_fallback": type(exc).__name__},
                )
            except Exception as cpu_exc:
                logger.warning(f"[{task_id}] word_align sidecar→CPU 兜底也失败, 段不带词: {cpu_exc}")
                return merged_segments, {
                    "enabled": True, "error": f"sidecar+cpu_fail: {cpu_exc}",
                    "language": effective_lang, "provider": "cpu",
                }
        except SidecarAlignError as exc:
            # sidecar 健康但普通对齐失败 → 无词 (不必 CPU 兜底)
            logger.warning(f"[{task_id}] word_align sidecar 对齐失败 (非资源), 段不带词: {exc}")
            return merged_segments, {
                "enabled": True, "error": str(exc),
                "language": effective_lang, "provider": "cuda_sidecar",
            }

        segs = attach_words_to_segments(merged_segments, words)
        stats = {
            **stats, "enabled": True, "language": effective_lang, "provider": "cuda_sidecar",
        }
        logger.info(
            f"[{task_id}] word_align via sidecar: "
            f"windows={stats.get('total_windows', 0)} failed={stats.get('failed_windows', 0)} "
            f"words={stats.get('total_words', 0)} lang={effective_lang}"
        )
        return segs, stats

    def _ensure_engine(self):
        """惰性构造 ASR 引擎(libllama + Metal context 加载耗时, 进程内复用)"""
        if self._asr_engine is None:
            logger.info(f"首次构造 Qwen3-ASR 引擎(asr_model_dir={self.asr_model_dir})")
            self._asr_engine = build_engine(self.asr_model_dir)
        return self._asr_engine

    async def initialize(self):
        """提前触发引擎加载(可选, 不调也会在 transcribe 时 lazy 加载).

        PR4 follow-up: 同时 eager warm sherpa embedding extractor, 让第一个
        cluster_merge task 不含 3-5s 的 extractor build overhead. extractor
        warmup 独立 try/except, 失败不阻塞 ASR engine 可用性 (生产时 sherpa
        模型缺失/加载慢等场景, ASR 仍能跑, cluster_merge 在 task 内 lazy 重试).

        ORT-CUDA sprint: 启动前打一行 runtime summary, 给运维确认当前 runtime
        + diarize backend + sherpa num_threads, 排查环境问题用. Mac 上保持 no-op
        (MacRuntime.validate() 不抛), Linux + CUDA 上 validate() 会 fail-fast.
        """
        from src.core.runtime import describe_runtime, detect_runtime

        runtime = detect_runtime()
        runtime.validate()
        logger.info(f"[qwen3] {describe_runtime(runtime)}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ensure_engine)
        try:
            await loop.run_in_executor(None, self._ensure_embedding_extractor_fn)
        except Exception as exc:
            logger.warning(
                f"sherpa embedding extractor 预热失败, 留给首次 cluster_merge lazy 加载: {exc}"
            )

    # ==================== 进度回调 ====================

    @staticmethod
    async def _report_progress(callback: Optional[Callable], pct: int, task_id: str):
        if callback is None:
            return
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(pct)
            else:
                callback(pct)
            logger.debug(f"[{task_id}] 进度: {pct}%")
        except Exception as e:
            logger.warning(f"[{task_id}] progress_callback 异常 (忽略): {e}")

    # ==================== 主入口 ====================

    async def transcribe(
        self,
        audio_path: str,
        task_id: str,
        progress_callback: Optional[Callable] = None,
        output_format: str = "json",
        options: Optional["TranscribeOptions"] = None,
    ) -> Union[Tuple[TranscriptionResult, dict], dict]:
        """跑一遍 ASR + Diarize + Merge, 返回 (TranscriptionResult, raw_result) 或 SRT dict

        options: per-request 转录选项 (E3 收拢):
        - language: 识别语言 ISO 码 (chi/eng/jpn/kor…), 驱动 word_align 词级时间戳
          语言; None 时走 self.word_align_language 兜底 (见 transcribe 内 word_align 层).
        - diarize: False 时跳过 diarize + speaker 后处理层, 输出不含说话人区分.
        """
        options = options or TranscribeOptions()
        language = options.language
        context = build_qwen_context(options.terms)
        start_time = time.time()
        loop = asyncio.get_event_loop()

        await self._report_progress(progress_callback, 5, task_id)

        # 引擎在线程池 lazy 加载, 不阻塞 event loop
        engine = await loop.run_in_executor(None, self._ensure_engine)

        await self._report_progress(progress_callback, 10, task_id)

        do_diarize = options.diarize

        if do_diarize:
            # ASR + Diarize 并行(PoC 实测真并行节省 ~50% wall time)
            asr_future = loop.run_in_executor(
                None,
                lambda: run_asr(
                    audio_path,
                    engine,
                    language=self.language,
                    temperature=self.temperature,
                    context=context,
                ),
            )
            diarize_future = loop.run_in_executor(
                None,
                lambda: run_diarization_dispatched(
                    audio_path,
                    segmentation_model=self.segmentation_model,
                    embedding_model=self.embedding_model,
                    num_speakers=self.num_speakers,
                    cluster_threshold=self.cluster_threshold,
                    num_threads=self.num_threads,
                    provider=self.provider,
                ),
            )

            # progress 在等待时定期报(简化: gather 完成时直接跳到 80)
            asr_result, turns = await asyncio.gather(asr_future, diarize_future)
        else:
            # diarize=False (D2/D5): 真跳过 diarize + speaker 后处理层, 省算力.
            # config num_speakers 此时闲置 (D7: num_speakers 本次不 per-request 化).
            if self.num_speakers is not None:
                logger.info(
                    f"[{task_id}] diarize=false: config num_speakers={self.num_speakers} 闲置不生效"
                )
            asr_result = await loop.run_in_executor(
                None,
                lambda: run_asr(
                    audio_path,
                    engine,
                    language=self.language,
                    temperature=self.temperature,
                    context=context,
                ),
            )
            turns = []
        await self._report_progress(progress_callback, 80, task_id)

        if do_diarize:
            # 过滤假说话人(短时长 spurious cluster)
            filtered_turns = filter_spurious_speakers(
                turns,
                min_speaker_total=self.spurious_min_total,
                min_speaker_share=self.spurious_min_share,
                audio_duration=asr_result.duration,
            )

            # PR3: cluster centroid merge — 多人场景把过聚的 cluster 合并 (在切文本前做, 减少误派文本)
            if self.cluster_merge_enabled:
                try:
                    # extractor lazy singleton: 同 worker 多 task 复用, 省 ~5-10s/task
                    extractor_fn = await loop.run_in_executor(
                        None, self._ensure_embedding_extractor_fn
                    )
                    filtered_turns, cluster_log = apply_cluster_centroid_merge_to_turns(
                        filtered_turns, audio_path, self, extractor_fn=extractor_fn
                    )
                    merge_events = [e for e in cluster_log if e.get("action", "").startswith("main_merged") or e.get("action") == "minor_to_main"]
                    if merge_events:
                        logger.info(
                            f"[{task_id}] cluster_merge: {len(merge_events)} merge events, "
                            f"final clusters={len(set(t['speaker'] for t in filtered_turns))}"
                        )
                except Exception as exc:
                    logger.warning(f"[{task_id}] cluster_merge 失败, 跳过: {exc}")

            # 文本切分到 turns。优先使用 Qwen3-ASR 内部 chunk 时间窗，避免整段线性切字漂移。
            if asr_result.chunks:
                merged_segments = merge_asr_chunks_and_diarize(asr_result.chunks, filtered_turns)
            else:
                merged_segments = merge_asr_and_diarize(asr_result.text, filtered_turns)

            # PR2: short-segment guard 后处理 (drop tiny / ABA 平滑 / 合并同 spk)
            # 由 config 控制开关与阈值, 默认全开. SRT/JSON 都走 guard 后的 segments.
            merged_segments, guard_stats = apply_short_segment_guard_to_segments(
                merged_segments, self
            )
            if guard_stats.get("enabled"):
                logger.info(
                    f"[{task_id}] short-guard: drop={guard_stats.get('drop', {}).get('dropped_total', 0)} "
                    f"aba={guard_stats.get('aba', {}).get('changed', 0)} "
                    f"merge_same={guard_stats.get('merge', {}).get('merged', 0)} "
                    f"final_segments={len(merged_segments)}"
                )
        else:
            # diarize=False: 无 turn 边界, 段直接来自 ASR ~40s chunk 时间窗
            # (无 chunks 时单段全文兜底). 内部 speaker 恒 0 (Segment.speaker:int
            # 永不为 None, T-D #3), null 只在出口转换层出现. 超长段的分层切分
            # (词级→静音) 由 split_long_segments 在 silence_align 后接管.
            filtered_turns = []
            chunk_items = [c for c in asr_result.chunks if getattr(c, "text", "")]
            if chunk_items:
                merged_segments = [
                    Segment(
                        start=float(c.start),
                        end=float(c.end),
                        speaker=0,
                        text=str(c.text),
                    )
                    for c in sorted(chunk_items, key=lambda c: (float(c.start), float(c.end)))
                ]
            elif asr_result.text:
                merged_segments = [
                    Segment(start=0.0, end=float(asr_result.duration), speaker=0, text=asr_result.text)
                ]
            else:
                merged_segments = []

        # silence-aware 切点对齐 (spike 405abf6): ffmpeg silencedetect 把段切点
        # 吸附到最近静音中点. 60s podcast align_ratio +19pp, 60min long +33pp.
        # 放在 short-guard 之后, 在干净段上吸附. ffmpeg 调用走线程池避免阻塞.
        # ffmpeg 失败时 helper 内部 fallback 为原 segments, 不阻塞主流程.
        merged_segments, silence_stats = await loop.run_in_executor(
            None,
            apply_silence_align_to_segments,
            merged_segments,
            audio_path,
            asr_result.duration,
            self,
        )
        if silence_stats.get("enabled"):
            if silence_stats.get("error"):
                logger.warning(
                    f"[{task_id}] silence-align: ffmpeg 失败, 跳过 ({silence_stats['error']})"
                )
            else:
                logger.info(
                    f"[{task_id}] silence-align: "
                    f"snap_starts={silence_stats.get('snapped_starts', 0)}/{silence_stats.get('total_starts', 0)} "
                    f"snap_ends={silence_stats.get('snapped_ends', 0)}/{silence_stats.get('total_ends', 0)} "
                    f"skip_zero={silence_stats.get('skipped_zero_dur', 0)} "
                    f"tol={silence_stats.get('tolerance_sec')}s "
                    f"speech_regions={silence_stats.get('speech_regions_count', 0)}"
                )

        # 词级时间戳 (word_align, MMS-300M CTC-FA): 在干净段上增量挂词, 不替换段边界.
        # 挂在 silence_align 之后、relabel 之前 (段已干净). 决策 1A: 读 options.word_align
        # (effective 值, enqueue 已解析 请求>config). 关 → 段 words 恒 None (向后兼容).
        # JSON-only (SRT 不带词, 跳过省 RTF). CUDA OOM → poison + CPU fallback (决策 A/2A-CQ),
        # 任何失败都不阻塞转录: 段照常出. 全部封装在 _word_align_segments (executor 跑).
        do_word_align = options.word_align
        word_align_stats: dict = {"enabled": False}
        if do_word_align and output_format == "json":
            await self._report_progress(progress_callback, 85, task_id)
            effective_lang = language or self.word_align_language
            merged_segments, word_align_stats = await loop.run_in_executor(
                None,
                self._word_align_segments,
                audio_path, asr_result.chunks, merged_segments, effective_lang, task_id,
            )

        # nospk 分层切段 (D5+T1, 仅 diarize=False): 超长 chunk 段切成字幕可用粒度.
        # 挂在 word_align 之后 — JSON 路径段上已挂 words 可词隙精确切;
        # SRT 路径无 words (word_align JSON-only 不变量) 永远走静音 fallback.
        if not do_diarize:
            merged_segments, nospk_split_stats = await loop.run_in_executor(
                None,
                apply_nospk_split_to_segments,
                merged_segments,
                audio_path,
                asr_result.duration,
                self,
            )
            if nospk_split_stats.get("enabled"):
                if nospk_split_stats.get("error"):
                    logger.warning(
                        f"[{task_id}] nospk-split: ffmpeg 失败, 跳过 ({nospk_split_stats['error']})"
                    )
                else:
                    logger.info(
                        f"[{task_id}] nospk-split: "
                        f"split={nospk_split_stats.get('split_segments', 0)} "
                        f"produced={nospk_split_stats.get('produced_segments', 0)} "
                        f"word={nospk_split_stats.get('word_split', 0)} "
                        f"silence={nospk_split_stats.get('silence_split', 0)} "
                        f"hard={nospk_split_stats.get('hard_cuts', 0)} "
                        f"final_segments={len(merged_segments)}"
                    )
        else:
            nospk_split_stats = {"enabled": False, "skipped": "diarize_on"}

        # Speaker ID 稳定化: 把内部 raw cluster int 按总时长降序重映射成 0/1/2/...,
        # 让下游 f"Speaker{i+1}" 输出的 Speaker1 始终是说话最多的人, 跨 backend
        # (ort_cuda / sherpa) / 跨平台 (cuda / Mac) 一致. 见 docs/开发/gpu加速/
        # 2026-05-22-cuda-diarize-accuracy.md 改进建议 §1.
        # relabel 用 dataclasses.replace, 自动透传 word_align 挂的 words.
        # diarize=False 跳过 (speaker 层之一, 内部恒 0 无需重映射).
        if do_diarize:
            merged_segments = relabel_segments_by_duration_desc(merged_segments)

        await self._report_progress(progress_callback, 90, task_id)

        file_hash = await calculate_file_hash(audio_path)
        processing_time = time.time() - start_time

        raw_result = {
            "asr_text": asr_result.text,
            "asr_rtf": asr_result.rtf,
            "asr_duration": asr_result.duration,
            "asr_elapsed": asr_result.elapsed,
            "asr_chunks": [
                {
                    "index": c.index,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                }
                for c in asr_result.chunks
            ],
            "turns": turns,
            "filtered_turns": filtered_turns,
            "word_align": word_align_stats,
            "diarize": do_diarize,
            "nospk_split": nospk_split_stats,
            "engine": "qwen3",
        }

        # 转 TranscriptionSegment, speaker int -> "Speaker{i+1}" (JSON / SRT 共用).
        # SRT 模式也携真 segments: 缓存层存真 segments 才能支撑 SRT 缓存命中重建
        # (qwen3 raw_result 无 sentence_info, funasr 私有重建路径会返回空, 见 T-B)
        # 与后续 diarize=false 投影 (T-A) 的 segments 重渲染.
        # word_align 挂的 s.words (内部 dict) 转成 WordTimestamp; confidence 暂留 None
        # (MMS score 是逐帧 log-prob 求和, 非 0-1 校准置信度, 不直接当 confidence 暴露).
        segments = [
            TranscriptionSegment(
                start_time=round(s.start, 2),
                end_time=round(s.end, 2),
                text=s.text,
                # diarize=False 出口转 null (D8): 未做说话人区分, 与"真只有一人"可区分
                speaker=f"Speaker{s.speaker + 1}" if do_diarize else None,
                words=(
                    [
                        WordTimestamp(
                            text=w["text"],
                            start=round(float(w["start"]), 3),
                            end=round(float(w["end"]), 3),
                            confidence=None,
                        )
                        for w in s.words
                    ]
                    if s.words
                    else None
                ),
            )
            for s in merged_segments
        ]
        # diarize=False: speakers 恒 [] (投影/fresh 两路保证一致, T-D #3)
        speakers = sorted(set(seg.speaker for seg in segments)) if do_diarize else []

        if output_format == "srt":
            srt_content = segments_to_srt(merged_segments, include_speaker=do_diarize)
            await self._report_progress(progress_callback, 100, task_id)
            logger.info(
                f"[{task_id}] Qwen3 转录完成 (SRT): "
                f"duration={asr_result.duration:.2f}s, "
                f"耗时={processing_time:.2f}s, "
                f"RTF={processing_time / asr_result.duration:.3f}"
            )
            return {
                "format": "srt",
                "content": srt_content,
                "file_name": os.path.basename(audio_path),
                "file_hash": file_hash,
                "duration": asr_result.duration,
                "processing_time": processing_time,
                "raw_result": raw_result,
                "segments": segments,
            }

        transcription_result = TranscriptionResult(
            task_id=task_id,
            file_name=os.path.basename(audio_path),
            file_hash=file_hash,
            duration=asr_result.duration,
            segments=segments,
            speakers=speakers,
            processing_time=processing_time,
        )

        await self._report_progress(progress_callback, 100, task_id)
        logger.info(
            f"[{task_id}] Qwen3 转录完成 (JSON): "
            f"segments={len(segments)}, speakers={len(speakers)}, "
            f"duration={asr_result.duration:.2f}s, 耗时={processing_time:.2f}s"
        )
        return (transcription_result, raw_result)


# ==================== 全局单例(C3 配置接通后从 config 读路径) ====================

_qwen3_singleton: Optional[Qwen3DiarizeTranscriber] = None


def get_qwen3_transcriber() -> Qwen3DiarizeTranscriber:
    """获取 Qwen3 转录器单例 (从 config.qwen3 读模型路径 + preset)"""
    global _qwen3_singleton
    if _qwen3_singleton is not None:
        return _qwen3_singleton

    from src.core.config import config

    q = config.qwen3
    _qwen3_singleton = Qwen3DiarizeTranscriber(
        asr_model_dir=q.asr_model_dir,
        segmentation_model=q.segmentation_model,
        embedding_model=q.embedding_model,
        num_speakers=q.num_speakers,
        cluster_threshold=q.cluster_threshold,
        num_threads=q.num_threads,
        provider=q.provider,
        language=q.language,
        temperature=q.temperature,
        short_segment_guard_enabled=q.short_segment_guard_enabled,
        short_segment_drop_sec=q.short_segment_drop_sec,
        short_segment_aba_max_mid_sec=q.short_segment_aba_max_mid_sec,
        short_segment_merge_same=q.short_segment_merge_same,
        cluster_merge_enabled=q.cluster_merge_enabled,
        cluster_merge_min_main_share=q.cluster_merge_min_main_share,
        cluster_merge_relabel_threshold=q.cluster_merge_relabel_threshold,
        cluster_merge_main_threshold=q.cluster_merge_main_threshold,
        cluster_merge_dominant_share=q.cluster_merge_dominant_share,
        cluster_merge_dominant_threshold=q.cluster_merge_dominant_threshold,
        cluster_merge_dominant_minor_threshold=q.cluster_merge_dominant_minor_threshold,
        silence_align_enabled=q.silence_align_enabled,
        silence_align_tolerance_sec=q.silence_align_tolerance_sec,
        silence_align_min_segment_dur_sec=q.silence_align_min_segment_dur_sec,
        silence_vad_noise_db=q.silence_vad_noise_db,
        silence_vad_min_silence_sec=q.silence_vad_min_silence_sec,
        nospk_split_enabled=q.nospk_split_enabled,
        nospk_split_max_segment_sec=q.nospk_split_max_segment_sec,
        nospk_split_min_segment_sec=q.nospk_split_min_segment_sec,
        word_align_enabled=q.word_align_enabled,
        word_align_language=q.word_align_language,
        word_align_model_path=q.word_align_model_path,
        word_align_provider=q.word_align_provider,
        word_align_batch_size=q.word_align_batch_size,
        word_align_cuda_batch_size=q.word_align_cuda_batch_size,
        word_align_preflight_enabled=q.word_align_preflight_enabled,
        word_align_preflight_free_mib=q.word_align_preflight_free_mib,
        word_align_sidecar_enabled=q.word_align_sidecar_enabled,
    )
    # transcriber 需要 embedding_model 字段 (build_embedding_extractor_fn 鸭子类型读)
    _qwen3_singleton.embedding_model = q.embedding_model
    return _qwen3_singleton


def reset_qwen3_transcriber_singleton():
    """重置单例(仅测试用)"""
    global _qwen3_singleton
    _qwen3_singleton = None
