"""ASR wrapper — 复用 vendor qwen_asr_gguf 引擎做离线文件转录

数据流:
  audio_file (mp3/wav/...) -> qwen_asr_gguf.QwenASREngine.asr(audio)
    -> TranscribeResult(text, alignment.items=[ForcedAlignItem(text, start, end)])

本文件移植自 spikes/qwen3_diarize/src/asr.py, 改造点:
- 删除 sys.path hack, 直接 import src.core.vendor.qwen_asr_gguf 子包
- 模型路径必须显式传入, 不再硬编码 spike 目录
- 加 loguru 日志, 用同一引擎实例单例化(后续 Qwen3DiarizeTranscriber 维护)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import psutil
from loguru import logger

# 延迟到函数内 import 引擎类, 避免 module import 时就触发 dylib 加载 +
# llama.cpp Metal device 初始化(这些日志很吵, 实际只在第一次构建 engine 时需要)


@dataclass
class WordItem:
    """词级时间戳(production 不带 aligner, 实际仅 segment 级)"""
    text: str
    start: float
    end: float


@dataclass
class ASRChunkItem:
    """Qwen3-ASR 内部分片的粗粒度时间戳."""
    text: str
    start: float
    end: float
    index: int


@dataclass
class ASRResult:
    """ASR 输出"""
    text: str
    items: List[WordItem]   # 词级时间戳
    chunks: List[ASRChunkItem]  # 40s ASR chunk 级时间戳
    duration: float          # 音频时长 (sec)
    elapsed: float           # ASR 总耗时 (sec)
    rtf: float               # elapsed / duration
    peak_rss_mb: float       # 进程峰值 RSS (MB)
    rss_delta_mb: float      # ASR 调用引入的 RSS 增量 (MB)


_DEFAULT_BACKEND_ONNX_FN = "qwen3_asr_encoder_backend.onnx"
_DEFAULT_BACKEND_MLPACKAGE_FN = "qwen3_asr_encoder_backend.mlpackage"
_QWEN_CONTEXT_PREFIX = "You are a helpful assistant.\nKnown terms:\n"


def build_qwen_context(terms: List[str]) -> Optional[str]:
    """Build the fixed Qwen context; empty terms must remain ``None``."""
    if not terms:
        return None
    return _QWEN_CONTEXT_PREFIX + "\n".join(terms)


def build_engine_config(
    model_dir: str,
    encoder_frontend_fn: str = "qwen3_asr_encoder_frontend.onnx",
    encoder_backend_fn: Optional[str] = None,
    llm_fn: str = "qwen3_asr_llm.gguf",
    onnx_provider: Optional[str] = None,
    llm_use_gpu: bool = True,
    enable_aligner: bool = False,
):
    """构造 ASREngineConfig

    Args:
        model_dir: Qwen3-ASR 模型目录(含 encoder/decoder 文件).
        encoder_frontend_fn/backend_fn: 实际文件名(production 转换版去掉了默认的 .int4 后缀).
            backend_fn 默认按 onnx_provider 自动选:
              COREML_ANE_FULL → "qwen3_asr_encoder_backend.mlpackage"
              其他            → "qwen3_asr_encoder_backend.onnx"
            显式传 backend_fn 则不覆盖.
        llm_fn: GGUF 文件名.
        onnx_provider: ONNX EP. None 时按平台感知:
            - macOS (darwin): "COREML_ANE_FE" — frontend 走 ANE 验证有效
              (PoC N=2 wall -7.5%, 跟 num_threads=4 组合 -16.1%), backend 卡
              axis 4 op 兼容用 CPU. 详见 spikes/qwen3_mac_hw_accel/SUMMARY.md
            - 其他平台: "CPU"
            通过 FUNASR_QWEN3_ASR_ENCODER_PROVIDER=coreml_ane_full 可显式启用 Phase 3
            (frontend ANE + backend mlpackage ANE); 详见 spikes/.../phase3_backend/.
            显式传值则不覆盖.
        llm_use_gpu: llama.cpp 是否走 Metal(Apple Silicon 上必开).
        enable_aligner: production 不带 aligner, 默认 False.

    Returns:
        ASREngineConfig 实例.
    """
    from src.core.vendor.qwen_asr_gguf.inference.schema import ASREngineConfig
    import sys as _sys

    # 治理 D4: 从 config 一次性读 asr_encoder_provider / backend_mlpackage_units /
    # encoder_timing_enabled, 透传给 ASREngineConfig → QwenAudioEncoder, vendor 不再读 env
    try:
        from src.core import config as _config_module
        qwen3_cfg = _config_module.config.qwen3
        cfg_value = (getattr(qwen3_cfg, "asr_encoder_provider", None) or "auto").lower()
        backend_mlpackage_units = getattr(qwen3_cfg, "backend_mlpackage_units", "CPU_AND_NE")
        encoder_timing_enabled = getattr(qwen3_cfg, "encoder_timing_enabled", False)
    except Exception:
        cfg_value = "auto"
        backend_mlpackage_units = "CPU_AND_NE"
        encoder_timing_enabled = False

    if onnx_provider is None:
        # 优先级: 显式参数 > config (含 env FUNASR_QWEN3_ASR_ENCODER_PROVIDER) > 平台感知
        if cfg_value == "cpu":
            onnx_provider = "CPU"
        elif cfg_value == "cuda":
            onnx_provider = "CUDA"
        elif cfg_value in ("tensorrt", "trt"):
            onnx_provider = "TENSORRT"
        elif cfg_value == "coreml_ane_fe":
            onnx_provider = "COREML_ANE_FE"
        elif cfg_value == "coreml_ane_full":
            onnx_provider = "COREML_ANE_FULL"
        else:  # auto / 未知值
            onnx_provider = "COREML_ANE_FE" if _sys.platform == "darwin" else "CPU"

    # backend_fn 按 provider 自动选 (用户显式给则不动)
    if encoder_backend_fn is None:
        encoder_backend_fn = (
            _DEFAULT_BACKEND_MLPACKAGE_FN
            if onnx_provider == "COREML_ANE_FULL"
            else _DEFAULT_BACKEND_ONNX_FN
        )

    return ASREngineConfig(
        model_dir=model_dir,
        encoder_frontend_fn=encoder_frontend_fn,
        encoder_backend_fn=encoder_backend_fn,
        llm_fn=llm_fn,
        onnx_provider=onnx_provider,
        llm_use_gpu=llm_use_gpu,
        enable_aligner=enable_aligner,
        backend_mlpackage_units=backend_mlpackage_units,
        encoder_timing_enabled=encoder_timing_enabled,
    )


def build_engine(model_dir: str):
    """构造 QwenASREngine 实例(惰性 import, 首次实例化会加载 GGUF + Metal context)"""
    from src.core.vendor.qwen_asr_gguf.inference.asr import QwenASREngine

    cfg = build_engine_config(model_dir=model_dir)
    logger.info(f"加载 Qwen3-ASR 引擎: model_dir={model_dir}")
    engine = QwenASREngine(cfg)
    logger.success("Qwen3-ASR 引擎加载完成")
    return engine


def _load_audio_file(audio_file: str, start_second: Optional[float] = None, duration: Optional[float] = None):
    """Load audio through vendor helper.

    Kept as a tiny wrapper so unit tests can patch it without importing the
    vendor package, whose ``__init__`` eagerly loads llama.cpp dylibs.
    """
    from src.core.vendor.qwen_asr_gguf.inference.audio import load_audio

    if start_second is None and duration is None:
        return load_audio(audio_file)
    return load_audio(audio_file, start_second=start_second, duration=duration)


def _run_asr_loaded_audio(
    audio,
    engine,
    language: str = "Chinese",
    temperature: float = 0.4,
    label: str = "audio",
    context: Optional[str] = None,
) -> ASRResult:
    """Run ASR on an already-loaded 16kHz mono numpy array."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    duration = len(audio) / 16000.0

    t0 = time.time()
    cfg = getattr(engine, "config", None)
    result = engine.asr(
        audio=audio,
        context=context,
        language=language,
        chunk_size_sec=getattr(cfg, "chunk_size", 40.0),
        memory_chunks=getattr(cfg, "memory_num", 1),
        temperature=temperature,
    )
    elapsed = time.time() - t0

    rss_after = proc.memory_info().rss
    peak_rss = rss_after  # psutil 在 macOS 不支持 peak_rss, 用 after 作为代理

    items: List[WordItem] = []
    if result.alignment and result.alignment.items:
        for it in result.alignment.items:
            items.append(WordItem(text=it.text, start=it.start_time, end=it.end_time))

    chunks: List[ASRChunkItem] = []
    for chunk in getattr(result, "chunks", []) or []:
        chunks.append(
            ASRChunkItem(
                text=chunk.text,
                start=float(chunk.start_time),
                end=float(chunk.end_time),
                index=int(chunk.index),
            )
        )

    rtf = elapsed / duration if duration > 0 else 0.0
    logger.info(
        f"ASR 完成: label={label} duration={duration:.2f}s elapsed={elapsed:.2f}s "
        f"RTF={rtf:.3f} chars={len(result.text)}"
    )
    return ASRResult(
        text=result.text,
        items=items,
        chunks=chunks,
        duration=duration,
        elapsed=elapsed,
        rtf=rtf,
        peak_rss_mb=peak_rss / (1024 * 1024),
        rss_delta_mb=(rss_after - rss_before) / (1024 * 1024),
    )


def run_asr(
    audio_file: str,
    engine,
    language: str = "Chinese",
    temperature: float = 0.4,
    context: Optional[str] = None,
) -> ASRResult:
    """端到端跑一次 ASR, 返回带 RTF/内存数据的结果.

    Args:
        audio_file: 输入音频文件(任何 soundfile/ffmpeg 支持的格式).
        engine: 已构造的 QwenASREngine 实例(单例由调用方维护).
        language: 识别语言, 默认 "Chinese".
        temperature: decoder 采样温度.
        context: 已构造的 Qwen context, 空 terms 时为 None.

    Returns:
        ASRResult 含 text / segment-level items / RTF / 内存数据.

    实现选择:
      - 不调 engine.transcribe(audio_file, ...) — 它的 duration=0 默认值会让 load_audio 加载 0 秒
      - 自己 load_audio(audio_file) 拿 numpy → engine.asr(audio, ...)
      - 不启 aligner — production 没 aligner 模型, 只拿全文 + segment 级时间(40s chunk)
    """
    audio = _load_audio_file(audio_file)
    return _run_asr_loaded_audio(
        audio=audio,
        engine=engine,
        language=language,
        temperature=temperature,
        label=audio_file,
        context=context,
    )


def run_asr_window(
    audio_file: str,
    engine,
    start_second: float,
    duration: float,
    language: str = "Chinese",
    temperature: float = 0.4,
    context: Optional[str] = None,
) -> ASRResult:
    """Run ASR on a bounded audio window while reusing the same engine.

    This is the PoC entry point for long-audio macro segments.  It avoids the
    vendor ``transcribe(duration=0.0)`` footgun and keeps the wrapper's
    ``ASRResult`` metrics shape.
    """
    if duration <= 0:
        raise ValueError("duration must be > 0 for run_asr_window")

    audio = _load_audio_file(audio_file, start_second=start_second, duration=duration)
    return _run_asr_loaded_audio(
        audio=audio,
        engine=engine,
        language=language,
        temperature=temperature,
        label=f"{audio_file}@{start_second:.1f}+{duration:.1f}",
        context=context,
    )
