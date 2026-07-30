"""
配置管理模块 - 支持环境变量覆盖和配置验证
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Literal
from pydantic import BaseModel, Field, validator, model_validator
from loguru import logger
from dotenv import load_dotenv


# ==================== FUNASR_PROFILE 套餐 (A1 治理) ====================
# 切平台 / 切环境一句 env 搞定: FUNASR_PROFILE=mac_prod / mac_dev / cuda_prod / cuda_dev
# 优先级: defaults < config.json < profile < env (env 仍可覆盖 profile)
# profile 覆盖 config.json 已有字段, 启动日志会列出被覆盖的字段防止"惊讶感"
# pool_size 全 profile 默认 1 (2026-06-10 用户拍板):
# - 3060 12GB 实测 pool=2 + word_align 双 MMS CUDA session 撞 BFCArena OOM
#   (fallback 虽不挂但词级时间戳静默丢失), 单实例稳定可预期
# - 并发需求再用 FUNASR_QWEN3_POOL_SIZE env 按机器显存/内存显式开
PROFILES: Dict[str, Dict[str, Any]] = {
    # Mac 主力引擎 = funasr: 速度快, 大内存(如 64G)并发拉得开(用 FUNASR_MAX_CONCURRENT_TASKS
    # 按内存调, 实测可 3 进程). qwen3-1.7B 准确度更高但 Mac 上提速需更强环境, 仅按需用
    # FUNASR_DEFAULT_ENGINE=qwen3 临时切. qwen3 配置(pool/encoder)保留以便临时切换即用.
    # CUDA profile 才默认 qwen3(高准确度 + GPU 算力补速度).
    "mac_prod": {
        "server": {"port": 8767},
        "transcription": {"default_engine": "funasr", "qwen3_pool_size": 1},
        "qwen3": {"asr_encoder_provider": "coreml_ane_full"},
    },
    "mac_dev": {
        "server": {"port": 8867},
        "transcription": {"default_engine": "funasr", "qwen3_pool_size": 1},
        "qwen3": {"asr_encoder_provider": "coreml_ane_full"},
        "logging": {"level": "DEBUG"},
    },
    # word_align (词级时间戳) 改 per-request API 开关 (2026-06-16 显存落地评审), profile
    # 不再强开. 原因: CUDA word_align session 显存高水位常驻 (batch>=2 在 3060 12GB 撞
    # BFCArena OOM, 见 docs/开发/gpu加速/2026-06-16-Qwen3-word-align显存PoC与落地计划.md),
    # 全局强开让每个请求都被迫吃 ~6GB 显存. 现默认关 (config 兜底显式 False everywhere,
    # codex #13: 否则 per-request 默认不是真 OFF), 要词的请求才按 request word_align=true 开,
    # CUDA 锁死 batch=1. 想全局默认开仍可走 env FUNASR_QWEN3_WORD_ALIGN_ENABLED=true (兜底层).
    "cuda_prod": {
        "transcription": {"default_engine": "qwen3", "qwen3_pool_size": 1},
        "qwen3": {"asr_encoder_provider": "cuda"},
    },
    "cuda_dev": {
        "server": {"port": 8867},
        "transcription": {"default_engine": "qwen3", "qwen3_pool_size": 1},
        "qwen3": {"asr_encoder_provider": "cuda"},
        "logging": {"level": "DEBUG"},
    },
}


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8767
    max_connections: int = 100
    max_file_size_mb: int = 5000
    allowed_extensions: list[str] = [".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".webm"]
    temp_dir: str = "./temp"
    upload_dir: str = "./uploads"
    connection_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 60

    model_config = {"protected_namespaces": ()}


class FunASRConfig(BaseModel):
    """FunASR模型配置"""
    model: str = "paraformer-zh"
    model_revision: str = "v2.0.4"
    vad_model: str = "fsmn-vad"
    vad_model_revision: str = "v2.0.4"
    punc_model: str = "ct-punc-c"
    punc_model_revision: str = "v2.0.4"
    spk_model: str = "cam++"
    spk_model_revision: str = "v2.0.2"
    model_dir: str = "./models"
    batch_size_s: int = 500
    ncpu: int = 16
    device: str = "auto"
    device_priority: List[str] = ["mps", "cpu"]
    fallback_on_error: bool = True
    disable_update: bool = True
    disable_pbar: bool = True

    model_config = {"protected_namespaces": ()}


class Qwen3Config(BaseModel):
    """Qwen3-Diarize 引擎配置(C3 落地)

    服务器启动时,如果 transcription.default_engine == "qwen3", 这些字段会被
    Qwen3DiarizeTranscriber 读取并构造引擎.

    模型路径默认指向 scripts/download_qwen3_models.sh 落地的位置.
    """
    # 模型路径
    asr_model_dir: str = "./models/qwen3_diarize/Qwen3-ASR-1.7B"
    segmentation_model: str = "./models/qwen3_diarize/sherpa/pyannote-segmentation-3.0/model.onnx"
    embedding_model: str = "./models/qwen3_diarize/sherpa/nemo-titanet-small/embedding.onnx"

    # Diarize preset (PoC 报告 preset=auto 普适最优)
    num_speakers: Optional[int] = None    # None = 自适应聚类
    cluster_threshold: float = 0.9
    # A2 治理 (D1): num_threads / provider 默认 "auto" → Pydantic model_validator 在 load 时
    # 一次性解析为具体 int / "cpu", 下游 (transcriber:58/237/399/575, inproc_pool:117) 永远拿到 int.
    # auto 解析规则:
    #   num_threads="auto" → detect_runtime().recommend_num_threads()
    #     · MacRuntime  → 4 (PoC: t=4 比 t=8 wall -11.5%, 10 cores M1 Max + pool=2 最优)
    #     · CudaRuntime → 2/4 按 vCPU 数 (≤4 vCPU=2, ≥5=4, 详见 runtime.py)
    #     · CpuRuntime  → 同 cuda 按 vCPU
    #   provider="auto" → "cpu" (sherpa SpeakerEmbeddingExtractor 在所有 runtime 都 cpu;
    #                            cuda 上 sherpa diarize 已被 ort_cuda 替代, embedding 只在
    #                            cluster_centroid_merge 用一次)
    # 显式 int / 显式 provider 字符串不被覆盖.
    # 详见 spikes/qwen3_mac_hw_accel/num_threads_tuning.md + docs/开发/config-治理-新session-prompt.md
    num_threads: int | str = "auto"
    provider: str = "auto"

    # Phase 2/3 escape hatch: 生产环境如果 ANE 出问题, 改 env 一键回退 CPU
    # auto: build_engine_config 按平台感知 (macOS → COREML_ANE_FE, 其他 → CPU)
    # cpu: 强制 CPU (关掉 ANE)
    # coreml_ane_fe: Phase 2 — frontend ANE + backend ONNX CPU (生产稳定路径, 默认)
    # coreml_ane_full: Phase 3 — frontend ANE + backend mlpackage ANE (需 .mlpackage)
    #                  ⚠️ macOS 26 Beta + diarize pipeline 端到端 SIGSEGV 已知问题
    #                  (CoreML MLE5ExecutionStream lingering reset 与 sherpa-onnx
    #                   libdispatch worker thread 抢 Python GIL race);
    #                  单 audio ASR (不带 diarize) parity 通过 cos=0.9891
    #                  Phase 3 路径暂作 dev 实验, 不切生产默认
    # 可通过 FUNASR_QWEN3_ASR_ENCODER_PROVIDER 环境变量覆盖
    asr_encoder_provider: str = "auto"

    # ASR 参数
    language: str = "Chinese"
    temperature: float = 0.4

    # PR2: short-segment guard 后处理
    # 默认开启 (PoC v12 数据显示对所有 baseline 都正向, 见 docs/开发/PR4-149min校对稿对比分析.md)
    short_segment_guard_enabled: bool = True
    short_segment_drop_sec: float = 1.5
    short_segment_aba_max_mid_sec: float = 1.5
    short_segment_merge_same: bool = True

    # PR3: cluster centroid merge (多人场景修复, 加载 audio + sherpa embedding extractor 算 centroid)
    # 默认开启 (PoC eval_set 5 个样本 1/2/3+/4/6 speaker 全过, 见 docs/开发/PR4-*)
    cluster_merge_enabled: bool = True
    cluster_merge_min_main_share: float = 0.03
    cluster_merge_relabel_threshold: float = 0.55
    cluster_merge_main_threshold: float = 0.78
    cluster_merge_dominant_share: float = 0.6
    cluster_merge_dominant_threshold: float = 0.6
    # dominant 模式吃相似 minor cluster 的阈值 (修 over-detect 兜底, 见
    # docs/开发/archive/spk-over-detect-归因调研结果.md). 默认 0.5, 比 relabel(0.55) 宽松.
    cluster_merge_dominant_minor_threshold: float = 0.5

    # silence-aware 段切点对齐 (携进自 spike 405abf6, 见 spikes/qwen3_silence_align/SUMMARY.md)
    # 在 merge_asr_chunks_and_diarize 输出后做 snap-to-silence 后处理:
    #   60s podcast: align_ratio 54.55% → 73.68% (+19pp)
    #   60min long: 27.41% → 60.73% (+33pp)
    # tolerance=2.0 是用户拍板的平衡值 (sweep 数据见 SUMMARY)
    # min_segment_dur=0.1 保护吸附后段不退化成 0 时长
    # VAD 默认 -25dB/0.2s (生产 FunASR 默认 -35dB/0.8s 在 podcast 完全失效, 必须改)
    silence_align_enabled: bool = True
    silence_align_tolerance_sec: float = 2.0
    silence_align_min_segment_dur_sec: float = 0.1
    silence_vad_noise_db: str = "-25dB"
    silence_vad_min_silence_sec: float = 0.20

    # nospk 分层切段 (diarize 开关 D5+T1): diarize=false 时段 = ASR ~40s chunk,
    # 对超长段做两层 fallback 切分 (有 words 词隙切 / 无 words 静音切 + char-ratio
    # 文本归属), 见 src/core/qwen3/segment_split.py.
    #   - max_segment_sec: 超过此值触发切分 (也是切出片的近似上界, 字幕可用粒度)
    #   - min_segment_sec: 切出片的最小时长 (吸收 short_segment_guard 的通用清理职责)
    nospk_split_enabled: bool = True
    nospk_split_max_segment_sec: float = 12.0
    nospk_split_min_segment_sec: float = 1.5

    # 词级时间戳 (word_align, MMS-300M CTC-FA): 增量挂 segment.words, 不替换段边界.
    # 见 docs/开发/2026-06-09-qwen3-词级时间戳-PoC计划.md.
    #   - word_align_enabled: 字段默认关; cuda_prod/cuda_dev profile 默认开 (CUDA 仅 +1% RTF),
    #     Mac profile 保持关 (CPU +17% RTF). 开启会加载 ~1.2GB MMS ONNX + 改 cache key.
    #   - word_align_language: ISO 码 (chi/eng/jpn/kor...) 兜底语言, per-request
    #     language 字段优先. 中英混排用 chi (preprocess_text 对 chi 逐字切, 能吃英文).
    #   - word_align_model_path: 本地预下的 MMS ONNX 路径 (download_qwen3_models.sh
    #     从 ~/ctc_forced_aligner/model.onnx 拷过来, 不走 deskpai 运行时下载).
    #   - word_align_provider: "auto" → runtime.recommend_word_align_provider()
    #     (Mac/Cpu → CPUExecutionProvider, Cuda → CUDAExecutionProvider). 显式 EP
    #     字符串不被覆盖. 解析在 word_align wrapper 做 (不在 Pydantic, 区别于 sherpa provider).
    #   - word_align_batch_size: generate_emissions ONNX 推理 batch (CPU/默认).
    #   - word_align_cuda_batch_size: CUDA EP 专用 batch. 显存落地 (2026-06-16):
    #     CUDA batch>=2 在 3060 12GB 撞 BFCArena OOM, 锁死 1. WordAligner 在 provider
    #     解析成 CUDA 时取此值, 否则取 word_align_batch_size. CPU fallback 仍用 16.
    word_align_enabled: bool = False
    word_align_language: str = "chi"
    word_align_model_path: str = "./models/qwen3_diarize/ctc_forced_aligner/model.onnx"
    word_align_provider: str = "auto"
    word_align_batch_size: int = 16
    word_align_cuda_batch_size: int = 1

    # VRAM preflight (TODOS #17, 评审定案 P1): 加载 CUDA word_align session 前探显存,
    # 不够直接走 CPU (事前预防, 不等 OOM). 探针 src/core/gpu_mem.py: free_vram_mib().
    #   - word_align_preflight_enabled: 关掉则回到「撞了 OOM 才 fallback」的旧行为.
    #   - word_align_preflight_free_mib: 加载 CUDA session 要求的最小空闲显存 (MiB).
    #     保守起点 4608 (~4.5GB: session ~3.4GB + 余量), 3060 真机标定后改准.
    #     换卡 / 调 pool_size 用 FUNASR_QWEN3_WORD_ALIGN_PREFLIGHT_FREE_MIB env 覆盖.
    #   ⚠️ preflight 不替代 OOM fallback (codex #11, TOCTOU): 探完到建 session 之间
    #     同卡其它进程仍可能吃掉显存, poison + CPU 兜底永远保留.
    word_align_preflight_enabled: bool = True
    word_align_preflight_free_mib: int = 4608

    # word_align CUDA sidecar (TODOS #18, 评审定案 A1/A3): CUDA word_align 拆独立
    # 长驻进程, 空闲 idle TTL 自杀**真正释放 VRAM** (ORT BFCArena 唯进程退出可还).
    # **仅 cuda runtime 生效** (runtime gate 另控; Mac worker 即退 / CPU 不碰显存).
    #   - word_align_sidecar_enabled: 关掉则 CUDA word_align 退回进程内加载 (Lane 1 行为).
    #   - word_align_sidecar_idle_ttl_sec: 空闲多久无请求 → sidecar 退出释放显存.
    #   - word_align_sidecar_align_timeout_sec: 单次对齐超时, 超时主进程杀 sidecar 再 CPU
    #     (codex #4 杜绝 CUDA+CPU 双跑). 长音频 word_align CUDA 仅 +1% RTF, 180s 够宽.
    word_align_sidecar_enabled: bool = True
    word_align_sidecar_idle_ttl_sec: float = 90.0
    word_align_sidecar_align_timeout_sec: float = 180.0

    # A2 治理 (D4): vendor encoder.py 原本直接读 os.environ.get(...) 这两个 env, 提升进 Pydantic
    # backend_mlpackage_units: COREML_ANE_FULL 时 backend mlpackage 跑在哪个芯片
    #   - CPU_AND_NE: 默认, frontend ANE + backend mlpackage ANE
    #     ⚠️ N=2 并发可能跟 frontend ANE 4 路冲突触发 llama.cpp ggml_abort, 用 CPU_AND_GPU 错开
    #   - CPU_AND_GPU: frontend 独占 ANE, backend 走 Metal GPU
    #   - ALL: 三路全开 (mac 26 Beta 有 issue, 仅 dev 实验)
    # encoder_timing_enabled: True 时 encoder 每段打印 mel/frontend/backend 耗时 (排查性能用)
    backend_mlpackage_units: Literal["CPU_AND_NE", "CPU_AND_GPU", "ALL"] = "CPU_AND_NE"
    encoder_timing_enabled: bool = False

    model_config = {"protected_namespaces": ()}

    @model_validator(mode="after")
    def _resolve_auto_sentinels(self):
        """A2 治理 (D1): num_threads/provider "auto" 在 model load 时一次性解析.

        解析后字段类型保证是 int / 具体字符串, 下游消费点 (5 处) 零改动.
        数字字符串 (env override "4" 走 _override_if_set int converter 已是 int, 这里兜底).
        """
        if isinstance(self.num_threads, str):
            v = self.num_threads.strip().lower()
            if v == "auto":
                from src.core.runtime import detect_runtime
                self.num_threads = detect_runtime().recommend_num_threads()
            else:
                self.num_threads = int(v)
        if isinstance(self.provider, str) and self.provider.strip().lower() == "auto":
            # sherpa SpeakerEmbeddingExtractor 在 cuda runtime 也走 cpu
            # (sherpa diarize 已被 ort_cuda 接管, embedding 只在 cluster_merge 用一次)
            self.provider = "cpu"
        return self


class TranscriptionConfig(BaseModel):
    """转录配置"""
    max_concurrent_tasks: int = 2
    max_queue_size: int = 50
    concurrency_mode: str = "pool"
    task_timeout_minutes: int = 30
    retry_times: int = 2
    cache_enabled: bool = True
    delete_after_transcription: bool = True
    transcription_speed_ratio: int = 10
    queue_status_enabled: bool = True
    # PR1: ASR 引擎默认选择。upload request 未指定 engine 时走这个。
    # 可通过 FUNASR_DEFAULT_ENGINE 环境变量覆盖。
    default_engine: str = "funasr"

    # PR2: 跨引擎缓存共享开关(默认 True)
    # True: get_cached_result 先按 (file_hash, engine) 精确查, miss 后回退按 file_hash 查任意 engine 最新行
    # False: strict 模式, 仅按 (file_hash, engine) 查
    # 跨引擎 SRT 命中时从 TranscriptionResult.segments 重建(避开 raw_result 引擎特定结构)
    # 可通过 FUNASR_CACHE_CROSS_ENGINE 环境变量覆盖
    cache_cross_engine: bool = True

    # PR3: Qwen3 引擎的 worker pool 大小(独立于 max_concurrent_tasks)
    # FunASR pool 用 max_concurrent_tasks(物理最优 2), Qwen3 pool 用 qwen3_pool_size(PoC v5 sweet spot 3)
    # 两者独立, 切引擎不需要改 env
    # 可通过 FUNASR_QWEN3_POOL_SIZE 环境变量覆盖
    qwen3_pool_size: int = 3

    # ===== 高负载队列健壮性（2026-06-16 止血，见 docs/开发/2026-06-16-高负载队列机制-止血修复计划.md）=====
    # self.tasks 内存清理：终态任务（completed/failed/cancelled/timed_out）保留 TTL。
    # 必须 ≥ 客户端轮询窗口，否则慢客户端轮询会 miss（返回 expired）。
    # 可通过 FUNASR_TASK_RETENTION_TTL_SECONDS 覆盖。
    task_retention_ttl_seconds: int = 3600
    # self.tasks 硬数量上限（size-cap 兜底）：超限从最老终态任务挤，防突发期 TTL 未到先撑爆内存。
    # 可通过 FUNASR_TASK_MAX_RETAINED 覆盖。
    task_max_retained: int = 500
    # 内存清理 + 看门狗的周期扫描间隔（秒）。
    task_cleanup_interval_seconds: int = 60
    # processing 看门狗：任务超此时长仍 PROCESSING → 强制 TIMED_OUT（明显卡死阈值，
    # 防 ASR 卡死/worker 挂导致永不终态 → 永不被清 + 永占并发名额）。
    # 可通过 FUNASR_TASK_MAX_PROCESSING_SECONDS 覆盖。
    task_max_processing_seconds: int = 3600
    # 分片上传 session：遗弃 session 的 TTL（秒），sweeper 清其 temp 文件防磁盘泄漏。
    upload_session_ttl_seconds: int = 1800
    # 并发 session 硬上限：超限拒新 session（防恶意/坏客户端堆满磁盘）。
    upload_session_max_count: int = 200
    # 孤儿上传文件 sweeper 宽限期（秒）：upload_dir 里 mtime 超此值 且 无 live 任务/session
    # 引用的文件才删。必须 >> task_max_processing_seconds(确保 worker 早读完, 避 unlink-under-worker)。
    # 默认 7200(2h)。delete_after_transcription off 时 sweeper 整体不跑。
    orphan_file_grace_seconds: int = 7200

    # ===== FunASR segment 合并视图（serve 投影层，issue #1）=====
    # 缓存存句级真值；JSON 出口对 funasr 结果做「同说话人相邻句合并」视图。
    # gap: 相邻句间隔 < 此值才可并（秒）。原 funasr_transcriber 硬编码 3.0 收编到此。
    # max_span: 合并后累计跨度上限（秒）；超限在句边界断开。<=0 关闭上限（旧无 cap 行为）。
    # env: FUNASR_SEGMENT_MERGE_GAP_SEC / FUNASR_SEGMENT_MERGE_MAX_SPAN_SEC
    segment_merge_gap_sec: float = 3.0
    segment_merge_max_span_sec: float = 120.0

    model_config = {"protected_namespaces": ()}


class DatabaseConfig(BaseModel):
    """数据库配置"""
    path: str = "./data/transcription_cache.db"
    max_cache_days: int = 30
    # 缓存数量软上限(size-cap):clean_old_cache 在 TTL 清理后, 若 COUNT > 此值,
    # 按 accessed_at LRU 挤到上限。防 30 天内高频转录无限堆积。<=0 关闭(防误配清空)。
    # 软上限:每小时随 _periodic_cleanup 跑, 非实时硬 cap。
    max_cache_count: int = 5000

    model_config = {"protected_namespaces": ()}


class NotificationConfig(BaseModel):
    """通知配置"""
    enabled: bool = True
    webhook_url: str = ""
    retry_times: int = 3
    timeout_seconds: int = 10

    model_config = {"protected_namespaces": ()}


class AuthConfig(BaseModel):
    """认证配置"""
    enabled: bool = False
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    model_config = {"protected_namespaces": ()}


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    rotation: str = "100 MB"
    retention: str = "7 days"
    log_dir: str = "./logs"

    model_config = {"protected_namespaces": ()}


class ObservabilityConfig(BaseModel):
    """可观测性仪表盘配置 (P1, 设计定案 2026-06-16).

    服务通过 websockets 同端口 process_request 钩子暴露 /health + /metrics:
    - /health: liveness (维护循环活 + 能接任务), JSON, 裸放 (只是死活无敏感信息)
    - /metrics: Prometheus 文本格式, 只出聚合计数

    A5 安全 (codex #16): server.host 默认 0.0.0.0 (不是 LAN 限定). 故
    metrics_token 未设 + 0.0.0.0 时 /metrics 拒绝服务 (在端点层判定), 强制
    要么设 token 要么显式绑 LAN/loopback. /metrics 响应绝不回显 token / secret.
    """
    # /health + /metrics 端点总开关. 关掉则 process_request 不拦这些路径 (退回纯 ws).
    metrics_enabled: bool = True
    # /metrics 访问 token (config 字段, env 覆盖). 设了则校验 Authorization header
    # 或 ?token= query; 未设 + host=0.0.0.0 → /metrics 拒绝 (A5). None = 未设.
    metrics_token: Optional[str] = None

    # ---- 告警 (A: scripts/metrics_alert.py 用, 与 server 进程解耦) ----
    # 告警总开关. server 进程本身不读这些字段, 仅外部告警脚本读 (放这里复用
    # config 体系 + env 覆盖 + metrics_token 共享, 单一 source of truth)。默认关。
    alert_enabled: bool = False
    # 队列饱和告警阈值: queue_size/queue_max >= 该比例则告警 (默认 0.8)。
    alert_queue_saturation_ratio: float = 0.8
    # 错误激增告警阈值: errors_total 单轮增量 >= 该值则告警 (默认 5)。
    alert_error_surge_threshold: int = 5
    # 同一状态型告警的去抖窗口 (秒): 该窗口内不重复通知 (默认 900=15min)。
    alert_cooldown_seconds: int = 900

    model_config = {"protected_namespaces": ()}


class Config(BaseModel):
    """总配置 - 支持环境变量覆盖"""
    server: ServerConfig = ServerConfig()
    funasr: FunASRConfig = FunASRConfig()
    qwen3: Qwen3Config = Qwen3Config()
    transcription: TranscriptionConfig = TranscriptionConfig()
    database: DatabaseConfig = DatabaseConfig()
    notification: NotificationConfig = NotificationConfig()
    auth: AuthConfig = AuthConfig()
    logging: LoggingConfig = LoggingConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> "Config":
        """
        从文件加载配置，并支持环境变量覆盖
        优先级: 环境变量 > config.json > 默认值
        """
        # 加载 .env 文件
        load_dotenv()

        # 从 config.json 加载基础配置
        config_data = cls._load_json_config(config_path)

        # 应用 FUNASR_PROFILE 套餐 (覆盖 config.json, env 仍可覆盖 profile)
        config_data = cls._apply_profile_defaults(config_data)

        # 应用环境变量覆盖
        config_data = cls._apply_env_overrides(config_data)

        # 创建配置实例
        config = cls(**config_data)

        # 验证配置
        cls._validate_config(config)

        # 注意: 配置打印延迟到 setup_logger() 之后进行，以保证日志格式统一
        # 不在这里调用 cls._print_config(config)

        return config

    @classmethod
    def _load_json_config(cls, config_path: str) -> Dict[str, Any]:
        """从 JSON 文件加载配置"""
        if not os.path.exists(config_path):
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 过滤掉注释字段
            return cls._filter_comments(data)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.warning("使用默认配置")
            return {}

    @classmethod
    def _filter_comments(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """递归过滤掉所有以 _comment 开头的键"""
        filtered = {}
        for key, value in data.items():
            if not key.startswith("_comment"):
                if isinstance(value, dict):
                    filtered[key] = cls._filter_comments(value)
                else:
                    filtered[key] = value
        return filtered

    @classmethod
    def _validate_engine_runtime(cls, config: "Config", errors: List[str]) -> None:
        """A3 治理 (D3): 启动时校验 engine ↔ runtime 兼容性.

        触发条件: default_engine=qwen3 且 runtime.validate() 抛 (典型: cuda runtime + ORT EP 缺).
        Mac/Cpu runtime 的 validate() 是 no-op, 自然不挂.

        per-request engine ≠ default_engine 已被 transcriber_dispatch.py:57 拒,
        所以这里只查 default_engine, 不需要扫所有可能 engine.
        """
        engine = config.transcription.default_engine
        if engine != "qwen3":
            return
        try:
            from src.core.runtime import detect_runtime
        except Exception as exc:
            errors.append(f"无法导入 runtime 模块校验 engine={engine}: {exc}")
            return
        runtime = detect_runtime()
        try:
            runtime.validate()
        except RuntimeError as exc:
            errors.append(
                f"default_engine={engine} + runtime={runtime.name} 启动校验失败: {exc}. "
                f"用 FUNASR_RUNTIME=cpu 降级或修依赖 (onnxruntime-gpu / LD_LIBRARY_PATH)."
            )

    @classmethod
    def _apply_profile_defaults(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """应用 FUNASR_PROFILE 套餐 — 在 env override 之前 apply, profile 覆盖 config.json.

        优先级链: defaults < config.json < profile < env
        - 未设 FUNASR_PROFILE: no-op, 直接返回 config_data
        - 未知 profile name: warn + ignore, 不挂
        - 已知 profile: deep-merge 进 config_data, profile 字段覆盖 config.json 已有字段

        启动日志列出被覆盖的字段, 防止"我明明 config.json 写了 X profile 怎么变 Y"的惊讶感.
        """
        profile_name = os.getenv("FUNASR_PROFILE", "").strip().lower()
        if not profile_name:
            return config_data

        if profile_name not in PROFILES:
            logger.warning(
                f"⚠️  FUNASR_PROFILE={profile_name!r} 未知, "
                f"可选: {list(PROFILES.keys())}. 忽略 profile, 走 config.json + env."
            )
            return config_data

        profile_data = PROFILES[profile_name]
        overridden = cls._deep_merge_profile(config_data, profile_data)
        logger.info(
            f"FUNASR_PROFILE={profile_name} applied. "
            f"覆盖字段 ({len(overridden)}): "
            + ", ".join(f"{path}({old!r}→{new!r})" for path, old, new in overridden)
        )
        return config_data

    @classmethod
    def _deep_merge_profile(
        cls,
        base: Dict[str, Any],
        override: Dict[str, Any],
        path: str = "",
    ) -> List[Tuple[str, Any, Any]]:
        """递归 deep-merge: override 覆盖 base. 返回 [(path, old_value, new_value), ...] 用于日志."""
        records: List[Tuple[str, Any, Any]] = []
        for k, v in override.items():
            full_path = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                base.setdefault(k, {})
                if not isinstance(base[k], dict):
                    # base 该 key 不是 dict, 直接覆盖整段
                    records.append((full_path, base[k], v))
                    base[k] = v
                else:
                    records.extend(cls._deep_merge_profile(base[k], v, full_path))
            else:
                old = base.get(k, "(默认)")
                if old != v:
                    records.append((full_path, old, v))
                base[k] = v
        return records

    @classmethod
    def _apply_env_overrides(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""

        # 初始化嵌套字典
        if "server" not in config_data:
            config_data["server"] = {}
        if "funasr" not in config_data:
            config_data["funasr"] = {}
        if "transcription" not in config_data:
            config_data["transcription"] = {}
        if "database" not in config_data:
            config_data["database"] = {}
        if "notification" not in config_data:
            config_data["notification"] = {}
        if "auth" not in config_data:
            config_data["auth"] = {}
        if "logging" not in config_data:
            config_data["logging"] = {}
        if "qwen3" not in config_data:
            config_data["qwen3"] = {}
        if "observability" not in config_data:
            config_data["observability"] = {}

        # ==================== 服务器配置 ====================
        cls._override_if_set(config_data["server"], "host", "FUNASR_SERVER_HOST")
        cls._override_if_set(config_data["server"], "port", "FUNASR_SERVER_PORT", int)
        cls._override_if_set(config_data["server"], "max_connections", "FUNASR_SERVER_MAX_CONNECTIONS", int)
        cls._override_if_set(config_data["server"], "max_file_size_mb", "FUNASR_SERVER_MAX_FILE_SIZE_MB", int)
        cls._override_if_set(config_data["server"], "temp_dir", "FUNASR_TEMP_DIR")
        cls._override_if_set(config_data["server"], "upload_dir", "FUNASR_UPLOAD_DIR")

        # ==================== FunASR 配置 ====================
        cls._override_if_set(config_data["funasr"], "model_dir", "FUNASR_MODEL_DIR")
        cls._override_if_set(config_data["funasr"], "device", "FUNASR_DEVICE")
        cls._override_if_set(config_data["funasr"], "ncpu", "FUNASR_NCPU", int)
        cls._override_if_set(config_data["funasr"], "batch_size_s", "FUNASR_BATCH_SIZE_S", int)

        # 设备优先级 (逗号分隔)
        device_priority = os.getenv("FUNASR_DEVICE_PRIORITY")
        if device_priority:
            config_data["funasr"]["device_priority"] = [d.strip() for d in device_priority.split(",")]

        # ==================== 转录配置 ====================
        cls._override_if_set(config_data["transcription"], "max_concurrent_tasks", "FUNASR_MAX_CONCURRENT_TASKS", int)
        cls._override_if_set(config_data["transcription"], "task_timeout_minutes", "FUNASR_TASK_TIMEOUT_MINUTES", int)
        cls._override_if_set(config_data["transcription"], "transcription_speed_ratio", "FUNASR_TRANSCRIPTION_SPEED_RATIO", int)
        cls._override_if_set(config_data["transcription"], "default_engine", "FUNASR_DEFAULT_ENGINE")
        cls._override_if_set(config_data["transcription"], "cache_cross_engine", "FUNASR_CACHE_CROSS_ENGINE", cls._parse_bool)
        cls._override_if_set(config_data["transcription"], "qwen3_pool_size", "FUNASR_QWEN3_POOL_SIZE", int)
        # 高负载队列健壮性 env override（2026-06-16 止血）
        cls._override_if_set(config_data["transcription"], "max_queue_size", "FUNASR_MAX_QUEUE_SIZE", int)
        cls._override_if_set(config_data["transcription"], "task_retention_ttl_seconds", "FUNASR_TASK_RETENTION_TTL_SECONDS", int)
        cls._override_if_set(config_data["transcription"], "task_max_retained", "FUNASR_TASK_MAX_RETAINED", int)
        cls._override_if_set(config_data["transcription"], "task_cleanup_interval_seconds", "FUNASR_TASK_CLEANUP_INTERVAL_SECONDS", int)
        cls._override_if_set(config_data["transcription"], "task_max_processing_seconds", "FUNASR_TASK_MAX_PROCESSING_SECONDS", int)
        cls._override_if_set(config_data["transcription"], "upload_session_ttl_seconds", "FUNASR_UPLOAD_SESSION_TTL_SECONDS", int)
        cls._override_if_set(config_data["transcription"], "upload_session_max_count", "FUNASR_UPLOAD_SESSION_MAX_COUNT", int)
        cls._override_if_set(config_data["transcription"], "orphan_file_grace_seconds", "FUNASR_ORPHAN_FILE_GRACE_SECONDS", int)
        # FunASR segment 合并视图（serve 投影层）
        cls._override_if_set(config_data["transcription"], "segment_merge_gap_sec", "FUNASR_SEGMENT_MERGE_GAP_SEC", float)
        cls._override_if_set(config_data["transcription"], "segment_merge_max_span_sec", "FUNASR_SEGMENT_MERGE_MAX_SPAN_SEC", float)

        # ==================== Database 配置 ====================
        cls._override_if_set(config_data["database"], "max_cache_days", "FUNASR_MAX_CACHE_DAYS", int)
        cls._override_if_set(config_data["database"], "max_cache_count", "FUNASR_MAX_CACHE_COUNT", int)

        # ==================== Qwen3-Diarize 配置 ====================
        cls._override_if_set(config_data["qwen3"], "asr_model_dir", "FUNASR_QWEN3_ASR_MODEL_DIR")
        cls._override_if_set(config_data["qwen3"], "segmentation_model", "FUNASR_QWEN3_SEGMENTATION_MODEL")
        cls._override_if_set(config_data["qwen3"], "embedding_model", "FUNASR_QWEN3_EMBEDDING_MODEL")
        cls._override_if_set(config_data["qwen3"], "cluster_threshold", "FUNASR_QWEN3_CLUSTER_THRESHOLD", float)
        cls._override_if_set(config_data["qwen3"], "num_threads", "FUNASR_QWEN3_NUM_THREADS", int)
        cls._override_if_set(config_data["qwen3"], "provider", "FUNASR_QWEN3_PROVIDER")
        cls._override_if_set(config_data["qwen3"], "asr_encoder_provider", "FUNASR_QWEN3_ASR_ENCODER_PROVIDER")
        cls._override_if_set(config_data["qwen3"], "language", "FUNASR_QWEN3_LANGUAGE")
        cls._override_if_set(config_data["qwen3"], "temperature", "FUNASR_QWEN3_TEMPERATURE", float)
        # PR2 short-segment guard
        cls._override_if_set(config_data["qwen3"], "short_segment_guard_enabled", "FUNASR_QWEN3_SHORT_GUARD_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["qwen3"], "short_segment_drop_sec", "FUNASR_QWEN3_SHORT_DROP_SEC", float)
        cls._override_if_set(config_data["qwen3"], "short_segment_aba_max_mid_sec", "FUNASR_QWEN3_SHORT_ABA_MAX_MID_SEC", float)
        cls._override_if_set(config_data["qwen3"], "short_segment_merge_same", "FUNASR_QWEN3_SHORT_MERGE_SAME", cls._parse_bool)
        # PR3 cluster centroid merge
        cls._override_if_set(config_data["qwen3"], "cluster_merge_enabled", "FUNASR_QWEN3_CLUSTER_MERGE_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["qwen3"], "cluster_merge_min_main_share", "FUNASR_QWEN3_CLUSTER_MERGE_MIN_MAIN_SHARE", float)
        cls._override_if_set(config_data["qwen3"], "cluster_merge_relabel_threshold", "FUNASR_QWEN3_CLUSTER_MERGE_RELABEL_THRESHOLD", float)
        cls._override_if_set(config_data["qwen3"], "cluster_merge_main_threshold", "FUNASR_QWEN3_CLUSTER_MERGE_MAIN_THRESHOLD", float)
        cls._override_if_set(config_data["qwen3"], "cluster_merge_dominant_share", "FUNASR_QWEN3_CLUSTER_MERGE_DOMINANT_SHARE", float)
        cls._override_if_set(config_data["qwen3"], "cluster_merge_dominant_threshold", "FUNASR_QWEN3_CLUSTER_MERGE_DOMINANT_THRESHOLD", float)
        cls._override_if_set(config_data["qwen3"], "cluster_merge_dominant_minor_threshold", "FUNASR_QWEN3_CLUSTER_MERGE_DOMINANT_MINOR_THRESHOLD", float)
        # silence-aware align (spike 405abf6)
        cls._override_if_set(config_data["qwen3"], "silence_align_enabled", "FUNASR_QWEN3_SILENCE_ALIGN_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["qwen3"], "silence_align_tolerance_sec", "FUNASR_QWEN3_SILENCE_ALIGN_TOLERANCE_SEC", float)
        cls._override_if_set(config_data["qwen3"], "silence_align_min_segment_dur_sec", "FUNASR_QWEN3_SILENCE_ALIGN_MIN_SEGMENT_DUR_SEC", float)
        cls._override_if_set(config_data["qwen3"], "silence_vad_noise_db", "FUNASR_QWEN3_SILENCE_VAD_NOISE_DB")
        cls._override_if_set(config_data["qwen3"], "silence_vad_min_silence_sec", "FUNASR_QWEN3_SILENCE_VAD_MIN_SILENCE_SEC", float)
        # nospk 分层切段 (diarize 开关)
        cls._override_if_set(config_data["qwen3"], "nospk_split_enabled", "FUNASR_QWEN3_NOSPK_SPLIT_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["qwen3"], "nospk_split_max_segment_sec", "FUNASR_QWEN3_NOSPK_SPLIT_MAX_SEGMENT_SEC", float)
        cls._override_if_set(config_data["qwen3"], "nospk_split_min_segment_sec", "FUNASR_QWEN3_NOSPK_SPLIT_MIN_SEGMENT_SEC", float)
        # 词级时间戳 word_align (MMS-300M CTC-FA)
        cls._override_if_set(config_data["qwen3"], "word_align_enabled", "FUNASR_QWEN3_WORD_ALIGN_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["qwen3"], "word_align_language", "FUNASR_QWEN3_WORD_ALIGN_LANGUAGE")
        cls._override_if_set(config_data["qwen3"], "word_align_model_path", "FUNASR_QWEN3_WORD_ALIGN_MODEL_PATH")
        cls._override_if_set(config_data["qwen3"], "word_align_provider", "FUNASR_QWEN3_WORD_ALIGN_PROVIDER")
        cls._override_if_set(config_data["qwen3"], "word_align_batch_size", "FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE", int)
        cls._override_if_set(config_data["qwen3"], "word_align_cuda_batch_size", "FUNASR_QWEN3_WORD_ALIGN_CUDA_BATCH_SIZE", int)
        cls._override_if_set(config_data["qwen3"], "word_align_preflight_enabled", "FUNASR_QWEN3_WORD_ALIGN_PREFLIGHT_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["qwen3"], "word_align_preflight_free_mib", "FUNASR_QWEN3_WORD_ALIGN_PREFLIGHT_FREE_MIB", int)
        cls._override_if_set(config_data["qwen3"], "word_align_sidecar_enabled", "FUNASR_QWEN3_WORD_ALIGN_SIDECAR_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["qwen3"], "word_align_sidecar_idle_ttl_sec", "FUNASR_QWEN3_WORD_ALIGN_SIDECAR_IDLE_TTL_SEC", float)
        cls._override_if_set(config_data["qwen3"], "word_align_sidecar_align_timeout_sec", "FUNASR_QWEN3_WORD_ALIGN_SIDECAR_ALIGN_TIMEOUT_SEC", float)
        # A2 治理: vendor 字段 (D4)
        cls._override_if_set(config_data["qwen3"], "backend_mlpackage_units", "FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS")
        cls._override_if_set(config_data["qwen3"], "encoder_timing_enabled", "FUNASR_QWEN3_ENCODER_TIMING", cls._parse_bool)
        # num_speakers: "" 或 "auto" 视为 None(自适应聚类), 否则转 int
        _num_spk_env = os.getenv("FUNASR_QWEN3_NUM_SPEAKERS")
        if _num_spk_env is not None:
            _s = _num_spk_env.strip().lower()
            if _s in ("", "auto", "none"):
                config_data["qwen3"]["num_speakers"] = None
            else:
                try:
                    config_data["qwen3"]["num_speakers"] = int(_s)
                except ValueError:
                    logger.error(f"FUNASR_QWEN3_NUM_SPEAKERS 必须是整数或 auto/none/空, 当前: {_num_spk_env!r}")

        # ==================== 数据库配置 ====================
        # 数据库路径由 FUNASR_DATA_DIR 构建
        data_dir = os.getenv("FUNASR_DATA_DIR")
        if data_dir:
            config_data["database"]["path"] = os.path.join(data_dir, "transcription_cache.db")

        # ==================== 通知配置 ====================
        cls._override_if_set(config_data["notification"], "enabled", "FUNASR_NOTIFICATION_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["notification"], "webhook_url", "FUNASR_WEBHOOK_URL")

        # ==================== 认证配置 ====================
        cls._override_if_set(config_data["auth"], "enabled", "FUNASR_AUTH_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["auth"], "secret_key", "FUNASR_AUTH_SECRET_KEY")

        # ==================== 日志配置 ====================
        cls._override_if_set(config_data["logging"], "level", "FUNASR_LOG_LEVEL")
        cls._override_if_set(config_data["logging"], "log_dir", "FUNASR_LOG_DIR")

        # ==================== 可观测性配置 (P1) ====================
        cls._override_if_set(config_data["observability"], "metrics_enabled", "FUNASR_METRICS_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["observability"], "metrics_token", "FUNASR_METRICS_TOKEN")
        cls._override_if_set(config_data["observability"], "alert_enabled", "FUNASR_ALERT_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["observability"], "alert_queue_saturation_ratio", "FUNASR_ALERT_QUEUE_SATURATION_RATIO", float)
        cls._override_if_set(config_data["observability"], "alert_error_surge_threshold", "FUNASR_ALERT_ERROR_SURGE_THRESHOLD", int)
        cls._override_if_set(config_data["observability"], "alert_cooldown_seconds", "FUNASR_ALERT_COOLDOWN_SECONDS", int)

        return config_data

    @staticmethod
    def _override_if_set(config_dict: Dict[str, Any], key: str, env_var: str, converter=None):
        """如果环境变量存在，则覆盖配置值"""
        value = os.getenv(env_var)
        if value is not None:
            try:
                config_dict[key] = converter(value) if converter else value
                # 配置加载阶段不输出日志,避免影响日志系统初始化
            except Exception as e:
                # 错误仍需要输出
                logger.error(f"环境变量 {env_var} 转换失败: {e}")

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """解析布尔值环境变量"""
        return value.lower() in ("true", "1", "yes", "on")

    @classmethod
    def _validate_config(cls, config: "Config"):
        """验证配置的完整性和正确性"""
        # 注意: 此方法在 setup_logger() 之前调用，所以不输出 info 级别日志
        # 只输出错误和警告（如果有的话）
        is_worker = os.getenv('FUNASR_WORKER_MODE') == '1'

        errors = []
        warnings = []

        # 验证通知配置
        if config.notification.enabled:
            if not config.notification.webhook_url:
                errors.append("通知功能已启用 (FUNASR_NOTIFICATION_ENABLED=true)，但未设置 FUNASR_WEBHOOK_URL")

        # 验证认证配置
        if config.auth.enabled:
            if config.auth.secret_key == "your-secret-key-change-this-in-production":
                warnings.append("认证功能已启用，但仍在使用默认密钥，生产环境请务必修改 FUNASR_AUTH_SECRET_KEY！")

        # 验证目录配置
        directories = {
            "模型目录": config.funasr.model_dir,
            "上传目录": config.server.upload_dir,
            "临时目录": config.server.temp_dir,
            "日志目录": config.logging.log_dir,
            "数据目录": str(Path(config.database.path).parent),
        }

        for name, path in directories.items():
            if not path:
                errors.append(f"{name} 未配置")

        # 验证端口范围
        if not (1 <= config.server.port <= 65535):
            errors.append(f"服务器端口 {config.server.port} 超出有效范围 (1-65535)")

        # 验证并发任务数
        if config.transcription.max_concurrent_tasks < 1:
            errors.append(f"最大并发任务数必须 >= 1，当前值: {config.transcription.max_concurrent_tasks}")

        # A3 治理 (D3): startup engine ↔ runtime 兼容性 fail-fast
        # default_engine=qwen3 + cuda runtime + ORT CUDA EP 缺 → 直接挂, 不留到第一个 task 才挂.
        # per-request engine ≠ default 已被 transcriber_dispatch.py:57 拒, startup 仅查 default 充分.
        cls._validate_engine_runtime(config, errors)

        # 打印警告
        for warning in warnings:
            if not is_worker:
                logger.warning(f"⚠️  {warning}")

        # 打印错误并退出
        if errors:
            if not is_worker:
                logger.error("配置验证失败，发现以下错误:")
                for error in errors:
                    logger.error(f"❌ {error}")
                logger.error("=" * 60)
                logger.error("请检查 .env 文件和 config.json 配置，修正后重新启动")
            sys.exit(1)

        # 验证通过的日志延迟到 print_config() 中输出，以保证格式统一

    def print_config(self):
        """打印配置信息（敏感信息脱敏）- 应在 setup_logger() 之后调用"""
        # Worker 模式下跳过配置打印
        is_worker = os.getenv('FUNASR_WORKER_MODE') == '1'
        if is_worker:
            return

        config = self

        # 首先输出配置验证结果
        logger.info("=" * 60)
        logger.info("✅ 配置验证通过")
        logger.info("=" * 60)
        logger.info("当前配置:")
        logger.info("-" * 60)

        # 服务器配置
        logger.info(f"[服务器]")
        logger.info(f"  地址: {config.server.host}:{config.server.port}")
        logger.info(f"  最大连接数: {config.server.max_connections}")
        logger.info(f"  最大文件大小: {config.server.max_file_size_mb} MB")
        logger.info(f"  上传目录: {config.server.upload_dir}")
        logger.info(f"  临时目录: {config.server.temp_dir}")

        # FunASR 配置
        logger.info(f"[FunASR]")
        logger.info(f"  模型: {config.funasr.model} ({config.funasr.model_revision})")
        logger.info(f"  模型目录: {config.funasr.model_dir}")
        logger.info(f"  设备: {config.funasr.device}")
        logger.info(f"  设备优先级: {config.funasr.device_priority}")
        logger.info(f"  CPU 线程数: {config.funasr.ncpu}")
        logger.info(f"  批处理大小: {config.funasr.batch_size_s}s")

        # 转录配置
        logger.info(f"[转录]")
        logger.info(f"  最大并发任务: {config.transcription.max_concurrent_tasks}")
        logger.info(f"  任务超时: {config.transcription.task_timeout_minutes} 分钟")
        logger.info(f"  缓存启用: {config.transcription.cache_enabled}")

        # 通知配置
        logger.info(f"[通知]")
        logger.info(f"  启用状态: {config.notification.enabled}")
        if config.notification.enabled and config.notification.webhook_url:
            # 脱敏显示 webhook URL (仅显示前 30 字符)
            masked_url = config.notification.webhook_url[:30] + "..." if len(config.notification.webhook_url) > 30 else config.notification.webhook_url
            logger.info(f"  Webhook URL: {masked_url}")

        # 认证配置
        logger.info(f"[认证]")
        logger.info(f"  启用状态: {config.auth.enabled}")
        if config.auth.enabled:
            logger.info(f"  密钥: ***hidden***")
            logger.info(f"  算法: {config.auth.algorithm}")

        # 日志配置
        logger.info(f"[日志]")
        logger.info(f"  级别: {config.logging.level}")
        logger.info(f"  目录: {config.logging.log_dir}")

        logger.info("=" * 60)

    def setup_directories(self):
        """创建必要的目录"""
        is_worker = os.getenv('FUNASR_WORKER_MODE') == '1'

        directories = [
            self.server.temp_dir,
            self.server.upload_dir,
            self.funasr.model_dir,
            Path(self.database.path).parent,
            self.logging.log_dir
        ]

        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                # 目录创建成功时不输出日志,避免影响日志系统初始化
            except Exception as e:
                logger.error(f"创建目录失败 {directory}: {e}")
                sys.exit(1)


# 全局配置实例
config = Config.load_from_file()
config.setup_directories()
