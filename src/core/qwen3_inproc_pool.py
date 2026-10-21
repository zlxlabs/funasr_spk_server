"""Qwen3InProcPool — 单进程内 N 个 Qwen3DiarizeTranscriber 实例 + asyncio.Queue 派发.

设计动机 (落档: docs/开发/gpu加速/2026-05-23-CUDA并发突破.md):
- multi-process pool 在 CUDA + ort_cuda backend 上撞 CUDNN cross-process race
- 单进程内多 llama_context + 多 ORT InferenceSession 共享同一 cuda context, race-free
- 实测 RTX 3060 + 8 vCPU 上 pool_size=2 跑 1800s × 2 并发 total wall 141.95s, RTF 0.079

实现要点:
- 构造时只存 pool_size + transcriber_factory, 不实际 build tx (lazy)
- initialize: 构造 pool_size 个 tx, 串行 init (避免同时 mmap 大权重 / 引擎 CUDA 初始化 race)
- transcribe: 从 asyncio.Queue acquire 一个 tx, 调它的 transcribe, 完成后 put 回 Queue
- 抛异常时 tx 必须回 Queue (try/finally), 否则 slot 漏掉

接口与 Qwen3PoolTranscriber 鸭子兼容 (resolve_transcriber 不区分):
    async def initialize()
    async def transcribe(audio_path, task_id, progress_callback, output_format) -> ...

只在 runtime=cuda 时被 get_qwen3_pool_transcriber() 选用. Mac 路径仍走原 multi-process pool.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from loguru import logger


class Qwen3InProcPool:
    """单进程内 N 个 transcriber 实例的池 — race-free 替代 multi-process pool"""

    def __init__(
        self,
        pool_size: int,
        transcriber_factory: Optional[Callable[[], Any]] = None,
    ):
        """
        Args:
            pool_size: 池中 transcriber 实例数 (CUDA 上建议 2, 受 GPU 显存制约).
            transcriber_factory: 无参 callable, 返回新 Qwen3DiarizeTranscriber 实例.
                                 None 时默认从 config 构造 (production 路径).
                                 测试时注入 mock factory.
        """
        if pool_size < 1:
            raise ValueError(f"pool_size 必须 >= 1, got {pool_size}")
        self.pool_size = pool_size
        self._transcriber_factory = transcriber_factory or _default_transcriber_factory
        self._txs: list = []
        self._available: Optional[asyncio.Queue] = None
        self._init_lock: Optional[asyncio.Lock] = None
        self._initialized = False

    async def initialize(self) -> None:
        """惰性构造 pool_size 个 transcriber 实例并 init.

        幂等 — 重复调用 no-op. 串行 init (相邻 model load 不撞).
        """
        # init_lock 必须 lazy 在 event loop 上下文里建 (Lock 绑定到当前 loop)
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._initialized:
                return
            self._available = asyncio.Queue()
            for i in range(self.pool_size):
                logger.info(f"[Qwen3InProcPool] 构造 transcriber 实例 {i + 1}/{self.pool_size}")
                tx = self._transcriber_factory()
                await tx.initialize()
                self._txs.append(tx)
                self._available.put_nowait(tx)
            self._initialized = True
            logger.info(f"[Qwen3InProcPool] 池就绪, pool_size={self.pool_size}")

    async def transcribe(
        self,
        audio_path: str,
        task_id: str,
        progress_callback: Optional[Callable] = None,
        output_format: str = "json",
        options: Optional[Any] = None,
    ) -> Any:
        """从 pool 取一个空闲 tx 跑 transcribe, 完成后 tx 回 pool.

        若 pool 未 init, 自动 lazy init (单次 model load).
        若 pool 中没空闲 tx (全在跑), await 排队直到有 tx 完成.
        若 tx.transcribe 抛错, tx 也必须回 pool (try/finally).
        """
        if not self._initialized:
            await self.initialize()
        assert self._available is not None  # 类型收窄, initialize() 后必非 None

        tx = await self._available.get()
        try:
            return await tx.transcribe(
                audio_path=audio_path,
                task_id=task_id,
                progress_callback=progress_callback,
                output_format=output_format,
                options=options,
            )
        finally:
            self._available.put_nowait(tx)


def _default_transcriber_factory():
    """Production 默认 factory: 从 config 构造 Qwen3DiarizeTranscriber 实例.

    跟 get_qwen3_transcriber 单例工厂的字段映射完全一致, 区别是每次调返回新实例.
    """
    from src.core.config import config
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

    q = config.qwen3
    tx = Qwen3DiarizeTranscriber(
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
    tx.embedding_model = q.embedding_model
    return tx
