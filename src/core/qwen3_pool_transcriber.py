"""
Qwen3PoolTranscriber — 主进程 wrapper, 通过 multi-process worker pool 调度任务

PR3: Qwen3 引擎走 multi-process worker pool, 跟 FunASR 同一套架构.
本类替代主进程直接持有 Qwen3DiarizeTranscriber 的旧路径.

为什么:
- libllama.cpp 单 context 设计上不支持并发, 同进程多协程同时调会串台
- 改为每个 worker subprocess 独立加载 Qwen3DiarizeTranscriber, libllama context 自然 per-worker 隔离
- 主进程只负责派发任务给 pool, 不再 import vendor 引擎

接口与 Qwen3DiarizeTranscriber.transcribe 鸭子兼容(同形):
    async def transcribe(audio_path, task_id, progress_callback, output_format)
        JSON: (TranscriptionResult, raw_result_dict)
        SRT:  {format, content, file_name, file_hash, duration, processing_time, raw_result}
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from loguru import logger

from src.core.file_based_process_pool import FileBasedProcessPool
from src.core.runtime import detect_runtime
from src.models.schemas import TranscribeOptions


class Qwen3PoolTranscriber:
    """Qwen3 池化转录器 — 把 transcribe 调用转成 worker subprocess 任务派发"""

    def __init__(
        self,
        pool_size: int,
        pool: Optional[FileBasedProcessPool] = None,
        heartbeat_interval_seconds: float = 30.0,
    ):
        """
        Args:
            pool_size: worker 进程数(PoC v5 sweet spot N=3).
            pool: 注入式 pool(测试用). 不传则按 qwen3_worker_process.py 作为 entry 新建.
            heartbeat_interval_seconds: progress 心跳间隔(秒, 默认 30s).
                长音频(149min)转录可能 30+ 分钟无中间 progress, 触发 client recv timeout.
                每个心跳调一次 progress_callback, 触发 task_progress WebSocket 消息, 重置 client timer.
        """
        if pool is not None:
            self._pool = pool
        else:
            # task_dir 必须与 FunASR 池物理隔离 — 否则同机器 PM2 daemon FunASR 和
            # 临时跑的 Qwen3 测试 / 进程会抢 worker_X_*.task 文件, 导致结果串台.
            self._pool = FileBasedProcessPool(
                pool_size=pool_size,
                worker_entry_script="src/core/qwen3_worker_process.py",
                task_dir="./temp/tasks_qwen3",
            )
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    async def initialize(self):
        """提前初始化 pool(启动所有 worker subprocess 并加载模型)"""
        await self._pool.initialize()

    async def cleanup(self):
        """清理 file-based worker pool；显式 initialize 可重新启动它。"""
        await self._pool.cleanup()

    async def transcribe(
        self,
        audio_path: str,
        task_id: str,
        progress_callback: Optional[Callable] = None,
        output_format: str = "json",
        options: Optional[TranscribeOptions] = None,
    ) -> Any:
        """通过 pool 派发任务给 worker subprocess.

        options 用 model_dump() 序列化进 .task JSON (跨进程边界), worker 端
        qwen3_worker_process 解析回 TranscribeOptions (缺省兜底).

        进度通知:
        - 0% 起始 / 100% 完成
        - 中间走 heartbeat (每 heartbeat_interval_seconds 一次), 进度值线性 5 → 95 封顶
        - 心跳目的: 长音频转录(>30 min)期间触发 task_progress 消息, 避免 client recv timeout

        worker 内部的进度日志走 worker 自己的 log 文件(跨进程精细 progress 不传递).
        """
        await self._notify_progress(progress_callback, 0, task_id)

        heartbeat_task: Optional[asyncio.Task] = None
        if progress_callback is not None:
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(progress_callback, task_id)
            )

        try:
            result = await self._pool.generate_with_pool(
                audio_path=audio_path,
                extra_task_fields={
                    "output_format": output_format,
                    "options": (options or TranscribeOptions()).model_dump(),
                },
            )
        except Exception:
            raise
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"[{task_id}] heartbeat 收尾异常(忽略): {e}")

        await self._notify_progress(progress_callback, 100, task_id)
        return result

    async def _heartbeat_loop(self, callback: Callable, task_id: str):
        """心跳循环: 每 interval 调一次 callback, 进度值线性递增封顶 95"""
        try:
            pct = 5
            # 启动时立即发一次 5% (让 client 立刻收到第一个进度心跳)
            await self._notify_progress(callback, pct, task_id)
            while True:
                await asyncio.sleep(self.heartbeat_interval_seconds)
                pct = min(pct + 3, 95)
                await self._notify_progress(callback, pct, task_id)
        except asyncio.CancelledError:
            raise

    @staticmethod
    async def _notify_progress(callback: Optional[Callable], pct: int, task_id: str):
        if callback is None:
            return
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(pct)
            else:
                callback(pct)
        except Exception as e:
            logger.warning(f"[{task_id}] progress_callback 异常(忽略): {e}")


# ==================== 全局单例 + runtime-aware dispatch ====================

_qwen3_pool_singleton: Optional[Any] = None  # Qwen3PoolTranscriber 或 Qwen3InProcPool


def get_qwen3_pool_transcriber() -> Any:
    """获取 Qwen3 池化转录器单例 — runtime-aware dispatch.

    cuda runtime → Qwen3InProcPool (单进程 N 实例, 避开 CUDNN cross-process race)
    其他 runtime  → Qwen3PoolTranscriber (file-based multi-process pool)

    pool_size 共用 config.transcription.qwen3_pool_size.

    Mac (MacRuntime) 行为 100% 不变: 仍走原 multi-process pool, 跟历史一致.
    详细决策见 docs/开发/gpu加速/2026-05-23-CUDA并发突破.md.
    """
    global _qwen3_pool_singleton
    if _qwen3_pool_singleton is not None:
        return _qwen3_pool_singleton

    from src.core.config import config

    pool_size = config.transcription.qwen3_pool_size
    runtime = detect_runtime()
    if runtime.name == "cuda":
        from src.core.qwen3_inproc_pool import Qwen3InProcPool
        logger.info(
            f"[qwen3-pool] runtime=cuda → Qwen3InProcPool(pool_size={pool_size}) "
            f"(避开 CUDNN cross-process race, 见 docs/开发/gpu加速/2026-05-23)"
        )
        _qwen3_pool_singleton = Qwen3InProcPool(pool_size=pool_size)
    else:
        logger.info(
            f"[qwen3-pool] runtime={runtime.name} → Qwen3PoolTranscriber multi-process pool "
            f"(pool_size={pool_size})"
        )
        _qwen3_pool_singleton = Qwen3PoolTranscriber(pool_size=pool_size)
    return _qwen3_pool_singleton


def reset_qwen3_pool_singleton():
    """重置单例(仅测试用)"""
    global _qwen3_pool_singleton
    _qwen3_pool_singleton = None
