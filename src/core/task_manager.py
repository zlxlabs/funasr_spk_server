"""
任务队列管理模块
"""
import asyncio
import re
import uuid
from collections import Counter, OrderedDict
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from src.core.config import config
from src.core.database import db_manager
from src.models.schemas import (
    FileUploadRequest,
    TaskStatus,
    TranscribeOptions,
    TranscriptionResult,
    TranscriptionTask,
    resolve_word_align,
)


# 终态状态集合：内存清理只清这些；非终态（PENDING/PROCESSING）永不被清。
_TERMINAL_STATUSES = frozenset({
    TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT,
})
# 处理时长 EMA 冷启动默认值（秒）。无历史样本时用它估 retry_after / estimated_wait。
_COLD_START_TASK_SECONDS = 90.0
# EMA 平滑系数（新样本权重）。0.3 = 适度跟随近期，抗长音频离群点污染。
_EMA_ALPHA = 0.3
# 已清理 task_id 的追踪上界（用于轮询区分 expired vs not_found）。
_EVICTED_TRACK_MAX = 10000


class QueueFullError(Exception):
    """任务队列已满（准入控制拒绝）。

    替代泛化 Exception，携带结构化字段让上层（websocket_handler）映射成
    可重试的 queue_full 信号（429 语义）：客户端据 retry_after 退避重投，
    而不是当成致命错误。
    """

    def __init__(self, retry_after: int, queue_size: int, max_queue_size: int):
        self.retry_after = retry_after          # 建议重试秒数（按队列位置 × 平均处理时长估）
        self.queue_size = queue_size            # 当前队列深度
        self.max_queue_size = max_queue_size    # 队列上限
        super().__init__(
            f"任务队列已满，最大容量: {max_queue_size}（当前 {queue_size}），"
            f"建议 {retry_after}s 后重试"
        )


class ErrorKind(str, Enum):
    """错误分类(#3 接收端): record_error kind + 重试决策的单一来源。

    替代散落的 _should_retry_error(中文子串匹配)+ _is_model_error(正则)。
    websocket 层软错误 kind(invalid_format / file_too_large / ...)本轮不纳入(范围二);
    底层引擎类型化异常(EngineInitError 等)= TODOS #3 完整版发送端, 本轮靠字符串兜底过渡。
    """
    QUEUE_FULL = "queue_full"
    TIMEOUT = "timeout"
    NON_RETRYABLE_INPUT = "non_retryable_input"  # 音频太短/格式/文件不存在/太大/认证 — 不重试
    MODEL_ERROR = "model_error"                   # VAD/index/dimension — 可重试 + 重置模型
    ENGINE_ERROR = "engine_error"                 # 其它转录异常 — 可重试

    @property
    def retryable(self) -> bool:
        """codex #11: 显式集合判定, 非 "not NON_RETRYABLE_INPUT" 取反 —
        否则 QUEUE_FULL / TIMEOUT 会被误判可重试, 坑下一个 caller。"""
        return self in _RETRYABLE_KINDS

    @property
    def is_model(self) -> bool:
        """模型相关错误: 触发 _try_reset_model(仅 funasr, 见 _process_task)。"""
        return self is ErrorKind.MODEL_ERROR


# 可重试 kind(显式集合, codex #11 防取反误判)
_RETRYABLE_KINDS = frozenset({ErrorKind.MODEL_ERROR, ErrorKind.ENGINE_ERROR})

# 不可重试: 文件/输入本身的问题, 重试无意义(沿用原 _should_retry_error 列表)
_NON_RETRYABLE_MARKERS = (
    "音频时长过短", "文件不存在", "不支持的文件格式", "文件太大", "认证失败",
)

# 模型相关错误: 可重试 + 触发模型重置(沿用原 _is_model_error 正则)
_MODEL_ERROR_RE = re.compile(
    r"VAD algorithm|index .* out of bounds|list index out of range|window size|dimension",
    re.IGNORECASE,
)


def classify_error(exc: Exception) -> ErrorKind:
    """单一错误分类入口: isinstance 优先(类型化异常), 字符串兜底(底层裸 Exception)。

    底层 funasr/qwen3 仍抛裸 Exception+中文 message(范围一不改底层), 故字符串兜底是过渡;
    待发送端类型化(TODOS #3 完整版)后, 此处加 isinstance 分支即可, 调用方无需改。
    注: classify_error(QueueFullError) 在 worker catch 里 dead(QueueFullError 在 submit_task
    抛, 不进 worker catch), 保留是为分类入口完整 + submit 复用其 kind(codex #12)。
    """
    if isinstance(exc, QueueFullError):
        return ErrorKind.QUEUE_FULL
    msg = str(exc)
    if any(m in msg for m in _NON_RETRYABLE_MARKERS):
        return ErrorKind.NON_RETRYABLE_INPUT
    if _MODEL_ERROR_RE.search(msg):
        return ErrorKind.MODEL_ERROR
    return ErrorKind.ENGINE_ERROR


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, TranscriptionTask] = {}
        self.task_queue = asyncio.Queue(maxsize=config.transcription.max_queue_size)
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=config.transcription.max_concurrent_tasks)
        self.is_running = False
        self.processing_tasks = 0  # 当前正在处理的任务数
        self._queue_lock = asyncio.Lock()  # 队列操作锁
        # 处理时长 EMA（None=冷启动，用 _COLD_START_TASK_SECONDS）。驱动 retry_after / estimated_wait。
        self._processing_seconds_ema: Optional[float] = None
        # 后台维护循环（内存清理 + 看门狗）句柄
        self._maintenance_task: Optional[asyncio.Task] = None
        # 已被内存清理的 task_id（有界 LRU）：让轮询能区分 expired(清过) vs not_found(从未有)
        self._evicted_task_ids: "OrderedDict[str, None]" = OrderedDict()
        # ===== 可观测性 (P1) 单调计数器 (A1/codex #14) =====
        # 终态/错误计数必须是累加的单调 Counter，不能扫 self.tasks 现算——self.tasks
        # 被 TTL 淘汰，扫描值是"当前驻留 gauge"，Prometheus rate() 会在任务被清后静默回退。
        self._terminal_counter: "Counter[str]" = Counter()  # {status_value: 累计数}
        self._error_counter: "Counter[str]" = Counter()     # {kind: 累计数}
    
    async def start(self):
        """启动任务管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_tasks = 0
        
        # 启动工作线程
        for i in range(config.transcription.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        
        # 启动后台维护循环（self.tasks 内存清理 + processing 看门狗）
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        logger.info(f"任务管理器已启动，{config.transcription.max_concurrent_tasks}个工作线程，队列最大容量: {config.transcription.max_queue_size}")

    async def stop(self):
        """停止任务管理器"""
        self.is_running = False

        # 取消所有工作线程
        for worker in self.workers:
            worker.cancel()
        # 取消维护循环
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            self.workers.append(self._maintenance_task)

        # 等待所有工作线程结束
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("任务管理器已停止")
    
    async def create_task(self, request: FileUploadRequest, task_id: str = None) -> TranscriptionTask:
        """创建新任务

        Raises:
            ValueError: request.engine 非空且与 server 配置不匹配(全局唯一引擎模式).
                       错误信息同时含 server engine + requested engine, 客户端可据此调整.
        """
        # 生成或使用提供的任务ID
        if task_id is None:
            task_id = str(uuid.uuid4())

        # PR2: 全局唯一引擎 — request.engine 非空时必须等于 server engine, 否则 reject
        server_engine = config.transcription.default_engine
        requested = (request.engine or "").strip()
        if requested and requested != server_engine:
            raise ValueError(
                f"Server configured with engine={server_engine!r}, "
                f"cannot accept engine={requested!r}. "
                f"Please omit the engine field or set it to {server_engine!r}."
            )
        engine = requested or server_engine

        # 创建任务对象
        task = TranscriptionTask(
            task_id=task_id,
            file_name=request.file_name,
            file_path="",  # 文件路径在上传完成后设置
            file_size=request.file_size,
            file_hash=request.file_hash,
            force_refresh=request.force_refresh,
            output_format=request.output_format,
            engine=engine,
            options=TranscribeOptions(
                language=request.language,
                diarize=request.diarize,
                # terms 已由 request validator 规范化；这里只复制，不重复 normalize。
                terms=list(request.terms),
                # 决策 1A: effective word_align 在此解析一次（请求 > config 兜底），
                # 写进 options.word_align，下游 transcribe/cache/metadata 全读它。
                word_align=resolve_word_align(request.word_align, config.qwen3.word_align_enabled),
            ),
        )

        # 保存任务
        self.tasks[task_id] = task

        logger.info(f"创建任务: {task_id} - {request.file_name} (engine={engine})")

        return task
    
    def get_task(self, task_id: str) -> Optional[TranscriptionTask]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    async def submit_task(self, task_id: str, file_path: str):
        """提交任务到队列"""
        task = self.get_task(task_id)
        if not task:
            raise Exception(f"任务不存在: {task_id}")
        
        # 更新文件路径
        task.file_path = file_path
        
        # 检查缓存（命中即秒返回不入队）。force_refresh / 折维 key(word_align/diarize,
        # 折维逻辑收拢在 database.cache_params_for, D4)均在 _try_complete_from_cache 内处理。
        if await self._try_complete_from_cache(task):
            return None
        
        # 检查队列是否已满（准入控制）
        async with self._queue_lock:
            queue_size = self.task_queue.qsize()
            max_size = config.transcription.max_queue_size
            if queue_size >= max_size:
                # codex 窟窿修复：create_task 已把任务写进 self.tasks，队列满拒绝时
                # 必须回滚该插入，否则被拒的 PENDING 任务永不终态 → 永久内存泄漏。
                self.tasks.pop(task_id, None)
                logger.warning(f"队列已满拒绝任务并回滚: {task_id} (queue={queue_size}/{max_size})")
                self.record_error(ErrorKind.QUEUE_FULL.value)
                raise QueueFullError(
                    retry_after=self._compute_retry_after(),
                    queue_size=queue_size,
                    max_queue_size=max_size,
                )

            # 将任务加入队列
            try:
                self.task_queue.put_nowait(task_id)
                logger.info(f"任务已加入队列: {task_id}")

                # 返回排队信息
                queue_info = None
                if config.transcription.queue_status_enabled:
                    # 排队位置 + 预估等待（用实际处理时长 EMA，非硬编码 2 分钟）
                    position = queue_size + 1
                    estimated_wait = self._estimate_wait_seconds(position)

                    queue_info = {
                        "queued": position > config.transcription.max_concurrent_tasks,
                        "position": position,
                        "estimated_wait": estimated_wait,  # 单位: 秒
                    }

                return queue_info

            except asyncio.QueueFull:
                # 极少数竞态：锁内仍被塞满。同样回滚 self.tasks 插入。
                self.tasks.pop(task_id, None)
                self.record_error(ErrorKind.QUEUE_FULL.value)
                raise QueueFullError(
                    retry_after=self._compute_retry_after(),
                    queue_size=self.task_queue.qsize(),
                    max_queue_size=config.transcription.max_queue_size,
                )
    
    def _populate_task_from_cache(self, task, cached_result):
        """缓存命中: 组装 task.result / srt_content / metadata(等价于原 submit 命中分支)。

        DRY(commit 1): submit_task + #22 _process_task 二次查缓存共用。projected 提取 +
        metadata 构建走 result_projection.cache_hit_metadata(3 处出口同一纯函数)。
        SRT 非 srt-dict(srt_ok=False)→ 沿用原行为不设 task.result。
        """
        from src.core.result_projection import cache_hit_metadata
        md, _projected, srt_ok = cache_hit_metadata(
            cached_result, engine=task.engine, options=task.options,
            output_format=task.output_format,
        )
        if task.output_format == "srt":
            if srt_ok:
                task.srt_content = cached_result["content"]
                from src.models.schemas import TranscriptionResult
                task.result = TranscriptionResult(
                    task_id=task.task_id,
                    file_name=task.file_name,
                    file_hash=task.file_hash,
                    duration=cached_result.get("duration", 0),
                    segments=[],
                    speakers=[],
                    processing_time=0,
                    error=None,
                )
                task.result.metadata = md
        else:
            # JSON格式缓存结果
            task.result = cached_result
            task.result.metadata = md

    async def _try_complete_from_cache(self, task) -> bool:
        """查缓存; 命中则组装结果 + 终态化 + 通知, 返回 True。force_refresh 跳过。

        submit_task(提交时)+ #22 _process_task(开工前二次查, 堵错开到达重复转录)共用。
        折维 key(word_align/diarize)收拢在 cache_params_for(D4)。
        """
        if task.force_refresh:
            return False
        from src.core.database import cache_params_for
        cache_engine, allow_cross = cache_params_for(task)
        cached_result = await db_manager.get_cached_result(
            task.file_hash, task.output_format,
            engine=cache_engine, allow_cross_engine=allow_cross, options=task.options,
        )
        if not cached_result:
            return False
        logger.info(f"使用缓存结果: {task.task_id}")
        task.status = TaskStatus.COMPLETED
        self._populate_task_from_cache(task, cached_result)
        task.progress = 100   # codex #2: 终态化也 set progress, 否则轮询见 completed+progress=0
        task.error = None     # codex #3: 清失败重试残留 error
        task.completed_at = datetime.now()
        self._record_terminal(TaskStatus.COMPLETED)  # 终态化点 1: 缓存命中
        await self._notify_task_complete(task)
        return True

    # ===== 处理时长估算（EMA）+ 重试退避 =====

    def _estimate_task_seconds(self) -> float:
        """估算单任务处理时长（秒）。无历史样本走冷启动默认值。

        用 EMA 而非简单均值：抗长音频离群点污染（一个 60min 文件不会把估值带飞）。
        """
        return self._processing_seconds_ema or _COLD_START_TASK_SECONDS

    def _record_processing_seconds(self, seconds: float):
        """记录一次真实处理时长，更新 EMA。"""
        if seconds is None or seconds <= 0:
            return
        if self._processing_seconds_ema is None:
            self._processing_seconds_ema = float(seconds)
        else:
            self._processing_seconds_ema = (
                _EMA_ALPHA * float(seconds) + (1 - _EMA_ALPHA) * self._processing_seconds_ema
            )

    def _effective_concurrency(self) -> int:
        """有效并发度（队列排空速率代理）。

        qwen3 真实并发 = qwen3_pool_size（worker 阻塞在池上）；funasr = max_concurrent_tasks。
        """
        if config.transcription.default_engine == "qwen3":
            c = config.transcription.qwen3_pool_size
        else:
            c = config.transcription.max_concurrent_tasks
        return max(1, int(c))

    def _compute_retry_after(self) -> int:
        """队列满时的建议重试秒数：约等于"一个队列槽位释放"的时间。

        一个槽位在一个在途任务完成时释放，平均每 est/并发 秒释放一个。
        客户端应在此基础上加抖动避免惊群。
        """
        est = self._estimate_task_seconds()
        return max(1, int(est / self._effective_concurrency()))

    def _estimate_wait_seconds(self, position: int) -> int:
        """排在 position 位的任务预估完成等待（秒）= 前面任务数 / 并发 × 单任务时长。"""
        est = self._estimate_task_seconds()
        ahead = max(0, position - 1)
        return int(ahead / self._effective_concurrency() * est)

    # ===== 内存清理（TTL + size-cap）+ 看门狗 =====

    def _mark_evicted(self, task_id: str):
        """记录被清理的 task_id（有界 LRU），供轮询区分 expired/not_found。"""
        self._evicted_task_ids[task_id] = None
        while len(self._evicted_task_ids) > _EVICTED_TRACK_MAX:
            self._evicted_task_ids.popitem(last=False)

    def was_evicted(self, task_id: str) -> bool:
        """该 task_id 是否曾被内存清理（轮询返回 expired 而非 not_found）。"""
        return task_id in self._evicted_task_ids

    def _evict_terminal_tasks(self) -> int:
        """清理 self.tasks 中的终态任务（TTL + size-cap 双保险），非终态永不清。

        - TTL：终态任务 completed_at 超 task_retention_ttl_seconds → 清。
        - size-cap：清完 TTL 后若仍超 task_max_retained，从最老终态任务继续挤。
        返回清理数量。
        """
        ttl = config.transcription.task_retention_ttl_seconds
        cap = config.transcription.task_max_retained
        now = datetime.now()
        evicted = 0

        def _terminal_age(task) -> float:
            """终态任务的"年龄"（秒），以 completed_at 为准，缺失回退 created_at。"""
            ref = task.completed_at or task.created_at
            return (now - ref).total_seconds()

        # 1) TTL 清理
        for tid in list(self.tasks.keys()):
            task = self.tasks[tid]
            if task.status in _TERMINAL_STATUSES and _terminal_age(task) > ttl:
                del self.tasks[tid]
                self._mark_evicted(tid)
                evicted += 1

        # 2) size-cap：仍超上限则挤最老终态（非终态不计入挤出对象）
        if len(self.tasks) > cap:
            terminal = [
                (tid, _terminal_age(self.tasks[tid]))
                for tid in self.tasks
                if self.tasks[tid].status in _TERMINAL_STATUSES
            ]
            # 最老（age 大）优先挤
            terminal.sort(key=lambda x: x[1], reverse=True)
            overflow = len(self.tasks) - cap
            for tid, _ in terminal[:overflow]:
                del self.tasks[tid]
                self._mark_evicted(tid)
                evicted += 1

        if evicted:
            logger.debug(f"内存清理: 清除 {evicted} 个终态任务, 剩余 {len(self.tasks)}")
        return evicted

    def _terminalize_stale_processing(self) -> List[TranscriptionTask]:
        """看门狗：把卡死的 PROCESSING 任务强制终态化为 TIMED_OUT。

        防 ASR 卡死 / worker 异常逃逸导致任务永远 PROCESSING → 永不被清 + 客户端轮询
        永远看到 processing。终态化后该任务能被 _evict_terminal_tasks 正常回收，
        客户端轮询也能拿到明确的 timed_out 状态。

        注：本方法只改任务状态（内存 + 客户端可见性），不强杀在途的 worker await
        （真正卡死的 worker 槽位回收需 worker 级取消，超本轮止血范围）。
        返回本轮被终态化的任务列表，供锁外发送终态通知。
        """
        limit = config.transcription.task_max_processing_seconds
        now = datetime.now()
        timed_out_tasks = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PROCESSING:
                continue
            started = task.started_at or task.created_at
            if (now - started).total_seconds() > limit:
                task.status = TaskStatus.TIMED_OUT
                task.error = f"任务处理超时（>{limit}s），被看门狗强制终止"
                task.completed_at = now
                self._record_terminal(TaskStatus.TIMED_OUT)  # 终态化点 5: 看门狗
                self.record_error(ErrorKind.TIMEOUT.value)
                timed_out_tasks.append(task)
                logger.warning(f"看门狗终态化卡死任务: {task.task_id}")
        return timed_out_tasks

    async def _sweep_orphan_upload_files(self) -> int:
        """孤儿上传文件 sweeper(P4 A2):删 upload_dir 里 mtime 超宽限期 且 无 live 引用的文件。

        孤儿来源:看门狗标 TIMED_OUT 不删文件 / 删失败 / 进程重启前未删的终态文件。
        文件系统 sweeper(非内存 task 兜底):扛进程重启(文件系统为准)+ 删失败下轮重试,
        宽限期 >> task_max_processing 规避 unlink-under-worker(codex 二审定案)。

        live 引用 = 任一 PENDING/PROCESSING 任务的 file_path, 或任一 upload session 的
        finalized_file_path。delete_after_transcription off 时整体不跑(尊重"用户要留文件")。
        返回实际回收数。本方法在 _queue_lock 外调:引用集快照无 await(原子), 文件 I/O 不占锁。
        """
        if not config.transcription.delete_after_transcription:
            return 0
        import os
        import time
        from pathlib import Path

        upload_dir = Path(config.server.upload_dir)
        if not upload_dir.is_dir():
            return 0

        # ① 快照 live 引用集(无 await, 单线程原子, 不会撞 dict 改动)
        referenced = set()
        for t in self.tasks.values():
            if t.status in (TaskStatus.PENDING, TaskStatus.PROCESSING) and t.file_path:
                referenced.add(os.path.abspath(t.file_path))
        try:
            from src.api.websocket_handler import ws_handler
            for sess in ws_handler.upload_sessions.values():
                fp = sess.get("finalized_file_path")
                if fp:
                    referenced.add(os.path.abspath(fp))
        except Exception as e:  # ws_handler 不可用不阻断清理
            logger.debug(f"sweeper 读 upload_sessions 失败(忽略): {e}")

        # ② 遍历 upload_dir, 删超宽限期且无引用的文件(文件 I/O, 有 await)
        grace = config.transcription.orphan_file_grace_seconds
        now = time.time()
        from src.utils.file_utils import delete_file
        deleted = 0
        for entry in upload_dir.iterdir():
            if not entry.is_file():
                continue
            try:
                if now - entry.stat().st_mtime <= grace:
                    continue  # 宽限期内(可能在途上传), 跳过
            except OSError:
                continue
            if os.path.abspath(str(entry)) in referenced:
                continue  # live 引用, 保留
            if await delete_file(str(entry)):
                deleted += 1
        if deleted:
            logger.info(f"孤儿文件 sweeper: 回收 {deleted} 个无引用旧上传文件(>{grace}s)")
        return deleted

    async def _maintenance_loop(self):
        """后台维护循环：周期跑内存清理 + 看门狗 + 孤儿文件 sweeper。"""
        interval = config.transcription.task_cleanup_interval_seconds
        logger.info(f"任务维护循环已启动（间隔 {interval}s）")
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                async with self._queue_lock:
                    timed_out_tasks = self._terminalize_stale_processing()
                    self._evict_terminal_tasks()

                # 终态通知必须在队列锁外发送，避免网络 I/O 阻塞提交/维护；单条失败
                # 也不能阻止同一轮的其它超时任务继续收到通知。
                # 对端停读时 websocket.send 会因 TCP 背压无限阻塞——单例维护循环
                # 绝不能被拖死（看门狗/淘汰/孤儿 sweeper 全停），故单次通知加超时。
                timeout = config.transcription.maintenance_notify_timeout_sec
                for task in timed_out_tasks:
                    try:
                        await asyncio.wait_for(
                            self._notify_task_failed(
                                task, kind=ErrorKind.TIMEOUT.value
                            ),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"看门狗终态通知超时({timeout}s)，跳过 {task.task_id} —— "
                            f"客户端可能已停止读取；维护循环继续"
                        )
                    except Exception as e:
                        logger.error(f"看门狗终态通知失败 {task.task_id}: {e}")

                # 孤儿 sweeper 在锁外做文件 I/O(引用集内部快照即可一致), 不阻塞队列
                await self._sweep_orphan_upload_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"维护循环错误: {e}")
        logger.info("任务维护循环已停止")

    async def _maybe_delete_task_file(self, task, *, reason: str = "") -> bool:
        """删上传文件:delete_after_transcription on + 有 file_path + 无其他 live
        (PENDING/PROCESSING)同 hash 任务 → best-effort 删。返回是否发起删除。

        DRY(F1):cancel / complete / failed 三处终态删除逻辑此前各写一份, 统一到此。
        注:cancel 删 PROCESSING 任务文件本就有 worker race(既有, 看门狗/取消不杀在途
        worker 的已知范围);本 helper 仅去重, 不改变该行为。
        """
        if not (config.transcription.delete_after_transcription and task.file_path):
            return False
        # 同 file_hash 仍有 live(PENDING/PROCESSING)任务 → 别人还要用, 不删
        has_live_same_hash = any(
            t.file_hash == task.file_hash
            and t.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]
            for t in self.tasks.values()
            if t.task_id != task.task_id
        )
        if has_live_same_hash:
            logger.debug(f"保留文件，还有其他任务使用: {task.file_path}")
            return False
        from src.utils.file_utils import delete_file
        await delete_file(task.file_path)
        logger.debug(f"{reason}文件已删除: {task.file_path}")
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.get_task(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.error = "用户取消"
        task.completed_at = datetime.now()
        self._record_terminal(TaskStatus.CANCELLED)  # 终态化点 4: 取消

        # 删除文件（F1: 统一走 _maybe_delete_task_file）
        await self._maybe_delete_task_file(task, reason="任务取消，")

        logger.info(f"任务已取消: {task_id}")
        return True
    
    async def _worker(self, worker_id: int):
        """工作线程"""
        logger.info(f"工作线程 {worker_id} 已启动")
        
        while self.is_running:
            try:
                # 获取任务（超时1秒）
                try:
                    task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # 处理任务
                await self._process_task(task_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"工作线程 {worker_id} 错误: {e}")
        
        logger.info(f"工作线程 {worker_id} 已停止")
    
    async def _process_task(self, task_id: str):
        """处理任务"""
        task = self.get_task(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return
        
        if task.status == TaskStatus.CANCELLED:
            logger.info(f"任务已取消，跳过处理: {task_id}")
            return
        
        # 增加处理任务计数
        async with self._queue_lock:
            self.processing_tasks += 1

        try:
            # #22 in-flight 去重(只堵错开传): 同 hash 前序任务在本任务排队期间已转完写缓存,
            # 开工前再查一次命中即秒返回, 免重复转录(submit 时查那次还是空的)。必须在 try 内:
            # 查缓存自身抛错绝不逃逸到 _worker(否则任务卡 PENDING / 不再入队 / 看门狗不管 →
            # 永久泄漏, codex #1)。"几乎同时到达"(多 worker 并发取出)堵不住 — 已知局限。
            try:
                if await self._try_complete_from_cache(task):
                    await self._maybe_delete_task_file(task)  # 命中即终态, 与正常完成一致删文件
                    return
            except Exception as e:
                logger.warning(f"开工前查缓存失败, 降级继续转录 {task_id}: {e}")

            # 更新任务状态
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            
            # 通知开始处理
            await self._notify_task_progress(task, 0, "开始处理")
            
            # 定义进度回调
            async def progress_callback(progress: float):
                await self._notify_task_progress(task, progress, f"转录进度: {progress:.1f}%")
            
            # 执行转录（PR1: 用 dispatch 函数根据 task.engine 选择 transcriber）
            from src.core.transcriber_dispatch import resolve_transcriber
            transcriber = resolve_transcriber(task.engine)
            result = await transcriber.transcribe(
                audio_path=task.file_path,
                task_id=task_id,
                progress_callback=progress_callback,
                output_format=task.output_format,
                options=task.options,
            )
            
            # 更新任务结果
            # 注: status=COMPLETED 在 task.result 组装完成后才翻转 (见下), 否则
            # save_result 的 await 窗口里轮询方会看到 COMPLETED 但 result=None.
            # 缓存写入 tag 与查询同 key (折维收拢在 database.cache_params_for, D4)
            from src.core.database import cache_params_for
            cache_engine_tag = cache_params_for(task)[0]
            fresh_projected = False  # fresh 出口是否做了投影 (funasr 照算路径)
            has_words = None         # JSON 出口词级时间戳是否实际挂上 (驱动 metadata delivered)
            word_align_error_msg = None  # word_align 失败原因 (回显 metadata.word_align_error)
            if task.output_format == "srt":
                # SRT格式结果
                # 创建转录结果对象用于缓存
                # T-B: SRT 模式也存真 segments — qwen3 raw_result 无 sentence_info,
                # 缓存命中重建 SRT 必须走 segments 路径; 空 segments 会让命中返回空 content.
                from src.models.schemas import TranscriptionResult
                srt_segments = result.get("segments") or []
                transcription_result = TranscriptionResult(
                    task_id=task_id,
                    file_name=result["file_name"],
                    file_hash=result["file_hash"],
                    duration=result["duration"],
                    segments=srt_segments,
                    speakers=sorted(set(s.speaker for s in srt_segments if s.speaker)),
                    processing_time=result["processing_time"],
                    error=None
                )

                # 保存到缓存（先于投影: 缓存永远存引擎真算结果, 投影是请求级出口行为）
                await db_manager.save_result(transcription_result, result["raw_result"], engine=cache_engine_tag)

                # fresh 结果出口投影 (D3 双出口之二): funasr 照算带 speaker,
                # diarize=false 请求需投影抹 speaker + SRT 重渲染无前缀.
                # qwen3 原生 nospk 输出 (speakers 已空) 不重渲染, 保留引擎 content.
                if not task.options.diarize and transcription_result.speakers:
                    from src.core.result_projection import (
                        project_result_nospk,
                        segments_to_srt_text,
                    )
                    transcription_result = project_result_nospk(transcription_result)
                    task.srt_content = segments_to_srt_text(transcription_result.segments)
                    fresh_projected = True
                else:
                    task.srt_content = result["content"]
                task.result = transcription_result
            else:
                # JSON格式结果
                transcription_result, raw_result = result

                # 决策 B (codex #5): 请求 word_align 但 segments 实际无词 (CUDA+CPU 都失败) →
                # 写入降 +wa tag, 不毒化该文件 +wa 缓存. has_words 也驱动 metadata delivered.
                has_words = any(getattr(s, "words", None) for s in transcription_result.segments)
                from src.core.database import cache_save_engine_for
                save_tag = cache_save_engine_for(task, has_words)

                # 保存到缓存（先于投影, 同上）— funasr 存句级真值, 合并视图只在 serve 出口
                await db_manager.save_result(transcription_result, raw_result, engine=save_tag)
                # word_align 失败原因 (fresh 出口回显 metadata.word_align_error, codex #11)
                # funasr 的 raw_result 是 model.generate() 原始 list (非 dict), word_align 是
                # qwen3 专属字段 → 仅 dict 形态才取, 否则 None (修生产事故: 'list' object has no attribute 'get')
                word_align_error_msg = (
                    (raw_result.get("word_align") or {}).get("error")
                    if isinstance(raw_result, dict) else None
                )

                # D3: funasr JSON 出口应用 merge 视图 (先 merge 后 nospk; SRT 分支不应用).
                # deep copy 后再改 segments, 避免 mutate 已交给 save_result 的对象引用
                # (save 虽已 model_dump_json 序列化, 但 mock/后续路径可能仍持有原对象).
                if task.engine == "funasr":
                    from src.core.config import config as _cfg
                    from src.core.result_projection import merge_segments_view
                    transcription_result = transcription_result.model_copy(deep=True)
                    transcription_result.segments = merge_segments_view(
                        transcription_result.segments,
                        gap_sec=_cfg.transcription.segment_merge_gap_sec,
                        max_span_sec=_cfg.transcription.segment_merge_max_span_sec,
                    )

                # fresh 结果出口投影 (funasr 照算路径; qwen3 原生 nospk 幂等跳过)
                if not task.options.diarize and transcription_result.speakers:
                    from src.core.result_projection import project_result_nospk
                    transcription_result = project_result_nospk(transcription_result)
                    fresh_projected = True
                task.result = transcription_result

            # E2: effective options 回显 (serve 层组装, 缓存写入已在前完成不被污染).
            # JSON 出口传 has_words + word_align_error → metadata.word_align 反映实际交付 (codex #12);
            # SRT 出口 word_align 恒 False (output_format 透传, 决策 2A).
            from src.core.result_projection import build_result_metadata
            task.result.metadata = build_result_metadata(
                engine=task.engine, options=task.options, output_format=task.output_format,
                projected=fresh_projected,
                has_words=(has_words if task.output_format != "srt" else None),
                word_align_error=(word_align_error_msg if task.output_format != "srt" else None),
            )

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100
            self._record_terminal(TaskStatus.COMPLETED)  # 终态化点 2: 正常完成

            # 记录真实处理时长（wall）→ EMA，供 retry_after / estimated_wait 估算
            if task.started_at:
                self._record_processing_seconds(
                    (task.completed_at - task.started_at).total_seconds()
                )

            # 通知完成
            await self._notify_task_complete(task)
            
            # 删除文件（F1: 统一走 _maybe_delete_task_file）
            await self._maybe_delete_task_file(task)
            
            # 可观测性: per-task diarize 生效值 + 投影标记 (stats 三维度之一)
            logger.info(
                f"任务完成: {task_id} (engine={task.engine}, "
                f"diarize={task.options.diarize}, projected={fresh_projected})"
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"任务处理失败 {task_id}: {error_msg}")
            # #3: 单一分类入口替代字符串匹配。错误计数按 kind 分(每次异常都记, 与终态 FAILED 解耦)
            kind = classify_error(e)
            self.record_error(kind.value)

            # 更新任务状态
            task.status = TaskStatus.FAILED
            task.error = error_msg
            task.completed_at = datetime.now()

            # 重试逻辑(kind.retryable 替代 _should_retry_error 字符串匹配)
            if kind.retryable and task.retry_count < config.transcription.retry_times:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"任务重试 {task_id}: 第{task.retry_count}次")

                # 模型相关错误尝试重置模型状态。codex #13: 仅 funasr 任务才 reset funasr —
                # 多引擎下 qwen3 错误判 MODEL_ERROR 不该去 reset 无关引擎(白重启)。
                if kind.is_model and task.engine == "funasr":
                    await self._try_reset_model()
                    # 添加短暂延迟，让模型有时间恢复
                    await asyncio.sleep(2)

                await self.task_queue.put(task_id)
            else:
                # 不重试或重试次数已达上限 → 真终态 FAILED（终态化点 3）
                self._record_terminal(TaskStatus.FAILED)
                if not kind.retryable:
                    logger.warning(f"任务 {task_id} 遇到不可重试的错误({kind.value}): {error_msg}")
                # 通知失败
                await self._notify_task_failed(task, kind=kind.value)
                
                # 删除文件（F1: 统一走 _maybe_delete_task_file）
                await self._maybe_delete_task_file(task, reason="任务失败，")

                # 发送企微通知
                await self._send_wework_notification(task, "failed")
        finally:
            # 减少处理任务计数
            async with self._queue_lock:
                self.processing_tasks = max(0, self.processing_tasks - 1)
    
    async def _notify_task_progress(self, task: TranscriptionTask, progress: float, message: str):
        """通知任务进度"""
        task.progress = progress
        
        try:
            from src.api.websocket_handler import ws_handler
            await ws_handler.notify_task_progress(
                task_id=task.task_id,
                progress=progress,
                status=task.status.value,
                message=message
            )
        except Exception as e:
            logger.error(f"通知任务进度失败: {e}")
    
    async def _notify_task_complete(self, task: TranscriptionTask):
        """通知任务完成"""
        try:
            from src.api.websocket_handler import ws_handler
            
            # 根据输出格式准备结果
            if task.output_format == "srt":
                result_data = {
                    "format": "srt",
                    "content": task.srt_content,
                    "file_name": task.file_name,
                    "file_hash": task.file_hash,
                    # E2: SRT 响应没有 TranscriptionResult 载体, metadata 挂 payload 顶层
                    "metadata": task.result.metadata if task.result else None,
                }
            else:
                result_data = task.result.model_dump() if task.result else None
            
            await ws_handler.notify_task_complete(
                task_id=task.task_id,
                result=result_data
            )
            
            # 发送企微通知
            await self._send_wework_notification(task, "completed")
            
        except Exception as e:
            logger.error(f"通知任务完成失败: {e}")
    
    async def _notify_task_failed(self, task: TranscriptionTask, kind: str | None = None):
        """保留 progress 失败通知，并追加显式 error 终态通知。"""
        try:
            from src.api.websocket_handler import ws_handler
            await ws_handler.notify_task_progress(
                task_id=task.task_id,
                progress=task.progress,
                status=task.status.value,
                message=f"任务失败: {task.error}"
            )
        except Exception as e:
            logger.error(f"通知任务失败进度消息失败: {e}")

        try:
            from src.api.websocket_handler import ws_handler
            await ws_handler.notify_task_error(
                task_id=task.task_id,
                error_type=kind or "task_failed",
                message=task.error or "任务失败",
                status=task.status.value,
            )
        except Exception as e:
            logger.error(f"通知任务失败终态消息失败: {e}")
    
    def _convert_json_to_srt(self, result: TranscriptionResult) -> str:
        """将JSON格式的转录结果转换为SRT格式"""
        srt_lines = []
        
        for idx, segment in enumerate(result.segments, 1):
            # 转换时间格式
            start_time = self._seconds_to_srt_time(segment.start_time)
            end_time = self._seconds_to_srt_time(segment.end_time)
            
            # SRT格式：序号 -> 时间 -> 文本
            srt_lines.append(f"{idx}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(f"{segment.speaker}:{segment.text}")
            srt_lines.append("")  # 空行分隔
        
        return "\n".join(srt_lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """将秒数转换为SRT时间格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    
    async def _send_wework_notification(self, task: TranscriptionTask, event_type: str):
        """发送企微通知"""
        if not config.notification.enabled or not config.notification.webhook_url:
            return
        
        try:
            from src.utils.notification import send_wework_notification
            await send_wework_notification(task, event_type)
        except Exception as e:
            logger.error(f"发送企微通知失败: {e}")
    
    async def _try_reset_model(self):
        """尝试重置模型状态"""
        try:
            logger.warning("检测到模型错误，尝试重置模型状态...")
            from src.core.funasr_transcriber import get_transcriber
            transcriber = get_transcriber()
            
            # 如果模型已初始化，尝试重新初始化
            if transcriber.is_initialized:
                # 清理现有模型
                if hasattr(transcriber, 'model') and transcriber.model:
                    try:
                        # 尝试清理模型资源
                        del transcriber.model
                        transcriber.model = None
                    except:
                        pass
                
                # 重置初始化标志
                transcriber.is_initialized = False
                
                # 重新初始化模型
                await transcriber.initialize()
                logger.info("模型状态重置成功")
            else:
                logger.info("模型未初始化，跳过重置")
                
        except Exception as e:
            logger.error(f"重置模型失败: {e}")
    
    # ===== 可观测性 (P1) 计数器 + 指标快照 =====

    def _record_terminal(self, status: "TaskStatus"):
        """累加终态计数（单调，A1）。在每个终态化点调用，不扫 self.tasks。"""
        key = status.value if hasattr(status, "value") else str(status)
        self._terminal_counter[key] += 1

    def record_error(self, kind: str):
        """累加错误计数（单调，按 kind 分）。catch 点 / 软错误点调用。

        kind 约定（#3 错误分类纪律）：转录失败经 classify_error 分 engine_error /
        model_error / non_retryable_input；queue_full（准入拒绝）/ timeout（看门狗）走
        ErrorKind 枚举值；websocket 层软错误（invalid_format / file_too_large / bad_hash /
        task_not_found / auth_failed）暂传字符串（范围二待并入 ErrorKind），由各调用点传入。
        """
        self._error_counter[kind] += 1

    def get_metrics_snapshot(self) -> dict:
        """指标快照：瞬时 gauge（读现态）+ 单调 counter（累计）+ EMA + 引擎信息。

        /metrics 端点的唯一数据源（同步、不 await、不碰子进程，安全在事件循环里读）。
        """
        return {
            # ---- 瞬时 gauge（读现态，self.tasks 只在主事件循环协程读写）----
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": config.transcription.max_queue_size,
            "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "processing": self.processing_tasks,
            "tasks_in_memory": len(self.tasks),
            # ---- 单调 counter（累计，活过 TTL 淘汰）----
            "terminal_total": dict(self._terminal_counter),
            "errors_total": dict(self._error_counter),
            # ---- 估算 + 引擎 ----
            "task_seconds_ema": self._estimate_task_seconds(),
            "engine": config.transcription.default_engine,
            "pool_size": self._effective_concurrency(),
        }

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = {
            "total_tasks": len(self.tasks),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "processing_tasks": self.processing_tasks,
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            "cancelled_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.CANCELLED),
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": config.transcription.max_queue_size,
            "max_concurrent_tasks": config.transcription.max_concurrent_tasks
        }
        return stats
    
    async def adjust_concurrency(self, new_max_tasks: int):
        """动态调整并发任务数"""
        if new_max_tasks <= 0 or new_max_tasks > 32:  # 限制最大并发数
            raise ValueError("并发任务数必须在1-32之间")
        
        logger.info(f"调整并发任务数: {config.transcription.max_concurrent_tasks} -> {new_max_tasks}")
        
        # 更新配置
        config.transcription.max_concurrent_tasks = new_max_tasks
        
        # 如果需要增加worker
        current_workers = len(self.workers)
        if new_max_tasks > current_workers:
            for i in range(current_workers, new_max_tasks):
                worker = asyncio.create_task(self._worker(i))
                self.workers.append(worker)
        # 如果需要减少worker，让多余的worker自然结束
        # 这里不主动取消worker，而是通过减少配置让它们自然减少工作负载
        
        logger.info(f"并发调整完成，当前worker数: {len(self.workers)}")


# 全局任务管理器实例
task_manager = TaskManager()
