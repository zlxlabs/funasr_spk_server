"""
基于文件系统的进程池 - 完全独立的多进程方案
不使用共享内存，不使用Manager，通过文件系统通信
"""
import asyncio
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from src.core.config import config as global_config


_DARWIN_USER_DIR_CONFSTR = 65536
_DARWIN_USER_TEMP_DIR_CONFSTR = 65537
_DARWIN_USER_CACHE_DIR_CONFSTR = 65538
@dataclass
class _TaskLease:
    """一次 pool 调用拥有的 task/audio/result 与 process 归属。"""

    task_id: str
    worker_id: int
    process: subprocess.Popen
    task_file: Path
    result_file: Path
    audio_file: Path
class FileBasedProcessPool:
    """
    基于文件系统的进程池管理器

    通过文件系统实现进程间通信，完全避免共享状态问题
    每个工作进程完全独立运行
    """

    def __init__(
        self,
        config_path: str = "config.json",
        pool_size: Optional[int] = None,
        worker_entry_script: str = "src/core/worker_process.py",
        task_dir: str = "./temp/tasks",
    ):
        """
        初始化进程池

        Args:
            config_path: 配置文件路径（已废弃，保留用于兼容性）
            pool_size: 进程池大小
            worker_entry_script: worker 子进程入口脚本路径。默认 FunASR worker;
                Qwen3 池传 "src/core/qwen3_worker_process.py"。
            task_dir: 任务文件目录。FunASR 默认 ./temp/tasks; Qwen3 必须用独立目录
                (如 ./temp/tasks_qwen3), 避免与 prod FunASR daemon 抢 worker_X_*.task 文件。
        """
        self.pool_size = pool_size or global_config.transcription.max_concurrent_tasks
        self.worker_entry_script = worker_entry_script

        # 任务目录
        self._task_root_dir = Path(task_dir)
        self._task_root_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir = self._task_root_dir
        self._run_task_dir: Optional[Path] = None

        # 进程管理
        self.worker_processes: List[Optional[subprocess.Popen]] = []
        self._worker_tasks: Dict[int, _TaskLease] = {}
        # generate_with_pool 与 cleanup 共同等待同一个退休 Task，异常原样传递。
        self._worker_releasing: Dict[int, asyncio.Task] = {}
        self._worker_replenish_tasks: Dict[int, asyncio.Task] = {}
        self._startup_tasks: Set[asyncio.Task] = set()
        self._worker_starting: Set[int] = set()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_requested = False
        self.is_initialized = False
        self.next_worker_id = 0  # 轮询分配任务

        # 协调重启
        self._management_lock = asyncio.Lock()

        # 巡检任务
        self._health_task: Optional[asyncio.Task] = None
        self._health_check_interval = 30  # seconds

        logger.info(f"初始化文件系统进程池，池大小: {self.pool_size}")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _log_worker_states(self, context: str) -> None:
        """输出当前已知的 worker 进程状态"""
        states = []
        for idx in range(self.pool_size):
            if idx >= len(self.worker_processes):
                states.append(f"{idx}[未创建]")
                continue
            process = self.worker_processes[idx]
            if process is None:
                states.append(f"{idx}[缺失]")
                continue
            pid = process.pid
            exit_code = process.poll()
            status = "运行中" if exit_code is None else f"已退出({exit_code})"
            states.append(f"{idx}[pid={pid},{status}]")
        if states:
            logger.info(f"{context} | 工作进程状态: {', '.join(states)}")
    def _ensure_capacity(self, worker_id: int) -> None:
        """确保 worker 列表扩容到指定索引"""
        needed = worker_id + 1
        while len(self.worker_processes) < needed:
            self.worker_processes.append(None)

    def _prepare_worker_environment(self, worker_id: int) -> Tuple[Dict[str, str], Tuple[Path, ...]]:
        """仅为 Mac FunASR worker 预建 Darwin 私有目录并返回子进程环境。"""
        environment = os.environ.copy()
        if sys.platform != "darwin" or Path(self.worker_entry_script).name != "worker_process.py":
            return environment, ()

        suffix = f"funasr-worker-{os.getpid()}-{worker_id}-{uuid.uuid4().hex[:8]}"
        private_dirs = []
        try:
            for confstr_id in (
                _DARWIN_USER_DIR_CONFSTR,
                _DARWIN_USER_TEMP_DIR_CONFSTR,
                _DARWIN_USER_CACHE_DIR_CONFSTR,
            ):
                private_dir = Path(os.confstr(confstr_id)) / suffix
                private_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
                private_dirs.append(private_dir)
        except Exception:
            self._remove_owned_paths(tuple(private_dirs))
            raise
        environment["DIRHELPER_USER_DIR_SUFFIX"] = suffix
        return environment, tuple(private_dirs)
    async def _reap_worker_process(self, process: subprocess.Popen, force: bool = False) -> None:
        """等待 worker 退出；必要时严格执行 terminate→wait→kill→wait。"""
        loop = asyncio.get_event_loop()
        if process.poll() is not None:
            await loop.run_in_executor(None, process.wait)
            return
        if force:
            process.terminate()
            try:
                await loop.run_in_executor(None, process.wait, 1)
            except subprocess.TimeoutExpired:
                process.kill()
                await loop.run_in_executor(None, process.wait, 5)
            return
        try:
            await loop.run_in_executor(None, process.wait, 1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                await loop.run_in_executor(None, process.wait, 1)
            except subprocess.TimeoutExpired:
                process.kill()
                await loop.run_in_executor(None, process.wait, 5)
    def _remove_owned_paths(self, paths: Tuple[Path, ...]) -> None:
        """只删除当前 lease 明确创建的文件和 Darwin 私有目录。"""
        for path in paths:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def _close_worker_log(self, process: subprocess.Popen) -> None:
        log_file = getattr(process, "_worker_log_file", None)
        if log_file is not None:
            log_file.close()
            process._worker_log_file = None
    def _worker_private_dirs_for_process(
        self, process: subprocess.Popen
    ) -> Tuple[Path, ...]:
        """读取 Popen 保存的 Darwin 私有目录归属，避免 pool 状态重复同步。"""
        return getattr(process, "_worker_private_dirs", ())
    def _launch_worker_process(self, worker_id: int) -> subprocess.Popen:
        """在当前事件循环内同步启动进程"""
        cmd = [
            sys.executable,
            self.worker_entry_script,
            "--worker-id",
            str(worker_id),
            "--task-dir",
            str(self.task_dir),
        ]

        # 创建日志目录
        log_dir = Path("logs/workers")
        log_dir.mkdir(parents=True, exist_ok=True)

        # 为每个 worker 创建独立的日志文件
        worker_log = log_dir / f"worker_{worker_id}.log"

        # 打开日志文件（追加模式）
        log_file = open(worker_log, 'a', encoding='utf-8')

        # 写入分隔符，标记新进程启动
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"Worker {worker_id} 启动 @ {timestamp}\n")
        log_file.write(f"{'='*80}\n")
        log_file.flush()
        private_dirs: Tuple[Path, ...] = ()
        try:
            environment, private_dirs = self._prepare_worker_environment(worker_id)
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    startupinfo=startupinfo,
                    text=True,
                    encoding="utf-8",
                    env=environment,
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    text=True,
                    env=environment,
                )
        except Exception:
            self._remove_owned_paths(private_dirs)
            log_file.close()
            raise
        # 将文件对象保存到进程对象中，以便后续关闭
        process._worker_log_file = log_file
        process._worker_private_dirs = private_dirs

        logger.debug(f"工作进程 {worker_id} 日志输出到: {worker_log}")
        return process
    async def _wait_for_worker_ready(
        self,
        worker_id: int,
        process: subprocess.Popen,
        timeout: int = 300,
    ) -> None:
        """等待 worker 写入 ready 文件"""
        ready_file = self.task_dir / f"worker_{worker_id}.ready"
        start = time.time()
        loop = asyncio.get_event_loop()

        while time.time() - start < timeout:
            if ready_file.exists():
                ready_pid = ready_file.read_text(encoding="utf-8").strip()
                if ready_pid == str(process.pid):
                    return

            # 如果进程已退出则立即失败
            exit_code = await loop.run_in_executor(None, process.poll)
            if exit_code is not None:
                raise RuntimeError(f"工作进程 {worker_id} 在就绪前退出 (exit={exit_code})")

            await asyncio.sleep(0.5)

        raise TimeoutError(f"工作进程 {worker_id} 在 {timeout} 秒内未写入就绪标记")
    async def _spawn_worker(self, worker_id: int) -> None:
        """创建 worker；同一 slot 的并发调用只等待同一个 process。"""
        async with self._management_lock:
            if self._shutdown_requested:
                raise RuntimeError("进程池正在关闭，拒绝启动 worker")
            self._ensure_capacity(worker_id)
            process = self.worker_processes[worker_id]
            if process is not None and process.poll() is None:
                needs_ready = worker_id in self._worker_starting
            else:
                if worker_id in self._worker_tasks:
                    return
                if process is not None:
                    await self._reap_worker_process(process)
                    self._close_worker_log(process)
                    self._remove_owned_paths(self._worker_private_dirs_for_process(process))
                self.task_dir.joinpath(f"worker_{worker_id}.ready").unlink(missing_ok=True)
                self.task_dir.joinpath(f"worker_{worker_id}.stop").unlink(missing_ok=True)
                process = self._launch_worker_process(worker_id)
                self.worker_processes[worker_id] = process
                self._worker_starting.add(worker_id)
                needs_ready = True

        if needs_ready:
            try:
                await self._wait_for_worker_ready(worker_id, process)
            except Exception as spawn_error:
                logger.error(f"工作进程 {worker_id} 启动失败: {spawn_error}")
                await self._reap_worker_process(process, force=True)
                self._close_worker_log(process)
                self._remove_owned_paths(self._worker_private_dirs_for_process(process))
                async with self._management_lock:
                    if self.worker_processes[worker_id] is process:
                        self.worker_processes[worker_id] = None
                raise
            finally:
                self._worker_starting.discard(worker_id)
        logger.info(f"工作进程 {worker_id} 已启动 (PID: {process.pid})")
    async def _replenish_worker(self, worker_id: int) -> None:
        """后台补齐正常完成后退役的 worker，不阻塞已交付结果。"""
        try:
            await self._spawn_worker(worker_id)
        except asyncio.CancelledError:
            raise
        except Exception as replenish_error:
            logger.error(f"工作进程 {worker_id} 后台补齐失败: {replenish_error}")
        finally:
            current_task = asyncio.current_task()
            if self._worker_replenish_tasks.get(worker_id) is current_task:
                self._worker_replenish_tasks.pop(worker_id, None)
    def _schedule_worker_replenish(self, worker_id: int) -> None:
        """登记单个 slot 的后台补齐任务，避免结果交付等待模型 ready。"""
        if not self.is_initialized or self._shutdown_requested:
            return
        existing = self._worker_replenish_tasks.get(worker_id)
        if existing is not None and not existing.done():
            return
        self._worker_replenish_tasks[worker_id] = asyncio.create_task(
            self._replenish_worker(worker_id)
        )
    # ------------------------------------------------------------------
    # 巡检任务
    # ------------------------------------------------------------------
    def _start_health_monitor(self) -> None:
        """启动后台巡检任务"""
        if self._health_task and not self._health_task.done():
            return
        loop = asyncio.get_event_loop()
        self._health_task = loop.create_task(self._health_check_loop())
        logger.debug("工作进程健康巡检任务已启动")

    def _stop_health_monitor(self) -> None:
        """停止后台巡检任务"""
        if not self._health_task:
            return

        task = self._health_task
        self._health_task = None

        def _finalizer(t: asyncio.Task) -> None:
            try:
                t.result()
            except asyncio.CancelledError:
                pass
            except Exception as err:
                logger.debug(f"巡检任务结束时捕获的异常: {err}")

        task.add_done_callback(_finalizer)
        task.cancel()
        logger.debug("工作进程健康巡检任务已请求停止")

    async def _health_check_loop(self) -> None:
        """周期性校验工作进程数量"""
        try:
            while self.is_initialized:
                await asyncio.sleep(self._health_check_interval)
                if not self.is_initialized:
                    break
                try:
                    await self._ensure_workers_alive()
                except asyncio.CancelledError:
                    raise
                except Exception as monitor_error:
                    logger.error(f"巡检任务执行失败: {monitor_error}")
        except asyncio.CancelledError:
            pass
        finally:
            logger.debug("工作进程健康巡检任务结束")

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------
    async def initialize(self):
        """初始化进程池 - 启动独立的工作进程"""
        async with self._management_lock:
            if self.is_initialized:
                return
            if self._cleanup_task is not None and not self._cleanup_task.done():
                raise RuntimeError("进程池正在清理，请稍后显式 initialize()")
            if self._run_task_dir is not None or self.worker_processes:
                raise RuntimeError("进程池上次清理未完成，请先完成 cleanup()")

            logger.info(f"启动 {self.pool_size} 个独立工作进程...")
            self._shutdown_requested = False
            self._run_task_dir = self._task_root_dir / f"run-{uuid.uuid4().hex}"
            self._run_task_dir.mkdir(mode=0o700)
            self.task_dir = self._run_task_dir
        spawn_tasks: List[asyncio.Task] = []
        try:
            spawn_tasks = [
                asyncio.create_task(self._spawn_worker(i)) for i in range(self.pool_size)
            ]
            async with self._management_lock:
                self._startup_tasks.update(spawn_tasks)
            spawn_results = await asyncio.gather(*spawn_tasks, return_exceptions=True)
            for spawn_result in spawn_results:
                if isinstance(spawn_result, BaseException):
                    raise spawn_result

            async with self._management_lock:
                if self._shutdown_requested:
                    raise RuntimeError("进程池在初始化期间开始关闭")
                self.is_initialized = True
            self._log_worker_states("初始化完成")
            self._start_health_monitor()

        except Exception as e:
            logger.error(f"进程池初始化失败: {e}")
            await self.cleanup()
            raise
        finally:
            async with self._management_lock:
                self._startup_tasks.difference_update(spawn_tasks)

    async def _ensure_workers_alive(self):
        """检查并补齐所有工作进程"""
        if not self.is_initialized or self._shutdown_requested:
            return
        self._log_worker_states("巡检前")
        missing_worker_ids = []
        for worker_id in range(self.pool_size):
            async with self._management_lock:
                self._ensure_capacity(worker_id)
                process = self.worker_processes[worker_id]
                needs_spawn = (
                    worker_id not in self._worker_tasks
                    and worker_id not in self._worker_starting
                    and (process is None or process.poll() is not None)
                )
            if needs_spawn:
                logger.warning(f"检测到工作进程 {worker_id} 不可用，尝试重启")
                missing_worker_ids.append(worker_id)

        if missing_worker_ids:
            spawn_results = await asyncio.gather(
                *(self._spawn_worker(worker_id) for worker_id in missing_worker_ids),
                return_exceptions=True,
            )
            for spawn_result in spawn_results:
                if isinstance(spawn_result, BaseException):
                    raise spawn_result
        self._log_worker_states("巡检后")
    async def _acquire_worker_lease(
        self, task_id: str, task_file: Path, result_file: Path, audio_file: Path
    ) -> _TaskLease:
        """在管理锁内占用一个 ready 且空闲的 worker slot。"""
        while True:
            if not self.is_initialized or self._shutdown_requested:
                raise RuntimeError("进程池正在关闭，拒绝新任务")
            async with self._management_lock:
                if not self.is_initialized or self._shutdown_requested:
                    raise RuntimeError("进程池正在关闭，拒绝新任务")
                for offset in range(self.pool_size):
                    worker_id = (self.next_worker_id + offset) % self.pool_size
                    process = self.worker_processes[worker_id] if worker_id < len(self.worker_processes) else None
                    if process is None or process.poll() is not None:
                        continue
                    if worker_id in self._worker_tasks or worker_id in self._worker_starting:
                        continue
                    lease = _TaskLease(
                        task_id=task_id,
                        worker_id=worker_id,
                        process=process,
                        task_file=task_file,
                        result_file=result_file,
                        audio_file=audio_file,
                    )
                    self._worker_tasks[worker_id] = lease
                    self.next_worker_id = (worker_id + 1) % self.pool_size
                    return lease
            await self._ensure_workers_alive()
            await asyncio.sleep(0.01)
    async def _release_worker_lease(
        self, lease: _TaskLease, force: bool, replenish: bool = False
    ) -> None:
        """等待单一退休任务；正常完成后补齐 worker，取消或关闭时不补齐。"""
        async with self._management_lock:
            if self._worker_tasks.get(lease.worker_id) is not lease:
                return
            release_task = self._worker_releasing.get(lease.worker_id)
            if release_task is None or release_task.done():
                release_task = asyncio.create_task(
                    self._finish_worker_lease_release(lease, force, replenish)
                )
                self._worker_releasing[lease.worker_id] = release_task

        await asyncio.shield(release_task)

    async def _finish_worker_lease_release(
        self, lease: _TaskLease, force: bool, replenish: bool
    ) -> None:
        """回收 lease 拥有的 process 与文件，失败时保留归属供 cleanup 重试。"""
        current_task = asyncio.current_task()
        try:
            await self._reap_worker_process(lease.process, force=force)
            self._remove_owned_paths(
                (
                    lease.task_file,
                    lease.result_file,
                    lease.audio_file,
                    lease.audio_file.with_suffix(".converted.wav"),
                )
            )
            self._remove_owned_paths(self._worker_private_dirs_for_process(lease.process))
        except BaseException as release_error:
            logger.error(f"工作进程 {lease.worker_id} 资源回收失败: {release_error}")
            raise

        async with self._management_lock:
            if self._worker_tasks.get(lease.worker_id) is lease:
                self._worker_tasks.pop(lease.worker_id, None)
            if self._worker_releasing.get(lease.worker_id) is current_task:
                self._worker_releasing.pop(lease.worker_id, None)
            if lease.worker_id < len(self.worker_processes):
                if self.worker_processes[lease.worker_id] is lease.process:
                    self.worker_processes[lease.worker_id] = None
        self._close_worker_log(lease.process)
        if replenish:
            self._schedule_worker_replenish(lease.worker_id)

    def _write_task_json(self, task_file: Path, task_data: Dict[str, Any]) -> None:
        """完整写入任务 JSON 后原子发布 `.task`，避免 worker 读取半文件。"""
        temporary = task_file.with_name(
            f".{task_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(task_data, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, task_file)
        finally:
            temporary.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # 任务调度
    # ------------------------------------------------------------------
    async def generate_with_pool(
        self,
        audio_path: str,
        batch_size_s: int = 300,
        hotword: str = "",
        use_pickle: bool = True,
        extra_task_fields: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        使用进程池进行推理

        Args:
            extra_task_fields: 引擎特定的额外任务字段(默认 None).
                FunASR 路径不传, Qwen3 池传 {"output_format": "json"/"srt"} 给 worker.
        """
        source_audio_path = Path(audio_path)
        if not source_audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if not self.is_initialized:
            if self._shutdown_requested:
                raise RuntimeError("进程池已关闭，请先显式 initialize()")
            await self.initialize()

        task_id = str(uuid.uuid4())
        local_audio_ext = source_audio_path.suffix or ".wav"
        local_audio_path = self.task_dir / f"{task_id}{local_audio_ext}"
        task_file = self.task_dir / f"worker_pending_{task_id}.task"
        result_file = (
            self.task_dir / f"worker_pending_{task_id}.pkl"
            if use_pickle
            else self.task_dir / f"worker_pending_{task_id}.result"
        )
        lease = await self._acquire_worker_lease(task_id, task_file, result_file, local_audio_path)
        worker_id = lease.worker_id
        task_file = self.task_dir / f"worker_{worker_id}_{task_id}.task"
        result_file = (
            self.task_dir / f"worker_{worker_id}_{task_id}.pkl"
            if use_pickle
            else self.task_dir / f"worker_{worker_id}_{task_id}.result"
        )
        lease.task_file = task_file
        lease.result_file = result_file

        try:
            shutil.copy2(source_audio_path, local_audio_path)
            task_data = {
                "task_id": task_id,
                "audio_path": str(local_audio_path),
                "source_audio_path": str(source_audio_path),
                "batch_size_s": batch_size_s,
                "hotword": hotword,
                "use_pickle": use_pickle,
            }
            if extra_task_fields:
                task_data.update(extra_task_fields)
            self._write_task_json(task_file, task_data)
        except asyncio.CancelledError:
            await self._release_worker_lease(lease, force=True)
            raise
        except Exception as copy_error:
            await self._release_worker_lease(lease, force=True)
            raise RuntimeError(f"复制音频文件失败: {copy_error}") from copy_error

        logger.debug(f"任务 {task_id} 分配给工作进程 {worker_id}")

        max_wait_time = self._calculate_timeout(audio_path)
        start_time = time.time()
        logger.info(f"任务 {task_id} 设置超时时间: {max_wait_time/60:.1f} 分钟")

        try:
            while time.time() - start_time < max_wait_time:
                if result_file.exists():
                    if result_file.suffix == ".pkl":
                        with result_file.open("rb") as handle:
                            result_data = pickle.load(handle)
                    else:
                        result_data = json.loads(result_file.read_text(encoding="utf-8"))
                    if (
                        result_data.get("task_id") != lease.task_id
                        or result_data.get("worker_pid") != lease.process.pid
                    ):
                        raise RuntimeError(f"result ownership mismatch for task {task_id}")
                    await self._release_worker_lease(lease, force=False, replenish=True)
                    lease = None
                    if result_data["success"]:
                        logger.debug(f"任务 {task_id} 处理成功 (工作进程 {worker_id})")
                        return result_data["result"]
                    error_msg = result_data.get("error", "未知错误")
                    logger.error(f"任务 {task_id} 处理失败: {error_msg}")
                    raise Exception(f"处理失败: {error_msg}")

                if lease.process.poll() is not None:
                    # worker 可能刚发布 result 就退出；确认退出后再读一次，
                    # 避免把已完成任务误报为 crash。
                    await asyncio.get_running_loop().run_in_executor(
                        None, lease.process.wait
                    )
                    if result_file.exists():
                        continue
                    exit_code = lease.process.returncode
                    await self._release_worker_lease(lease, force=False)
                    lease = None
                    raise RuntimeError(f"worker {worker_id} exited (exit={exit_code})")
                if not self.is_initialized or self._shutdown_requested:
                    raise RuntimeError("进程池正在关闭，任务未完成")
                await asyncio.sleep(0.1)
            raise TimeoutError(f"任务处理超时（{max_wait_time}秒）")
        except asyncio.CancelledError:
            if lease is not None:
                await self._release_worker_lease(lease, force=True)
            raise
        except Exception:
            if lease is not None:
                await self._release_worker_lease(lease, force=True)
            raise

    # ------------------------------------------------------------------
    # 其它工具
    # ------------------------------------------------------------------
    def _calculate_timeout(self, audio_path: str) -> int:
        """根据音频时长动态计算超时时间"""
        try:
            import librosa

            duration = librosa.get_duration(filename=audio_path)

            base_timeout = 300  # 5 分钟
            duration_factor = 0.3  # 每秒音频需要 0.3 秒处理
            calculated_timeout = int(base_timeout + duration * duration_factor)

            min_timeout = 600
            max_timeout = 3600

            timeout = max(min_timeout, min(calculated_timeout, max_timeout))

            logger.info(
                f"音频时长: {duration:.1f}s ({duration/60:.1f}分钟), 计算超时: {timeout}s ({timeout/60:.1f}分钟)"
            )
            return timeout

        except Exception as e:
            logger.warning(f"无法获取音频时长，使用默认超时: {e}")
            return 1200  # 默认20分钟

    def _is_worker_alive(self, worker_id: int) -> bool:
        """检查工作进程是否存活"""
        if worker_id < len(self.worker_processes):
            process = self.worker_processes[worker_id]
            return process is not None and process.poll() is None
        return False

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    async def cleanup(self):
        """停止 health，并行回收 process；失败时保留归属与 run 目录。"""
        async with self._management_lock:
            cleanup_task = self._cleanup_task
            if cleanup_task is None or cleanup_task.done():
                cleanup_task = asyncio.create_task(self._cleanup_owned_resources())
                self._cleanup_task = cleanup_task

        try:
            await asyncio.shield(cleanup_task)
        finally:
            if cleanup_task.done():
                async with self._management_lock:
                    if self._cleanup_task is cleanup_task:
                        self._cleanup_task = None

    async def _cleanup_idle_worker(
        self, worker_id: int, process: subprocess.Popen
    ) -> None:
        """回收无 lease worker；reap 或私有目录清理失败则保留 slot。"""
        await self._reap_worker_process(process, force=True)
        self._remove_owned_paths(self._worker_private_dirs_for_process(process))
        self._close_worker_log(process)
        async with self._management_lock:
            if worker_id < len(self.worker_processes):
                if self.worker_processes[worker_id] is process:
                    self.worker_processes[worker_id] = None

    async def _cleanup_owned_resources(self) -> None:
        """cleanup 的唯一所有者，等待所有 reap 完成后才删除生命周期目录。"""
        logger.info("清理进程池资源...")
        self.is_initialized = False
        self._shutdown_requested = True
        self._stop_health_monitor()

        async with self._management_lock:
            startup_tasks = list(self._startup_tasks)
            replenish_tasks = list(self._worker_replenish_tasks.values())
            leases = list(self._worker_tasks.values())
            active_worker_ids = set(self._worker_tasks)
            idle_processes = [
                (worker_id, process)
                for worker_id, process in enumerate(self.worker_processes)
                if process is not None and worker_id not in active_worker_ids
            ]
            self._worker_starting.clear()

        for task in (*startup_tasks, *replenish_tasks):
            if not task.done():
                task.cancel()
        if startup_tasks or replenish_tasks:
            await asyncio.gather(*startup_tasks, *replenish_tasks, return_exceptions=True)

        release_operations = [
            self._release_worker_lease(lease, force=True) for lease in leases
        ]
        idle_operations = [
            self._cleanup_idle_worker(worker_id, process)
            for worker_id, process in idle_processes
        ]
        results = await asyncio.gather(
            *release_operations, *idle_operations, return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            for cleanup_error in errors:
                logger.error(f"进程池 cleanup 未完成: {cleanup_error}")
            raise errors[0]

        async with self._management_lock:
            if self._worker_tasks or self._worker_releasing:
                raise RuntimeError("进程池仍有未完成的 worker lease 回收")
            remaining_processes = [process for process in self.worker_processes if process is not None]
            if any(process.poll() is None for process in remaining_processes):
                raise RuntimeError("进程池仍有运行中的 worker，保留资源目录")
            run_task_dir = self._run_task_dir

        if run_task_dir is not None:
            shutil.rmtree(run_task_dir)

        async with self._management_lock:
            self.worker_processes.clear()
            self._worker_replenish_tasks.clear()
            self._run_task_dir = None
            self.task_dir = self._task_root_dir
        self._log_worker_states("清理完成")
        logger.info("进程池资源已清理")

    def __del__(self):
        """析构函数 - 确保资源清理.

        解释器关闭路径调用时 main thread event loop 已不存在,
        asyncio.get_event_loop() 抛 RuntimeError. 全局 try/except 兜底,
        生产端走 src/main.py SIGTERM signal_handler 主动 cleanup, 这里只是防
        stderr 噪声.
        """
        try:
            if hasattr(self, "is_initialized") and self.is_initialized:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
                else:
                    loop.run_until_complete(self.cleanup())
        except Exception:
            pass


# 全局进程池实例
file_based_pool = None


def get_file_based_pool(
    config_path: str = "config.json", pool_size: Optional[int] = None
) -> FileBasedProcessPool:
    """
    获取全局进程池实例
    """
    global file_based_pool
    if file_based_pool is None:
        file_based_pool = FileBasedProcessPool(config_path, pool_size)
    return file_based_pool
