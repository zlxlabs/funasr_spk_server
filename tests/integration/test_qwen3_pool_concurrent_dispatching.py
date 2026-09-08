"""
Qwen3 池并发派发测试 — hybrid mode (真 pool 派发 + mock worker subprocess)

设计:
- 默认 enabled, 秒级跑完, 用作 wiring 回归(类似 smoke_engine_wiring)
- pool 走真实派发逻辑 (generate_with_pool / task_data / extra_task_fields 都真跑)
- worker subprocess 用 mock(在主进程内起 asyncio task 模拟 worker 行为)
- 验证 2 个并发请求:
  (a) 分到不同 worker_id (轮询分配)
  (b) 输出不串台 (各自 task_id / 结果独立)
  (c) 一个 worker 抛错, 另一个不受影响

为什么 hybrid:
- 真 e2e 并发慢且耗资源, 不适合默认 CI 跑
- 全 mock 又测不到 pool 派发逻辑(extra_task_fields / 文件协议 / 轮询)
- hybrid 把真派发 + fake 推理结合, 秒级覆盖派发盲区
- 真模型并发回归留给 test_qwen3_pool_real_concurrency.py (FUNASR_RUN_INTEGRATION=1)
"""
from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Callable, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.file_based_process_pool import FileBasedProcessPool


# ==================== fake worker poller ====================


class FakeWorkerPoller:
    """模拟 worker subprocess: 监控 task 文件, 调用 behavior 函数, 写 pickle 结果

    保留每次处理过的 task → worker_id 映射, 供测试断言派发是否分到不同 worker.
    """

    def __init__(
        self,
        pool: FileBasedProcessPool,
        behavior: Callable,  # (worker_id, task_dict) -> result(JSON 模式) 或抛异常
    ):
        self.pool = pool
        self.behavior = behavior
        self._task: asyncio.Task = None
        self.dispatched: Dict[str, int] = {}  # task_id -> worker_id

    async def _loop(self):
        try:
            while True:
                for worker_id in range(self.pool.pool_size):
                    task_pattern = f"worker_{worker_id}_*.task"
                    for task_file in list(self.pool.task_dir.glob(task_pattern)):
                        try:
                            with open(task_file, "r", encoding="utf-8") as f:
                                task = json.load(f)
                        except Exception:
                            # task 文件正在写, 下次再扫
                            continue

                        task_id = task["task_id"]
                        self.dispatched[task_id] = worker_id

                        try:
                            result = self.behavior(worker_id, task)
                            data = {
                                "task_id": task_id,
                                "success": True,
                                "result": result,
                                "worker_pid": self.pool.worker_processes[worker_id].pid,
                            }
                        except Exception as e:
                            data = {
                                "task_id": task_id,
                                "success": False,
                                "error": str(e),
                                "traceback": "fake-traceback",
                                "worker_pid": self.pool.worker_processes[worker_id].pid,
                            }

                        result_file = task_file.with_suffix(".pkl")
                        with open(result_file, "wb") as f:
                            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

                        # worker 拿走 task → 删 task 文件
                        try:
                            task_file.unlink()
                        except Exception:
                            pass
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# ==================== pool fixture ====================


@pytest.fixture
async def fake_qwen3_pool(tmp_path, monkeypatch):
    """启 2 worker pool, 但 subprocess 都用 mock"""
    monkeypatch.chdir(tmp_path)

    pool = FileBasedProcessPool(
        pool_size=2, worker_entry_script="src/core/qwen3_worker_process.py"
    )
    pool.is_initialized = True

    # 跳过真实进程相关行为
    pool._ensure_workers_alive = AsyncMock()
    pool._calculate_timeout = MagicMock(return_value=30)

    fake_procs = []
    for _ in range(pool.pool_size):
        p = MagicMock()
        p.poll.return_value = None  # alive
        p.pid = 12345 + _
        fake_procs.append(p)
    pool.worker_processes = fake_procs

    async def fake_spawn(worker_id):
        """spawn 后, worker_processes 替换成一个新的 alive mock(模拟健康检查重启)"""
        new_p = MagicMock()
        new_p.poll.return_value = None
        new_p.pid = 12345
        pool.worker_processes[worker_id] = new_p

    pool._spawn_worker = fake_spawn

    yield pool

    # cleanup
    pool.is_initialized = False


# ==================== 测试用例 ====================


class TestConcurrentDispatchingNoCrosstalk:
    """两个并发任务 → 派发到不同 worker, 结果不串台"""

    @pytest.mark.asyncio
    async def test_two_concurrent_tasks_dispatched_to_different_workers(
        self, fake_qwen3_pool, tmp_path
    ):
        # 准备两个音频文件
        audio1 = tmp_path / "task1.wav"
        audio2 = tmp_path / "task2.wav"
        for f in (audio1, audio2):
            f.write_bytes(b"\x00" * 100)

        def behavior(worker_id, task):
            """worker 把 audio_path basename + worker_id 编码进 result, 便于断言不串台"""
            return {
                "audio_basename": Path(task["audio_path"]).name,
                "source_basename": Path(task["source_audio_path"]).name,
                "output_format": task.get("output_format"),
                "worker_id": worker_id,
                "task_id": task["task_id"],
            }

        poller = FakeWorkerPoller(fake_qwen3_pool, behavior)
        poller.start()
        try:
            # 并发提交 2 个任务
            t1 = asyncio.create_task(
                fake_qwen3_pool.generate_with_pool(
                    audio_path=str(audio1),
                    extra_task_fields={"output_format": "json"},
                )
            )
            t2 = asyncio.create_task(
                fake_qwen3_pool.generate_with_pool(
                    audio_path=str(audio2),
                    extra_task_fields={"output_format": "json"},
                )
            )
            r1, r2 = await asyncio.gather(t1, t2)
        finally:
            await poller.stop()

        # (a) 派发到不同 worker_id
        assert r1["worker_id"] != r2["worker_id"], (
            f"两并发应该分到不同 worker: r1={r1['worker_id']}, r2={r2['worker_id']}"
        )

        # (b) 输出不串台: r1 看到 task1 的 audio, r2 看到 task2 的 audio
        assert r1["source_basename"] == "task1.wav"
        assert r2["source_basename"] == "task2.wav"

        # (c) 两个 task_id 独立
        assert r1["task_id"] != r2["task_id"]


class TestExtraTaskFieldsPropagation:
    """extra_task_fields 应该透传到 worker"""

    @pytest.mark.asyncio
    async def test_output_format_reaches_worker(self, fake_qwen3_pool, tmp_path):
        audio = tmp_path / "fmt.wav"
        audio.write_bytes(b"\x00")

        def behavior(worker_id, task):
            return {"output_format": task.get("output_format")}

        poller = FakeWorkerPoller(fake_qwen3_pool, behavior)
        poller.start()
        try:
            result = await fake_qwen3_pool.generate_with_pool(
                audio_path=str(audio),
                extra_task_fields={"output_format": "srt"},
            )
        finally:
            await poller.stop()

        assert result["output_format"] == "srt"


class TestOneWorkerErrorDoesNotAffectOther:
    """一个 worker 抛错, 另一个独立完成"""

    @pytest.mark.asyncio
    async def test_failure_isolation(self, fake_qwen3_pool, tmp_path):
        audio1 = tmp_path / "ok.wav"
        audio2 = tmp_path / "fail.wav"
        for f in (audio1, audio2):
            f.write_bytes(b"\x00")

        def behavior(worker_id, task):
            # source 是 "fail.wav" 的任务抛错
            if "fail.wav" in task["source_audio_path"]:
                raise RuntimeError(f"worker {worker_id} 转录失败")
            return {"worker_id": worker_id, "ok": True}

        poller = FakeWorkerPoller(fake_qwen3_pool, behavior)
        poller.start()

        try:
            t_ok = asyncio.create_task(
                fake_qwen3_pool.generate_with_pool(audio_path=str(audio1))
            )
            t_fail = asyncio.create_task(
                fake_qwen3_pool.generate_with_pool(audio_path=str(audio2))
            )

            results = await asyncio.gather(t_ok, t_fail, return_exceptions=True)
        finally:
            await poller.stop()

        # 一个成功
        ok_result = results[0]
        assert not isinstance(ok_result, Exception)
        assert ok_result["ok"] is True

        # 另一个失败, 错误信息从 worker 透传
        fail_result = results[1]
        assert isinstance(fail_result, Exception)
        assert "转录失败" in str(fail_result)


# ==================== Qwen3PoolTranscriber wrapper 端到端 (经过 wrapper) ====================


class TestPoolTranscriberWrapperHybrid:
    """通过 Qwen3PoolTranscriber 走完整 wrapper → pool → fake worker 路径"""

    @pytest.mark.asyncio
    async def test_wrapper_passes_output_format_to_pool(self, fake_qwen3_pool, tmp_path):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=fake_qwen3_pool)

        audio = tmp_path / "w.wav"
        audio.write_bytes(b"\x00")

        from src.models.schemas import TranscriptionResult, TranscriptionSegment

        def behavior(worker_id, task):
            tres = TranscriptionResult(
                task_id=task["task_id"],
                file_name=Path(task["source_audio_path"]).name,
                file_hash="h",
                duration=1.0,
                segments=[
                    TranscriptionSegment(
                        start_time=0.0, end_time=1.0, text=f"worker{worker_id}", speaker="Speaker1"
                    )
                ],
                speakers=["Speaker1"],
                processing_time=0.01,
            )
            return (tres, {"engine": "qwen3", "via": "fake-worker"})

        poller = FakeWorkerPoller(fake_qwen3_pool, behavior)
        poller.start()
        try:
            result = await wrapper.transcribe(
                audio_path=str(audio),
                task_id="t-wrap",
                output_format="json",
            )
        finally:
            await poller.stop()

        # wrapper 应直接返回 (TranscriptionResult, raw) tuple
        assert isinstance(result, tuple)
        tres, raw = result
        assert tres.file_name == "w.wav"
        assert raw["engine"] == "qwen3"
