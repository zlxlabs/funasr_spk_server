"""可重试失败重入有界队列 → worker 死锁（回归锁）

缺陷（src/core/task_manager.py 可重试失败分支，旧实现）::

    await self.task_queue.put(task_id)      # 把任务回灌有界队列

执行这条 put 的 worker 正是该有界队列的消费者。队列满（max_queue_size）时 put 永久阻塞；
两个 worker 各撞一次可重试错误即全部卡在 put 上 → 队列再无人消费 → 服务停转（/health 仍 200）。

修法: 可重试失败在当前 worker 内原地重跑（不回灌有界队列），
重试语义不变（retry_count 上限 / funasr 模型错误重置+延迟 / 状态 / 日志 / 进度通知）。

本文件锁死四件事:
1. 队列满 + 每个 worker 都遇可重试错误 → 5s 内全部任务到终态、worker 不卡死（死锁回归）。
2. 重试次数上限仍生效（retry_times=2 → 共 3 次尝试后终态 FAILED）。
3. 重试走原地重跑路径: 日志出现 TASK-RETRY-INLINE 标记, 且任务 id 从未 put 回队列。
4. funasr MODEL_ERROR 仍触发 _try_reset_model + 延迟（重试语义没被修法削掉）。
"""
from __future__ import annotations

import asyncio
import time
from contextlib import ExitStack, contextmanager
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from src.core.task_manager import TaskManager
from src.models.schemas import (
    FileUploadRequest,
    TaskStatus,
    TranscriptionResult,
    TranscriptionSegment,
)

# 终态集合（与 task_manager._TERMINAL_STATUSES 同义，测试侧独立列一份避免依赖私有常量）
TERMINAL_STATUSES = frozenset({
    TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT,
})
# 死锁回归窗口: 旧实现下任务永远到不了终态, 这里给 5s 判定"卡死"
DEADLOCK_TIMEOUT = 5.0
# 原地重跑的日志标记（必须出现在 src/core/task_manager.py 的失败/降级日志行里）
RETRY_MARKER = "TASK-RETRY-INLINE"

# 原版 asyncio.sleep（monkeypatch 快速 sleep 时用来真正让出控制权, 避免自递归）
_ORIG_SLEEP = asyncio.sleep


@pytest.fixture
def cfg_guard():
    """隔离对 config.transcription 的临时改动（队列容量/并发/重试上限/删文件）。"""
    from src.core.config import config
    t = config.transcription
    saved = (
        t.max_queue_size, t.max_concurrent_tasks, t.retry_times,
        t.delete_after_transcription,
    )
    yield t
    (t.max_queue_size, t.max_concurrent_tasks, t.retry_times,
     t.delete_after_transcription) = saved


class _SpyQueue(asyncio.Queue):
    """记录 put 调用的有界队列（不改变任何行为, 只做观测）。

    用来直接锁死不变式: 可重试失败路径不得对有界队列做 put。
    """

    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)
        self.put_calls: List[str] = []

    async def put(self, item):
        self.put_calls.append(item)
        await super().put(item)


class _FakeTranscriber:
    """最外层模型调用的替身: 按 task_id 决定成功/失败。

    hold=True 时失败前先卡在 gate 上, 让测试有机会先把有界队列填满 ——
    这样"队列满 + worker 撞可重试错误"的死锁条件是确定性的, 不靠 sleep 赌时序。
    """

    def __init__(self, failing: Dict[str, str], hold: bool = False):
        self.failing = failing          # {task_id: 错误信息}
        self.hold = hold
        self.calls: List[str] = []
        self.inflight = 0
        self.both_inflight = asyncio.Event()
        self.gate = asyncio.Event()

    async def transcribe(self, audio_path, task_id, progress_callback, output_format, options):
        self.calls.append(task_id)
        self.inflight += 1
        if self.inflight >= 2:
            self.both_inflight.set()
        if task_id in self.failing:
            if self.hold:
                await self.gate.wait()
            raise RuntimeError(self.failing[task_id])
        return _ok_result(task_id)


def _ok_result(task_id: str):
    """成功转写结果（JSON 出口: (TranscriptionResult, raw_result)）。"""
    result = TranscriptionResult(
        task_id=task_id,
        file_name=f"{task_id}.wav",
        file_hash=f"hash-{task_id}",
        duration=1.0,
        segments=[TranscriptionSegment(start_time=0.0, end_time=1.0, text="你好", speaker="Speaker1")],
        speakers=["Speaker1"],
        processing_time=0.1,
    )
    return result, {"sentence_info": []}


async def _submit(mgr: TaskManager, task_id: str, tmp_path):
    """走公开入口提交任务（create_task + submit_task）, 不直接调内部 helper 喂参。"""
    request = FileUploadRequest(
        file_name=f"{task_id}.wav", file_size=10, file_hash=f"hash-{task_id}",
    )
    await mgr.create_task(request, task_id=task_id)
    audio = tmp_path / f"{task_id}.wav"
    audio.write_bytes(b"\x00" * 10)
    await mgr.submit_task(task_id, str(audio))


@contextmanager
def _patched(mgr: TaskManager, fake, reset=None):
    """只替身最外层模型调用（转写器）; 缓存层与 websocket 通知层隔离。"""
    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake), \
         patch("src.core.task_manager.db_manager") as db:
        db.get_cached_result = AsyncMock(return_value=None)
        db.save_result = AsyncMock()
        with ExitStack() as stack:
            for name in ("_notify_task_progress", "_notify_task_complete",
                         "_notify_task_failed", "_send_wework_notification"):
                stack.enter_context(patch.object(mgr, name, new=AsyncMock()))
            if reset is not None:
                stack.enter_context(patch.object(mgr, "_try_reset_model", new=reset))
            yield


async def _wait_terminal(mgr: TaskManager, task_ids: List[str], timeout: float) -> bool:
    """等待所有任务到终态; 超时返回 False（= 疑似卡死）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(
            (t := mgr.get_task(tid)) is not None and t.status in TERMINAL_STATUSES
            for tid in task_ids
        ):
            return True
        await asyncio.sleep(0.02)
    return False


def _manager(cfg_guard, fake) -> tuple:
    """按当前配置建 TaskManager + 观测用有界队列。"""
    mgr = TaskManager()
    spy = _SpyQueue(maxsize=cfg_guard.max_queue_size)
    mgr.task_queue = spy
    return mgr, spy


@pytest.mark.asyncio
async def test_retryable_failure_with_full_queue_does_not_deadlock_workers(cfg_guard, tmp_path):
    """死锁回归: 有界队列满 + 每个 worker 撞可重试错误 → 5s 内全部任务到终态。

    旧实现: worker 在 `await self.task_queue.put(task_id)` 上永久阻塞（队列满且无人消费）
    → 任务永不变终态 → 本断言超时失败。
    """
    cfg_guard.max_concurrent_tasks = 2
    cfg_guard.max_queue_size = 2
    cfg_guard.retry_times = 2
    cfg_guard.delete_after_transcription = False

    fake = _FakeTranscriber(
        failing={"A": "transient engine failure", "B": "transient engine failure"},
        hold=True,
    )
    mgr, spy = _manager(cfg_guard, fake)

    with _patched(mgr, fake):
        await mgr.start()
        try:
            await _submit(mgr, "A", tmp_path)
            await _submit(mgr, "B", tmp_path)

            # 等两个 worker 都进了转写 → 此刻队列空、两个 worker 都被占用
            await asyncio.wait_for(fake.both_inflight.wait(), timeout=5.0)

            # 填满有界队列 → 旧实现在这里 put(task_id) 永久阻塞
            for i in range(cfg_guard.max_queue_size):
                spy.put_nowait(f"filler-{i}")
            assert spy.qsize() == cfg_guard.max_queue_size

            # 放行失败: 两个 worker 同时进入重试分支
            fake.gate.set()

            reached = await _wait_terminal(mgr, ["A", "B"], timeout=DEADLOCK_TIMEOUT)
            assert reached, (
                "可重试失败后任务未在 "
                f"{DEADLOCK_TIMEOUT}s 内到达终态 —— worker 疑似卡死在有界队列 put 上"
                f"（状态: A={mgr.get_task('A').status}, B={mgr.get_task('B').status}）"
            )

            # 终态是 FAILED（重试次数用尽后真失败）
            assert mgr.get_task("A").status is TaskStatus.FAILED
            assert mgr.get_task("B").status is TaskStatus.FAILED
        finally:
            await mgr.stop()

    # 不变式: 可重试失败路径不得把任务 put 回有界队列
    assert "A" not in spy.put_calls, f"任务 A 被回灌有界队列: {spy.put_calls}"
    assert "B" not in spy.put_calls, f"任务 B 被回灌有界队列: {spy.put_calls}"


@pytest.mark.asyncio
async def test_workers_keep_serving_new_tasks_after_retry_storm(cfg_guard, tmp_path):
    """死锁回归的另一半: 重试风暴过后服务仍能收新任务（worker 没被拖死）。"""
    cfg_guard.max_concurrent_tasks = 2
    cfg_guard.max_queue_size = 2
    cfg_guard.retry_times = 1
    cfg_guard.delete_after_transcription = False

    fake = _FakeTranscriber(failing={"A": "transient engine failure"})
    mgr, spy = _manager(cfg_guard, fake)

    with _patched(mgr, fake):
        await mgr.start()
        try:
            await _submit(mgr, "A", tmp_path)
            assert await _wait_terminal(mgr, ["A"], timeout=DEADLOCK_TIMEOUT)
            assert mgr.get_task("A").status is TaskStatus.FAILED

            # 腾出队列空间后提交新任务 → 应正常完成（说明 worker 仍活着）
            while not spy.empty():
                spy.get_nowait()
            await _submit(mgr, "C", tmp_path)
            assert await _wait_terminal(mgr, ["C"], timeout=DEADLOCK_TIMEOUT), (
                "重试风暴后新任务未在 "
                f"{DEADLOCK_TIMEOUT}s 内到达终态 —— worker 可能已卡死"
            )
            assert mgr.get_task("C").status is TaskStatus.COMPLETED
        finally:
            await mgr.stop()


@pytest.mark.asyncio
async def test_retry_count_limit_still_enforced(cfg_guard, tmp_path):
    """重试次数上限仍生效: retry_times=2 → 共 3 次尝试（1 次首发 + 2 次重试）后终态 FAILED。"""
    cfg_guard.max_concurrent_tasks = 1
    cfg_guard.max_queue_size = 2
    cfg_guard.retry_times = 2
    cfg_guard.delete_after_transcription = False

    fake = _FakeTranscriber(failing={"A": "transient engine failure"})
    mgr, _spy = _manager(cfg_guard, fake)

    with _patched(mgr, fake):
        await mgr.start()
        try:
            await _submit(mgr, "A", tmp_path)
            assert await _wait_terminal(mgr, ["A"], timeout=DEADLOCK_TIMEOUT)
        finally:
            await mgr.stop()

    task = mgr.get_task("A")
    assert task.status is TaskStatus.FAILED
    assert task.retry_count == 2, f"重试次数应等于上限 2, 实际 {task.retry_count}"
    assert fake.calls.count("A") == 3, f"应共尝试 3 次, 实际 {fake.calls.count('A')}"
    assert "transient engine failure" in (task.error or "")


@pytest.mark.asyncio
async def test_retry_is_logged_with_inline_marker(cfg_guard, tmp_path):
    """失败可见性: 原地重跑必须留 TASK-RETRY-INLINE 日志标记（可 grep）。"""
    cfg_guard.max_concurrent_tasks = 1
    cfg_guard.max_queue_size = 2
    cfg_guard.retry_times = 1
    cfg_guard.delete_after_transcription = False

    fake = _FakeTranscriber(failing={"A": "transient engine failure"})
    mgr, _spy = _manager(cfg_guard, fake)

    records: List[str] = []
    sink = logger.add(records.append, level="WARNING", format="{message}")
    try:
        with _patched(mgr, fake):
            await mgr.start()
            try:
                await _submit(mgr, "A", tmp_path)
                assert await _wait_terminal(mgr, ["A"], timeout=DEADLOCK_TIMEOUT)
            finally:
                await mgr.stop()
    finally:
        logger.remove(sink)

    assert any(RETRY_MARKER in str(r) for r in records), (
        f"日志里没找到 {RETRY_MARKER} 标记（实际记录: {records}）"
    )


@pytest.mark.asyncio
async def test_model_error_still_resets_model_and_delays(cfg_guard, tmp_path, monkeypatch):
    """funasr MODEL_ERROR 的重试语义未被修法削掉: 仍 _try_reset_model + 延迟(2s)。"""
    cfg_guard.max_concurrent_tasks = 1
    cfg_guard.max_queue_size = 2
    cfg_guard.retry_times = 1
    cfg_guard.delete_after_transcription = False

    slept: List[float] = []

    async def _fast_sleep(delay, *args, **kwargs):
        slept.append(delay)
        await _ORIG_SLEEP(0)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    fake = _FakeTranscriber(failing={"A": "VAD algorithm error"})
    mgr, _spy = _manager(cfg_guard, fake)
    reset = AsyncMock()

    with _patched(mgr, fake, reset=reset):
        await mgr.start()
        try:
            await _submit(mgr, "A", tmp_path)
            assert await _wait_terminal(mgr, ["A"], timeout=DEADLOCK_TIMEOUT)
        finally:
            await mgr.stop()

    assert reset.await_count == 1, f"MODEL_ERROR 应触发一次模型重置, 实际 {reset.await_count}"
    assert 2 in slept, f"MODEL_ERROR 重试应保留 2s 延迟, 实际 sleep 调用: {slept}"
    assert mgr.get_task("A").status is TaskStatus.FAILED
