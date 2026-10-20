"""回归：任务被取消或看门狗超时后，worker 不得改写终态或继续重跑。"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import time
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

TERMINAL_KEPT_MARKER = "TASK-TERMINAL-KEPT"


@pytest.fixture
def config_guard():
    """隔离任务队列、重试、看门狗与文件清理配置。"""
    from src.core.config import config

    transcription = config.transcription
    saved = (
        transcription.max_queue_size,
        transcription.max_concurrent_tasks,
        transcription.retry_times,
        transcription.delete_after_transcription,
        transcription.task_max_processing_seconds,
    )
    transcription.max_queue_size = 4
    transcription.max_concurrent_tasks = 1
    transcription.retry_times = 2
    transcription.delete_after_transcription = False
    transcription.task_max_processing_seconds = 3600
    yield transcription
    (
        transcription.max_queue_size,
        transcription.max_concurrent_tasks,
        transcription.retry_times,
        transcription.delete_after_transcription,
        transcription.task_max_processing_seconds,
    ) = saved


class _BlockedTranscriber:
    """转写调用可控阻塞，放行后按用例成功或抛可重试错误。"""

    def __init__(self, error: str | None = None):
        self.error = error
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.calls = 0

    async def transcribe(self, audio_path, task_id, progress_callback, output_format, options):
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        self.finished.set()
        if self.error:
            raise RuntimeError(self.error)
        return _success_result(task_id), {"sentence_info": []}


def _success_result(task_id: str) -> TranscriptionResult:
    """构造 JSON 转写出口需要的最小有效结果。"""
    return TranscriptionResult(
        task_id=task_id,
        file_name=f"{task_id}.wav",
        file_hash=f"hash-{task_id}",
        duration=1.0,
        segments=[
            TranscriptionSegment(
                start_time=0.0,
                end_time=1.0,
                text="你好",
                speaker="Speaker1",
            )
        ],
        speakers=["Speaker1"],
        processing_time=0.1,
    )


async def _submit(mgr: TaskManager, task_id: str, tmp_path) -> None:
    """从任务管理器的创建、提交入口排入真实 worker。"""
    request = FileUploadRequest(
        file_name=f"{task_id}.wav",
        file_size=10,
        file_hash=f"hash-{task_id}",
        force_refresh=True,
    )
    await mgr.create_task(request, task_id=task_id)
    audio_path = tmp_path / f"{task_id}.wav"
    audio_path.write_bytes(b"\x00" * 10)
    await mgr.submit_task(task_id, str(audio_path))


def _patch_runtime(mgr: TaskManager, fake: _BlockedTranscriber):
    """隔离缓存、通知与模型边界，保留真实 TaskManager/worker 流程。"""
    stack = patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake)
    stack.start()
    db_patch = patch("src.core.task_manager.db_manager")
    db = db_patch.start()
    db.get_cached_result = AsyncMock(return_value=None)
    db.save_result = AsyncMock()
    cache_patch = patch("src.core.database.cache_allowed_for", return_value=False)
    cache_patch.start()
    notify_patches = []
    for name in (
        "_notify_task_progress",
        "_notify_task_complete",
        "_notify_task_failed",
        "_send_wework_notification",
    ):
        patcher = patch.object(mgr, name, new=AsyncMock())
        patcher.start()
        notify_patches.append(patcher)
    return [stack, db_patch, cache_patch, *notify_patches]


def _stop_patches(patches) -> None:
    """按建立顺序恢复测试替身。"""
    for patcher in reversed(patches):
        patcher.stop()


async def _wait_processing_done(mgr: TaskManager, timeout: float = 2.0) -> None:
    """等 worker 完成该轮处理，避免只观察到转写器刚返回的中间态。"""
    deadline = time.monotonic() + timeout
    while mgr.processing_tasks and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert mgr.processing_tasks == 0, "worker 未在时限内结束本轮处理"


@pytest.mark.asyncio
async def test_cancelled_during_transcription_stays_cancelled_after_success(config_guard, tmp_path):
    """在途取消后转写成功，仍为 CANCELLED 且 worker 不发完成通知。"""
    mgr = TaskManager()
    fake = _BlockedTranscriber()
    patches = _patch_runtime(mgr, fake)
    records = []
    sink = logger.add(records.append, level="WARNING", format="{message}")
    await mgr.start()
    try:
        await _submit(mgr, "cancelled", tmp_path)
        await asyncio.wait_for(fake.entered.wait(), timeout=2.0)

        assert await mgr.cancel_task("cancelled") is True
        fake.release.set()
        await asyncio.wait_for(fake.finished.wait(), timeout=2.0)
        await _wait_processing_done(mgr)

        task = mgr.get_task("cancelled")
        assert task.status is TaskStatus.CANCELLED
        assert mgr._terminal_counter[TaskStatus.CANCELLED.value] == 1
        assert mgr._terminal_counter[TaskStatus.COMPLETED.value] == 0
        mgr._notify_task_complete.assert_not_awaited()
        assert fake.calls == 1
        assert any(TERMINAL_KEPT_MARKER in str(record) for record in records)
    finally:
        await mgr.stop()
        logger.remove(sink)
        _stop_patches(patches)


@pytest.mark.asyncio
async def test_timed_out_during_transcription_does_not_retry_failure(config_guard, tmp_path):
    """看门狗超时后转写抛可重试错误，不复活、不重跑或重复通知。"""
    config_guard.task_max_processing_seconds = 1
    mgr = TaskManager()
    fake = _BlockedTranscriber(error="temporary engine failure")
    patches = _patch_runtime(mgr, fake)
    records = []
    sink = logger.add(records.append, level="WARNING", format="{message}")
    await mgr.start()
    try:
        await _submit(mgr, "timed-out", tmp_path)
        await asyncio.wait_for(fake.entered.wait(), timeout=2.0)

        task = mgr.get_task("timed-out")
        task.started_at = datetime.now() - timedelta(seconds=2)
        assert mgr._terminalize_stale_processing() == [task]
        assert task.status is TaskStatus.TIMED_OUT
        assert mgr._terminal_counter[TaskStatus.TIMED_OUT.value] == 1
        # 看门狗的通知属于超时终态化；清空后仅核 worker 是否另发失败通知。
        mgr._notify_task_failed.reset_mock()

        fake.release.set()
        await asyncio.wait_for(fake.finished.wait(), timeout=2.0)
        await _wait_processing_done(mgr)

        assert task.status is TaskStatus.TIMED_OUT
        assert task.retry_count == 0
        assert fake.calls == 1
        assert mgr._terminal_counter[TaskStatus.TIMED_OUT.value] == 1
        assert mgr._terminal_counter[TaskStatus.FAILED.value] == 0
        mgr._notify_task_failed.assert_not_awaited()
        mgr._notify_task_complete.assert_not_awaited()
        assert any(TERMINAL_KEPT_MARKER in str(record) for record in records)
    finally:
        await mgr.stop()
        logger.remove(sink)
        _stop_patches(patches)


@pytest.mark.asyncio
async def test_cancel_timed_out_task_returns_false_without_terminal_count_change(config_guard):
    """cancel_task 遇到看门狗终态返回 False，不能重复计数。"""
    mgr = TaskManager()
    task = await mgr.create_task(
        FileUploadRequest(file_name="timed-out.wav", file_size=10, file_hash="timed-out"),
        task_id="timed-out",
    )
    task.status = TaskStatus.PROCESSING
    task.started_at = datetime.now() - timedelta(seconds=2)
    config_guard.task_max_processing_seconds = 1
    assert mgr._terminalize_stale_processing() == [task]
    terminal_counts = dict(mgr._terminal_counter)

    assert await mgr.cancel_task("timed-out") is False
    assert task.status is TaskStatus.TIMED_OUT
    assert dict(mgr._terminal_counter) == terminal_counts


@pytest.mark.asyncio
async def test_cancel_during_cache_lookup_cannot_become_completed(config_guard, tmp_path):
    """缓存查询 await 期间取消，稍后命中缓存也不能覆盖终态。"""
    mgr = TaskManager()
    task = await mgr.create_task(
        FileUploadRequest(file_name="cache.wav", file_size=10, file_hash="cache-hash"),
        task_id="cache-race",
    )
    audio_path = tmp_path / "cache.wav"
    audio_path.write_bytes(b"\x00" * 10)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _cached_result(*args, **kwargs):
        entered.set()
        await release.wait()
        return _success_result(task.task_id)

    notify_complete = AsyncMock()
    with patch("src.core.task_manager.db_manager") as db:
        db.get_cached_result = AsyncMock(side_effect=_cached_result)
        db.save_result = AsyncMock()
        with patch.object(mgr, "_notify_task_complete", new=notify_complete):
            submit = asyncio.create_task(mgr.submit_task(task.task_id, str(audio_path)))
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            assert await mgr.cancel_task(task.task_id) is True
            release.set()
            await asyncio.wait_for(submit, timeout=2.0)

    assert task.status is TaskStatus.CANCELLED
    assert mgr.task_queue.empty()
    assert mgr._terminal_counter[TaskStatus.CANCELLED.value] == 1
    assert mgr._terminal_counter[TaskStatus.COMPLETED.value] == 0
    notify_complete.assert_not_awaited()
