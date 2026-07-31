"""
高负载队列机制止血 — Lane A: task_manager 内存清理 + 看门狗 + queue-full 回滚

覆盖（红→绿）：
1. submit_task 队列满 → 抛 QueueFullError 且回滚 self.tasks 插入（codex 窟窿）
2. _evict_terminal_tasks: TTL 清理 + size-cap，非终态永不清
3. _terminalize_stale_processing: 卡死 PROCESSING → TIMED_OUT
4. EMA 处理时长估算 + retry_after

mock db_manager 隔离缓存查询。
"""
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock

import pytest

from src.core.task_manager import TaskManager, QueueFullError
from src.models.schemas import TranscriptionTask, TaskStatus


def make_task(task_id, status=TaskStatus.PENDING, completed_ago=None, started_ago=None):
    """构造一个任务对象，可指定终态/开始时间（用于 TTL/看门狗测试）"""
    t = TranscriptionTask(
        task_id=task_id, file_name="x.wav", file_path="/tmp/x.wav",
        file_size=100, file_hash="h-" + task_id, engine="qwen3",
    )
    t.status = status
    if completed_ago is not None:
        t.completed_at = datetime.now() - timedelta(seconds=completed_ago)
    if started_ago is not None:
        t.started_at = datetime.now() - timedelta(seconds=started_ago)
    return t


@pytest.fixture
def cfg_guard():
    """隔离对 config.transcription 的临时改动"""
    from src.core.config import config
    t = config.transcription
    saved = (
        t.max_queue_size, t.task_retention_ttl_seconds, t.task_max_retained,
        t.task_max_processing_seconds, t.default_engine, t.qwen3_pool_size,
        t.max_concurrent_tasks,
    )
    yield t
    (t.max_queue_size, t.task_retention_ttl_seconds, t.task_max_retained,
     t.task_max_processing_seconds, t.default_engine, t.qwen3_pool_size,
     t.max_concurrent_tasks) = saved


class TestQueueFullRollback:
    """codex 窟窿: create_task 先写 self.tasks，submit_task 才查容量；
    队列满必须回滚插入，否则被拒 PENDING 永久泄漏。"""

    @pytest.mark.asyncio
    async def test_queue_full_raises_queue_full_error(self, cfg_guard, tmp_path):
        cfg_guard.max_queue_size = 2
        mgr = TaskManager()
        # 填满队列
        mgr.task_queue.put_nowait("dummy1")
        mgr.task_queue.put_nowait("dummy2")

        # 真任务已在 self.tasks（模拟 create_task 已插入）
        mgr.tasks["real"] = make_task("real")
        f = tmp_path / "x.wav"; f.write_bytes(b"\x00" * 10)

        with patch("src.core.task_manager.db_manager") as db:
            db.get_cached_result = AsyncMock(return_value=None)
            with pytest.raises(QueueFullError) as exc:
                await mgr.submit_task("real", str(f))
            assert exc.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_queue_full_rolls_back_self_tasks(self, cfg_guard, tmp_path):
        cfg_guard.max_queue_size = 1
        mgr = TaskManager()
        mgr.task_queue.put_nowait("dummy1")
        mgr.tasks["real"] = make_task("real")
        f = tmp_path / "x.wav"; f.write_bytes(b"\x00" * 10)

        with patch("src.core.task_manager.db_manager") as db:
            db.get_cached_result = AsyncMock(return_value=None)
            with pytest.raises(QueueFullError):
                await mgr.submit_task("real", str(f))
        # 关键：被拒任务不得残留在 self.tasks（否则非终态永不清 → 永久泄漏）
        assert "real" not in mgr.tasks

    @pytest.mark.asyncio
    async def test_success_keeps_task_in_self_tasks(self, cfg_guard, tmp_path):
        cfg_guard.max_queue_size = 10
        mgr = TaskManager()
        mgr.tasks["real"] = make_task("real")
        f = tmp_path / "x.wav"; f.write_bytes(b"\x00" * 10)
        with patch("src.core.task_manager.db_manager") as db:
            db.get_cached_result = AsyncMock(return_value=None)
            await mgr.submit_task("real", str(f))
        assert "real" in mgr.tasks
        assert mgr.task_queue.qsize() == 1


class TestEvictTerminalTasks:
    """TTL + size-cap 双保险；非终态永不清"""

    def test_ttl_evicts_old_terminal(self, cfg_guard):
        cfg_guard.task_retention_ttl_seconds = 100
        cfg_guard.task_max_retained = 1000
        mgr = TaskManager()
        mgr.tasks["old"] = make_task("old", TaskStatus.COMPLETED, completed_ago=500)
        mgr.tasks["fresh"] = make_task("fresh", TaskStatus.COMPLETED, completed_ago=10)
        mgr._evict_terminal_tasks()
        assert "old" not in mgr.tasks
        assert "fresh" in mgr.tasks

    def test_ttl_never_evicts_non_terminal(self, cfg_guard):
        cfg_guard.task_retention_ttl_seconds = 100
        cfg_guard.task_max_retained = 1000
        mgr = TaskManager()
        # 即便"年龄"很老，PENDING/PROCESSING 也永不被 TTL 清
        mgr.tasks["pending"] = make_task("pending", TaskStatus.PENDING, completed_ago=99999)
        mgr.tasks["processing"] = make_task("processing", TaskStatus.PROCESSING, started_ago=99999)
        mgr._evict_terminal_tasks()
        assert "pending" in mgr.tasks
        assert "processing" in mgr.tasks

    def test_size_cap_evicts_oldest_terminal(self, cfg_guard):
        cfg_guard.task_retention_ttl_seconds = 999999  # TTL 不触发，只测 size-cap
        cfg_guard.task_max_retained = 2
        mgr = TaskManager()
        for i, ago in enumerate([300, 200, 100]):  # newest = ago 小
            mgr.tasks[f"t{i}"] = make_task(f"t{i}", TaskStatus.COMPLETED, completed_ago=ago)
        mgr._evict_terminal_tasks()
        assert len(mgr.tasks) == 2
        assert "t0" not in mgr.tasks  # 最老被挤
        assert "t2" in mgr.tasks      # 最新保留

    def test_size_cap_preserves_non_terminal_over_cap(self, cfg_guard):
        cfg_guard.task_retention_ttl_seconds = 999999
        cfg_guard.task_max_retained = 1
        mgr = TaskManager()
        mgr.tasks["p1"] = make_task("p1", TaskStatus.PROCESSING, started_ago=50)
        mgr.tasks["p2"] = make_task("p2", TaskStatus.PENDING)
        mgr.tasks["done"] = make_task("done", TaskStatus.COMPLETED, completed_ago=10)
        mgr._evict_terminal_tasks()
        # 非终态保留（哪怕超 cap），只挤终态
        assert "p1" in mgr.tasks
        assert "p2" in mgr.tasks


class TestTerminalizeStaleProcessing:
    """看门狗：卡死 PROCESSING → TIMED_OUT，可被正常回收"""

    def test_stale_processing_becomes_timed_out(self, cfg_guard):
        cfg_guard.task_max_processing_seconds = 100
        mgr = TaskManager()
        mgr.tasks["stuck"] = make_task("stuck", TaskStatus.PROCESSING, started_ago=500)
        timed_out_tasks = mgr._terminalize_stale_processing()
        assert len(timed_out_tasks) == 1
        assert mgr.tasks["stuck"].status == TaskStatus.TIMED_OUT
        assert mgr.tasks["stuck"].completed_at is not None
        assert mgr.tasks["stuck"].error

    def test_fresh_processing_untouched(self, cfg_guard):
        cfg_guard.task_max_processing_seconds = 100
        mgr = TaskManager()
        mgr.tasks["running"] = make_task("running", TaskStatus.PROCESSING, started_ago=10)
        mgr._terminalize_stale_processing()
        assert mgr.tasks["running"].status == TaskStatus.PROCESSING

    def test_non_processing_untouched(self, cfg_guard):
        cfg_guard.task_max_processing_seconds = 100
        mgr = TaskManager()
        mgr.tasks["pending"] = make_task("pending", TaskStatus.PENDING)
        mgr.tasks["done"] = make_task("done", TaskStatus.COMPLETED, completed_ago=500)
        mgr._terminalize_stale_processing()
        assert mgr.tasks["pending"].status == TaskStatus.PENDING
        assert mgr.tasks["done"].status == TaskStatus.COMPLETED

    def test_timed_out_is_evictable(self, cfg_guard):
        """看门狗终态化后，TIMED_OUT 任务能被 TTL 清理（闭环）"""
        cfg_guard.task_max_processing_seconds = 100
        cfg_guard.task_retention_ttl_seconds = 50
        cfg_guard.task_max_retained = 1000
        mgr = TaskManager()
        mgr.tasks["stuck"] = make_task("stuck", TaskStatus.PROCESSING, started_ago=500)
        mgr._terminalize_stale_processing()
        # completed_at = now，刚终态化未到 TTL → 暂留
        mgr._evict_terminal_tasks()
        assert "stuck" in mgr.tasks
        # 把 completed_at 推到 TTL 之外 → 应被清
        mgr.tasks["stuck"].completed_at = datetime.now() - timedelta(seconds=100)
        mgr._evict_terminal_tasks()
        assert "stuck" not in mgr.tasks


class TestProcessingTimeEstimate:
    """EMA 处理时长 + retry_after 估算"""

    def test_cold_start_default_before_any_record(self, cfg_guard):
        mgr = TaskManager()
        est = mgr._estimate_task_seconds()
        assert est > 0  # 冷启动有默认值，不为 0

    def test_ema_moves_toward_recorded(self, cfg_guard):
        mgr = TaskManager()
        cold = mgr._estimate_task_seconds()
        # 记录一个远大于冷启动的处理时长，EMA 应上移
        mgr._record_processing_seconds(cold + 1000)
        assert mgr._estimate_task_seconds() > cold

    def test_retry_after_positive_and_bounded(self, cfg_guard):
        cfg_guard.default_engine = "qwen3"
        cfg_guard.qwen3_pool_size = 1
        mgr = TaskManager()
        ra = mgr._compute_retry_after()
        assert ra >= 1
