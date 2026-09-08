"""FileBasedProcessPool 生命周期、归属和真实 subprocess 协议测试。"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.file_based_process_pool import FileBasedProcessPool


FAKE_WORKER = Path(__file__).parents[1] / "fixtures/workers/fake_file_pool_worker.py"


def _pool(tmp_path: Path, monkeypatch, mode: str = "success", **env) -> FileBasedProcessPool:
    monkeypatch.chdir(tmp_path)
    for key, value in {"FAKE_POOL_MODE": mode, **env}.items():
        monkeypatch.setenv(key, value)
    return FileBasedProcessPool(
        pool_size=1,
        worker_entry_script=str(FAKE_WORKER),
        task_dir=str(tmp_path / "tasks"),
    )


@pytest.mark.asyncio
async def test_real_subprocess_receives_argv_and_complete_task_bytes(tmp_path, monkeypatch):
    capture = tmp_path / "capture.json"
    pool = _pool(tmp_path, monkeypatch, FAKE_POOL_CAPTURE=str(capture))
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio-bytes")
    await pool.initialize()
    run_task_dir = str(pool.task_dir)
    try:
        result = await pool.generate_with_pool(
            str(audio), extra_task_fields={"output_format": "json", "marker": "m"}
        )
    finally:
        await pool.cleanup()

    captured = json.loads(capture.read_text(encoding="utf-8"))
    assert captured["argv"][0] == str(FAKE_WORKER)
    assert captured["argv"][1:] == ["--worker-id", "0", "--task-dir", run_task_dir]
    task_bytes = bytes.fromhex(captured["task_bytes"])
    task = json.loads(task_bytes)
    assert task["task_id"] == result["task_id"]
    assert task["output_format"] == "json"
    assert task["marker"] == "m"
    assert result["worker_pid"] == captured["pid"]


@pytest.mark.asyncio
async def test_out_of_order_completion_assigns_new_task_to_free_slot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_POOL_DELAYS", json.dumps({"a.wav": 0.35, "b.wav": 0.05, "c.wav": 0.35, "d.wav": 0.01}))
    pool = FileBasedProcessPool(
        pool_size=3, worker_entry_script=str(FAKE_WORKER), task_dir=str(tmp_path / "tasks")
    )
    await pool.initialize()
    audio_files = []
    try:
        for name in ("a.wav", "b.wav", "c.wav", "d.wav"):
            audio = tmp_path / name
            audio.write_bytes(name.encode())
            audio_files.append(audio)
        tasks = [asyncio.create_task(pool.generate_with_pool(str(path))) for path in audio_files[:3]]
        await asyncio.sleep(0.12)
        task_d = asyncio.create_task(pool.generate_with_pool(str(audio_files[3])))
        results = await asyncio.gather(*tasks, task_d)
    finally:
        await pool.cleanup()

    by_source = {item["source_basename"]: item for item in results}
    assert {by_source[name]["worker_id"] for name in ("a.wav", "b.wav", "c.wav")} == {0, 1, 2}
    assert by_source["d.wav"]["worker_id"] == 1


@pytest.mark.asyncio
async def test_success_replenishes_workers_for_next_batch(tmp_path, monkeypatch):
    """正常完成后立即补齐 worker，下一批任务无需重新串行预热。"""
    monkeypatch.chdir(tmp_path)
    pool = FileBasedProcessPool(
        pool_size=3, worker_entry_script=str(FAKE_WORKER), task_dir=str(tmp_path / "tasks")
    )
    audio_files = []
    await pool.initialize()
    initial_pids = [process.pid for process in pool.worker_processes]
    try:
        for name in ("first-a.wav", "first-b.wav", "first-c.wav"):
            audio = tmp_path / name
            audio.write_bytes(name.encode())
            audio_files.append(audio)

        results = await asyncio.gather(
            *(pool.generate_with_pool(str(audio)) for audio in audio_files)
        )

        assert {item["worker_id"] for item in results} == {0, 1, 2}
        for _ in range(100):
            if all(process.poll() is None for process in pool.worker_processes):
                break
            await asyncio.sleep(0.01)
        replacement_pids = [process.pid for process in pool.worker_processes]
        assert all(process.poll() is None for process in pool.worker_processes)
        assert set(replacement_pids).isdisjoint(initial_pids)
    finally:
        await pool.cleanup()


@pytest.mark.asyncio
async def test_success_delivery_does_not_wait_for_replenish_ready(tmp_path, monkeypatch):
    """正常结果先交付，后台补齐的 ready 等待不能卡住调用。"""
    pool = _pool(tmp_path, monkeypatch)
    audio = tmp_path / "fast.wav"
    audio.write_bytes(b"fast")
    await pool.initialize()
    replenish_started = asyncio.Event()
    release_replenish = asyncio.Event()

    async def delayed_replenish(worker_id):
        replenish_started.set()
        await release_replenish.wait()

    pool._spawn_worker = delayed_replenish
    pool._calculate_timeout = MagicMock(return_value=2)
    try:
        started_at = time.monotonic()
        result = await pool.generate_with_pool(str(audio))
        elapsed = time.monotonic() - started_at
        assert result["source_basename"] == audio.name
        assert elapsed < 1.0
        await asyncio.wait_for(replenish_started.wait(), timeout=1.0)
    finally:
        await pool.cleanup()


@pytest.mark.asyncio
async def test_missing_workers_are_replenished_in_parallel(tmp_path, monkeypatch):
    """巡检补齐多个 dead slot 时并行等待 ready。"""
    monkeypatch.chdir(tmp_path)
    pool = FileBasedProcessPool(pool_size=3, task_dir=str(tmp_path / "tasks"))
    pool.is_initialized = True
    pool.worker_processes = []
    for _ in range(pool.pool_size):
        process = MagicMock()
        process.poll.return_value = 17
        pool.worker_processes.append(process)

    active = 0
    peak_active = 0

    async def spawn(worker_id):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.02)
        active -= 1

    pool._spawn_worker = spawn
    await pool._ensure_workers_alive()

    assert peak_active == pool.pool_size


@pytest.mark.asyncio
async def test_health_replenishes_dead_slot_without_touching_busy_worker(tmp_path, monkeypatch):
    """health 只补无 lease 的 dead slot，不抢占正在处理的 worker。"""
    monkeypatch.chdir(tmp_path)
    pool = FileBasedProcessPool(pool_size=2, task_dir=str(tmp_path / "tasks"))
    pool.is_initialized = True
    busy_process = MagicMock()
    busy_process.poll.return_value = None
    busy_process.pid = 1001
    dead_process = MagicMock()
    dead_process.poll.return_value = 17
    dead_process.pid = 1002
    pool.worker_processes = [busy_process, dead_process]
    pool._worker_tasks[0] = MagicMock()
    pool._spawn_worker = AsyncMock()

    await pool._ensure_workers_alive()

    pool._spawn_worker.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_ready_worker_is_not_blocked_by_missing_slot(tmp_path, monkeypatch):
    """已有 ready slot 时，获取任务不等待另一个 dead slot 的补齐。"""
    monkeypatch.chdir(tmp_path)
    pool = FileBasedProcessPool(pool_size=2, task_dir=str(tmp_path / "tasks"))
    pool.is_initialized = True
    dead_process = MagicMock()
    dead_process.poll.return_value = 17
    ready_process = MagicMock()
    ready_process.poll.return_value = None
    ready_process.pid = 4321
    pool.worker_processes = [dead_process, ready_process]
    pool._ensure_workers_alive = AsyncMock(side_effect=AssertionError("must not wait"))

    lease = await pool._acquire_worker_lease(
        "task-ready",
        pool.task_dir / "worker_1_task-ready.task",
        pool.task_dir / "worker_1_task-ready.pkl",
        pool.task_dir / "task-ready.wav",
    )

    assert lease.worker_id == 1
    pool._worker_tasks.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "cancel", "crash"])
async def test_failed_task_slot_recovers_for_next_request(tmp_path, monkeypatch, failure):
    """timeout/cancel/crash 后下一请求可重新占用同一 slot。"""
    pool = _pool(tmp_path, monkeypatch, mode="hang" if failure != "crash" else "crash")
    first_audio = tmp_path / f"{failure}-first.wav"
    first_audio.write_bytes(b"first")
    await pool.initialize()
    run_task_dir = pool.task_dir

    try:
        if failure == "timeout":
            pool._calculate_timeout = MagicMock(return_value=0.15)
            with pytest.raises(TimeoutError):
                await pool.generate_with_pool(str(first_audio))
        elif failure == "cancel":
            task = asyncio.create_task(pool.generate_with_pool(str(first_audio)))
            for _ in range(100):
                if list(run_task_dir.glob("*.task")):
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            pool._calculate_timeout = MagicMock(return_value=2)
            with pytest.raises(RuntimeError, match="worker 0 exited"):
                await pool.generate_with_pool(str(first_audio))

        monkeypatch.setenv("FAKE_POOL_MODE", "success")
        next_audio = tmp_path / f"{failure}-next.wav"
        next_audio.write_bytes(b"next")
        result = await pool.generate_with_pool(str(next_audio))
        assert result["source_basename"] == next_audio.name
    finally:
        await pool.cleanup()

    assert not run_task_dir.exists()


@pytest.mark.asyncio
async def test_timeout_reaps_process_before_cleaning_owned_files(tmp_path, monkeypatch):
    pool = _pool(tmp_path, monkeypatch, mode="hang")
    pool._calculate_timeout = MagicMock(return_value=0.15)
    keep = tmp_path / "tasks" / "unrelated.sentinel"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_bytes(b"keep")
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"long")
    await pool.initialize()
    run_task_dir = pool.task_dir
    process = pool.worker_processes[0]
    with pytest.raises(TimeoutError):
        await pool.generate_with_pool(str(audio))
    assert process.poll() is not None
    assert keep.read_bytes() == b"keep"
    assert list(run_task_dir.glob("*.task")) == []
    assert list(run_task_dir.glob("*.pkl")) == []
    await pool.cleanup()


@pytest.mark.asyncio
async def test_cancel_reaps_process_and_cleans_owned_files(tmp_path, monkeypatch):
    pool = _pool(tmp_path, monkeypatch, mode="hang")
    audio = tmp_path / "cancel.wav"
    audio.write_bytes(b"cancel")
    await pool.initialize()
    run_task_dir = pool.task_dir
    process = pool.worker_processes[0]
    task = asyncio.create_task(pool.generate_with_pool(str(audio)))
    for _ in range(100):
        if list(run_task_dir.glob("*.task")):
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.poll() is not None
    assert list(run_task_dir.glob("worker_0_*")) == []
    await pool.cleanup()


@pytest.mark.asyncio
async def test_worker_crash_is_reported_without_pool_retry_or_deadline_reset(tmp_path, monkeypatch):
    pool = _pool(tmp_path, monkeypatch, mode="crash")
    pool._calculate_timeout = MagicMock(return_value=2)
    audio = tmp_path / "crash.wav"
    audio.write_bytes(b"crash")
    await pool.initialize()
    run_task_dir = pool.task_dir
    process = pool.worker_processes[0]
    with pytest.raises(RuntimeError, match="worker 0 exited"):
        await pool.generate_with_pool(str(audio))
    assert process.poll() is not None
    assert list(run_task_dir.glob("worker_0_*")) == []
    await pool.cleanup()
    assert not run_task_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["wrong_task", "wrong_pid"])
async def test_result_owner_mismatch_is_rejected(tmp_path, monkeypatch, mode):
    pool = _pool(tmp_path, monkeypatch, mode=mode)
    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"bad")
    await pool.initialize()
    run_task_dir = pool.task_dir
    with pytest.raises(RuntimeError, match="result ownership mismatch"):
        await pool.generate_with_pool(str(audio))
    assert list(run_task_dir.glob("worker_0_*")) == []
    await pool.cleanup()
    assert not run_task_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_waits_after_kill_when_terminate_is_ineffective(tmp_path, monkeypatch):
    pool = _pool(tmp_path, monkeypatch)
    process = MagicMock()
    process.pid = 123
    process.poll.side_effect = [None, None]
    process.wait.side_effect = [subprocess.TimeoutExpired("fake", 1), None]
    pool.worker_processes = [process]
    pool.is_initialized = True
    await pool.cleanup()
    assert process.terminate.call_count == 1
    assert process.kill.call_count == 1
    assert process.wait.call_count == 2


@pytest.mark.asyncio
async def test_cleanup_reaps_three_workers_in_parallel_and_removes_run_dir(tmp_path, monkeypatch):
    """shutdown 并行回收真实 worker，并删除本池 run 目录而保留根目录文件。"""
    monkeypatch.chdir(tmp_path)
    pool = FileBasedProcessPool(
        pool_size=3, worker_entry_script=str(FAKE_WORKER), task_dir=str(tmp_path / "tasks")
    )
    sentinel = tmp_path / "tasks" / "other-pool.sentinel"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_bytes(b"keep")
    await pool.initialize()
    run_task_dir = pool.task_dir

    active = 0
    peak_active = 0
    original_reap = pool._reap_worker_process

    async def tracked_reap(process, force=False):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        try:
            await asyncio.sleep(0.05)
            return await original_reap(process, force=force)
        finally:
            active -= 1

    pool._reap_worker_process = tracked_reap
    await pool.cleanup()

    assert peak_active == 3
    assert not run_task_dir.exists()
    assert sentinel.read_bytes() == b"keep"


def test_darwin_worker_environment_owns_private_cache_lifecycle(tmp_path, monkeypatch):
    """FunASR worker 使用 Darwin 数值 confstr 常量，并只回收自己的缓存根。"""
    import src.core.file_based_process_pool as pool_module

    base_dirs = {
        65536: tmp_path / "user",
        65537: tmp_path / "temp",
        65538: tmp_path / "cache",
    }
    for directory in base_dirs.values():
        directory.mkdir()
    monkeypatch.setattr(pool_module.sys, "platform", "darwin")
    monkeypatch.setattr(pool_module.os, "confstr", lambda name: str(base_dirs[name]))
    pool = pool_module.FileBasedProcessPool(
        pool_size=1,
        worker_entry_script="src/core/worker_process.py",
        task_dir=str(tmp_path / "tasks"),
    )

    environment, private_dirs = pool._prepare_worker_environment(0)

    assert environment["DIRHELPER_USER_DIR_SUFFIX"].startswith("funasr-worker-")
    assert tuple(path.parent for path in private_dirs) == tuple(base_dirs.values())
    graph_cache = private_dirs[1] / "com.apple.MetalPerformanceShadersGraph"
    graph_cache.mkdir()
    (graph_cache / "compiled.bin").write_bytes(b"cache")
    pool._remove_owned_paths(private_dirs)
    assert all(not path.exists() for path in private_dirs)


@pytest.mark.asyncio
async def test_spawn_failure_cleans_run_directory(tmp_path, monkeypatch):
    """worker 启动失败后不留下本次 run 目录或协议文件。"""
    monkeypatch.chdir(tmp_path)
    pool = FileBasedProcessPool(
        pool_size=1,
        worker_entry_script=str(tmp_path / "missing-worker.py"),
        task_dir=str(tmp_path / "tasks"),
    )

    with pytest.raises(RuntimeError):
        await pool.initialize()

    assert list((tmp_path / "tasks").glob("run-*")) == []


@pytest.mark.asyncio
async def test_exit_after_publish_rechecks_result_before_reporting_crash(tmp_path, monkeypatch):
    """poll 先看到退出时，仍复查 worker 已发布的结果文件。"""
    monkeypatch.chdir(tmp_path)
    pool = FileBasedProcessPool(pool_size=1, task_dir=str(tmp_path / "tasks"))
    pool.is_initialized = True
    process = MagicMock()
    process.pid = 4321
    poll_count = 0

    def poll_process():
        nonlocal poll_count
        poll_count += 1
        task_files = list(pool.task_dir.glob("worker_0_*.task"))
        if poll_count >= 2 and task_files:
            task_file = task_files[0]
            task = json.loads(task_file.read_text(encoding="utf-8"))
            result_file = task_file.with_suffix(".pkl")
            result_file.write_bytes(
                pickle.dumps({
                    "task_id": task["task_id"], "success": True,
                    "result": "published", "worker_pid": process.pid,
                })
            )
            task_file.unlink()
            return 0
        return None

    process.poll.side_effect = poll_process
    process.wait.return_value = 0
    pool.worker_processes = [process]
    pool._calculate_timeout = MagicMock(return_value=2)
    pool._schedule_worker_replenish = MagicMock()
    audio = tmp_path / "race.wav"
    audio.write_bytes(b"race")

    assert await pool.generate_with_pool(str(audio)) == "published"
    await pool.cleanup()


@pytest.mark.asyncio
async def test_generate_does_not_revive_pool_after_explicit_cleanup(tmp_path, monkeypatch):
    """显式 cleanup 后，generate 不得自动重新初始化进程池。"""
    pool = _pool(tmp_path, monkeypatch)
    await pool.cleanup()
    audio = tmp_path / "after-cleanup.wav"
    audio.write_bytes(b"audio")

    with pytest.raises(RuntimeError, match="请先显式 initialize"):
        await pool.generate_with_pool(str(audio))

    await pool.initialize()
    assert all(process.poll() is None for process in pool.worker_processes)
    await pool.cleanup()


@pytest.mark.asyncio
async def test_cleanup_waits_for_generate_release_before_deleting_run_dir(tmp_path, monkeypatch):
    """cleanup 在 generate 的真实 release/reap 窗口必须等待旧 worker。"""
    pool = _pool(tmp_path, monkeypatch)
    audio = tmp_path / "release-window.wav"
    audio.write_bytes(b"audio")
    await pool.initialize()
    run_task_dir = pool.task_dir
    reap_started = asyncio.Event()
    allow_reap = asyncio.Event()
    original_reap = pool._reap_worker_process

    async def blocked_reap(process, force=False):
        reap_started.set()
        await allow_reap.wait()
        return await original_reap(process, force=force)

    pool._reap_worker_process = blocked_reap
    generate_task = asyncio.create_task(pool.generate_with_pool(str(audio)))
    await asyncio.wait_for(reap_started.wait(), timeout=2)
    cleanup_task = asyncio.create_task(pool.cleanup())
    await asyncio.sleep(0.05)
    assert not cleanup_task.done()
    assert run_task_dir.exists()

    allow_reap.set()
    assert (await generate_task)["source_basename"] == audio.name
    await cleanup_task
    assert not run_task_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_reap_failure_is_visible_and_preserves_run_dir(tmp_path, monkeypatch):
    """reap 失败时 cleanup 必须 fail loud，不能删除活 worker 的资源。"""
    pool = _pool(tmp_path, monkeypatch)
    audio = tmp_path / "reap-failure.wav"
    audio.write_bytes(b"audio")
    await pool.initialize()
    run_task_dir = pool.task_dir
    process = pool.worker_processes[0]
    original_reap = pool._reap_worker_process

    async def failed_reap(process, force=False):
        raise RuntimeError("injected reap failure")

    pool._reap_worker_process = failed_reap
    with pytest.raises(RuntimeError, match="injected reap failure"):
        await pool.cleanup()
    assert run_task_dir.exists()
    assert process.poll() is None

    pool._reap_worker_process = original_reap
    await pool.cleanup()
    assert not run_task_dir.exists()
