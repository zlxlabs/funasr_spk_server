"""成功结果交付不应被 worker lease 回收失败覆盖。"""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from loguru import logger

from src.core import file_based_process_pool as pool_module
from src.core.file_based_process_pool import FileBasedProcessPool


FAKE_WORKER = Path(__file__).parents[1] / "fixtures/workers/fake_file_pool_worker.py"


@pytest.mark.asyncio
async def test_success_result_survives_lease_reap_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_POOL_MODE", "success")
    pool = FileBasedProcessPool(
        pool_size=1,
        worker_entry_script=str(FAKE_WORKER),
        task_dir=str(tmp_path / "tasks"),
    )
    pool._calculate_timeout = MagicMock(return_value=5)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio-bytes")
    await pool.initialize()
    original_reap = pool._reap_worker_process
    log_messages = []
    sink_id = logger.add(log_messages.append, level="ERROR", format="{message}")

    async def fail_reap(process, force=False):
        raise OSError("injected reap failure")

    pool._reap_worker_process = fail_reap
    try:
        try:
            result = await pool.generate_with_pool(str(audio))
        except OSError as error:
            pytest.fail(f"成功 result 被 lease 回收异常覆盖: {error!r}")

        assert result["source_basename"] == audio.name
        assert result["worker_id"] == 0
        assert result["worker_pid"] > 0
        assert pool._worker_tasks[0].task_id == result["task_id"]
        assert any(
            "POOL-LEASE-RELEASE-FAILED" in message
            and f"task_id={result['task_id']}" in message
            and "worker_id=0" in message
            and "OSError('injected reap failure')" in message
            for message in log_messages
        )
    finally:
        pool._reap_worker_process = original_reap
        await pool.cleanup()
        logger.remove(sink_id)


@pytest.mark.asyncio
async def test_failed_result_still_releases_lease_and_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_POOL_MODE", "success")
    pool = FileBasedProcessPool(
        pool_size=1,
        worker_entry_script=str(FAKE_WORKER),
        task_dir=str(tmp_path / "tasks"),
    )
    pool._calculate_timeout = MagicMock(return_value=5)
    audio = tmp_path / "failed.wav"
    audio.write_bytes(b"audio-bytes")
    await pool.initialize()
    original_load = pickle.load

    def load_failed_result(handle):
        result_data = original_load(handle)
        result_data["success"] = False
        result_data["error"] = "injected inference failure"
        return result_data

    monkeypatch.setattr(pool_module.pickle, "load", load_failed_result)
    try:
        with pytest.raises(Exception, match="处理失败: injected inference failure"):
            await pool.generate_with_pool(str(audio))
        assert pool._worker_tasks == {}
    finally:
        await pool.cleanup()
