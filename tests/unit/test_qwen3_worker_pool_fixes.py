"""Qwen3 worker 与文件进程池审查修复回归测试。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from src.core.file_based_process_pool import FileBasedProcessPool


FAKE_WORKER = Path(__file__).parents[1] / "fixtures/workers/fake_file_pool_worker.py"


def _write_task(task_dir: Path, *, task_id: str, audio_path: Path) -> Path:
    task_file = task_dir / f"worker_0_{task_id}.task"
    task_file.write_text(
        json.dumps({
            "task_id": task_id,
            "audio_path": str(audio_path),
            "source_audio_path": str(audio_path),
            "output_format": "srt",
        }),
        encoding="utf-8",
    )
    return task_file


def test_transcoded_result_hash_is_original_upload_md5(tmp_path):
    """转码只改模型输入；客户端与缓存看到的 hash 仍对应原始上传字节。"""
    from src.core import qwen3_worker_process as worker

    original = tmp_path / "upload.m4a"
    original_bytes = b"original-upload-container"
    original.write_bytes(original_bytes)
    task_file = _write_task(tmp_path, task_id="hash", audio_path=original)
    converted_bytes = b"transcoded-wav-payload"

    def convert(_input, output_path=None):
        Path(output_path).write_bytes(converted_bytes)
        return output_path

    class Transcriber:
        async def transcribe(self, audio_path, **_kwargs):
            assert Path(audio_path).read_bytes() == converted_bytes
            return {
                "format": "srt",
                "content": "",
                "file_name": Path(audio_path).name,
                "file_hash": hashlib.md5(converted_bytes).hexdigest(),
                "duration": 0.0,
                "processing_time": 0.0,
                "raw_result": {},
            }

    with patch("src.utils.file_utils.convert_to_wav", side_effect=convert):
        worker.process_task(0, Transcriber(), str(task_file), str(tmp_path))

    result = pickle.loads((tmp_path / "worker_0_hash.pkl").read_bytes())
    assert result["result"]["file_hash"] == hashlib.md5(original_bytes).hexdigest()


def test_warmup_failure_does_not_publish_ready_and_raises(tmp_path, monkeypatch):
    """预热失败必须让 worker 非零退出路径可见，且 ready 不得发布。"""
    from src.core import qwen3_worker_process as worker

    class BrokenTranscriber:
        async def initialize(self):
            raise RuntimeError("warmup exploded")

    monkeypatch.setattr(worker, "load_qwen3_transcriber", lambda: BrokenTranscriber())
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(RuntimeError, match="warmup exploded"):
        worker.worker_loop(0, str(tmp_path))

    assert not (tmp_path / "worker_0.ready").exists()


@pytest.mark.asyncio
async def test_unencodable_terms_fail_without_killing_worker_or_misreporting_copy(
    tmp_path, monkeypatch
):
    """孤立代理字符是参数错误；拒绝发生在派发前，热 worker 继续存活。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_POOL_MODE", "hang")
    pool = FileBasedProcessPool(
        pool_size=1,
        worker_entry_script=str(FAKE_WORKER),
        task_dir=str(tmp_path / "tasks"),
    )
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    records = []
    sink_id = logger.add(records.append, level="ERROR", format="{message}")
    await pool.initialize()
    process = pool.worker_processes[0]
    try:
        with pytest.raises(ValueError, match="UTF-8") as invalid_terms:
            await pool.generate_with_pool(
                str(audio),
                extra_task_fields={"options": {"terms": ["valid", "\ud800"]}},
            )
        from src.core.task_manager import ErrorKind, classify_error
        assert classify_error(invalid_terms.value) is ErrorKind.NON_RETRYABLE_INPUT
        assert process.poll() is None
        assert any("WORKER-TERMS-INVALID" in message for message in records)
    finally:
        logger.remove(sink_id)
        await pool.cleanup()


def test_worker_exits_when_parent_pid_changes(tmp_path, monkeypatch, capsys):
    """父进程被杀后，worker 发现 PPID 变化、记录原因并退出。"""
    from src.core import qwen3_worker_process as worker

    class ReadyTranscriber:
        async def initialize(self):
            return None

    monkeypatch.setattr(worker, "load_qwen3_transcriber", lambda: ReadyTranscriber())
    parent_pids = iter((4321, 1))
    monkeypatch.setattr(worker.os, "getppid", lambda: next(parent_pids))
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    worker.worker_loop(0, str(tmp_path))

    output = capsys.readouterr().out
    assert "父进程已消失" in output
    assert not (tmp_path / "worker_0.ready").exists()


@pytest.mark.asyncio
async def test_pool_startup_removes_only_run_dirs_with_confirmed_dead_owner(
    tmp_path, monkeypatch
):
    """启动清理只删除带 owner PID 且经 kill(pid, 0) 确认不存在的 run 目录。"""
    monkeypatch.chdir(tmp_path)
    task_root = tmp_path / "tasks"
    dead = task_root / "run-dead"
    live = task_root / "run-live"
    unknown = task_root / "run-unknown"
    for directory in (dead, live, unknown):
        directory.mkdir(parents=True)
    (dead / ".owner_pid").write_text("91234", encoding="ascii")
    (live / ".owner_pid").write_text("91235", encoding="ascii")

    def pid_probe(pid, signal):
        assert signal == 0
        if pid == 91234:
            raise ProcessLookupError(pid)
        return None

    monkeypatch.setattr(os, "kill", pid_probe)
    pool = FileBasedProcessPool(
        pool_size=1, worker_entry_script=str(FAKE_WORKER), task_dir=str(task_root)
    )
    pool._spawn_worker = AsyncMock()
    await pool.initialize()
    try:
        assert not dead.exists()
        assert live.exists()
        assert unknown.exists()
    finally:
        await pool.cleanup()


@pytest.mark.asyncio
async def test_waiting_for_worker_lease_does_not_emit_repeated_info_logs(tmp_path, monkeypatch):
    """占用中的 worker 等待租约时不应每轮刷两条 INFO 状态日志。"""
    monkeypatch.chdir(tmp_path)
    pool = FileBasedProcessPool(
        pool_size=1, worker_entry_script=str(FAKE_WORKER), task_dir=str(tmp_path / "tasks")
    )
    audio = tmp_path / "waiting.wav"
    audio.write_bytes(b"audio")
    process = MagicMock()
    process.poll.return_value = None
    process.pid = 421
    pool.worker_processes = [process]
    pool.is_initialized = True
    pool._worker_tasks[0] = MagicMock()
    records = []
    sink_id = logger.add(
        lambda message: records.append(message.record), level="DEBUG", format="{message}"
    )
    waiting = asyncio.create_task(pool.generate_with_pool(str(audio)))
    try:
        await asyncio.sleep(0.055)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        state_info = [
            record for record in records
            if record["level"].name == "INFO" and "工作进程状态" in record["message"]
        ]
        assert state_info == []
    finally:
        logger.remove(sink_id)
        pool.is_initialized = False
        pool.worker_processes.clear()
        pool._worker_tasks.clear()
