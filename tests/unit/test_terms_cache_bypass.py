"""I1-T2：有效 terms 绕过普通缓存，空 terms 仍保留旧缓存路径。"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.database import cache_allowed_for
from src.core.task_manager import TaskManager
from src.models.schemas import (
    TranscribeOptions,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
)


def _task(tmp_path, *, task_id="terms-cache", output_format="json", terms=None):
    audio = tmp_path / f"{task_id}.wav"
    audio.write_bytes(b"\0" * 10)
    return TranscriptionTask(
        task_id=task_id, file_name=audio.name, file_path=str(audio), file_size=10,
        file_hash=task_id, output_format=output_format,
        options=TranscribeOptions(terms=terms or []),
    )


def _result(task_id, output_format):
    segment = TranscriptionSegment(start_time=0, end_time=1, text="ok", speaker="Speaker1")
    result = TranscriptionResult(
        task_id=task_id, file_name="x.wav", file_hash=task_id, duration=1,
        segments=[segment], speakers=["Speaker1"], processing_time=0.1,
    )
    if output_format == "json":
        return result, {}
    return {
        "segments": [segment], "file_name": "x.wav", "file_hash": task_id,
        "duration": 1, "processing_time": 0.1, "content": "1\ntext",
        "raw_result": [],
    }


def test_cache_allowed_for_terms_is_pure_and_empty_compatible():
    assert cache_allowed_for(TranscribeOptions()) is True
    assert cache_allowed_for(TranscribeOptions(terms=["Alpha"])) is False


@pytest.mark.asyncio
async def test_submit_terms_skips_cache_read(tmp_path):
    manager = TaskManager()
    task = _task(tmp_path, terms=["Alpha"])
    manager.tasks[task.task_id] = task
    with patch("src.core.task_manager.db_manager") as db, \
         patch("src.core.database.cache_params_for") as params:
        db.get_cached_result = AsyncMock(return_value=None)
        await manager.submit_task(task.task_id, task.file_path)
    db.get_cached_result.assert_not_called()
    params.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["json", "srt"])
async def test_worker_terms_skips_second_read_and_result_write(tmp_path, output_format):
    manager = TaskManager()
    task = _task(tmp_path, task_id=f"terms-{output_format}", output_format=output_format, terms=["Alpha"])
    manager.tasks[task.task_id] = task
    transcriber = MagicMock(transcribe=AsyncMock(return_value=_result(task.task_id, output_format)))
    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=transcriber), \
         patch("src.core.task_manager.db_manager") as db, \
         patch("src.core.database.cache_params_for") as params, \
         patch("src.core.database.cache_save_engine_for") as save_tag:
        db.get_cached_result = AsyncMock(return_value=None)
        db.save_result = AsyncMock()
        with patch.object(manager, "_notify_task_progress", new=AsyncMock()), \
             patch.object(manager, "_notify_task_complete", new=AsyncMock()), \
             patch.object(manager, "_maybe_delete_task_file", new=AsyncMock()):
            await manager._process_task(task.task_id)
    transcriber.transcribe.assert_awaited_once()
    db.get_cached_result.assert_not_called()
    db.save_result.assert_not_called()
    params.assert_not_called()
    save_tag.assert_not_called()
