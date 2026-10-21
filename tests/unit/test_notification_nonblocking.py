"""完成通知不得阻塞 worker 消费后续任务的回归测试。"""
from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import config
from src.core.task_manager import TaskManager
from src.models.schemas import (
    FileUploadRequest,
    TranscriptionResult,
    TranscriptionSegment,
)


class _ImmediateTranscriber:
    """替身模型：立即完成转写，并记录第二个任务何时开工。"""

    def __init__(self):
        self.second_started = asyncio.Event()

    async def transcribe(self, audio_path, task_id, progress_callback, output_format, options):
        if task_id == "second":
            self.second_started.set()
        result = TranscriptionResult(
            task_id=task_id,
            file_name=f"{task_id}.wav",
            file_hash=f"hash-{task_id}",
            duration=1.0,
            segments=[
                TranscriptionSegment(
                    start_time=0.0,
                    end_time=1.0,
                    text="测试",
                    speaker="Speaker1",
                )
            ],
            speakers=["Speaker1"],
            processing_time=0.01,
        )
        return result, {"sentence_info": []}


async def _submit(manager: TaskManager, task_id: str, tmp_path):
    """通过 create_task/submit_task 的公开入口将任务交给 worker。"""
    request = FileUploadRequest(
        file_name=f"{task_id}.wav",
        file_size=10,
        file_hash=f"hash-{task_id}",
    )
    await manager.create_task(request, task_id=task_id)
    audio_path = tmp_path / f"{task_id}.wav"
    audio_path.write_bytes(b"audio-data")
    await manager.submit_task(task_id, str(audio_path))


@pytest.mark.asyncio
async def test_completed_webhook_does_not_block_worker_and_is_eventually_sent(
    monkeypatch, tmp_path
):
    """两秒 webhook 在途期间，单 worker 必须已开始处理队列中的下一个任务。"""
    transcription = config.transcription
    old_concurrency = transcription.max_concurrent_tasks
    old_queue_size = transcription.max_queue_size
    old_delete_after = transcription.delete_after_transcription
    notification = config.notification
    old_enabled = notification.enabled
    old_webhook_url = notification.webhook_url
    transcription.max_concurrent_tasks = 1
    transcription.max_queue_size = 4
    transcription.delete_after_transcription = False
    monkeypatch.setattr(notification, "enabled", True)
    monkeypatch.setattr(notification, "webhook_url", "https://example.invalid/webhook")

    transcriber = _ImmediateTranscriber()
    webhook_started = asyncio.Event()
    both_notifications_finished = asyncio.Event()
    sent_events: list[str] = []

    async def slow_webhook(task, event_type):
        webhook_started.set()
        await asyncio.sleep(2)
        sent_events.append(task.task_id)
        if len(sent_events) == 2:
            both_notifications_finished.set()

    manager = TaskManager()
    fake_ws_handler = SimpleNamespace(
        notify_task_progress=AsyncMock(),
        notify_task_complete=AsyncMock(),
    )
    fake_websocket_module = SimpleNamespace(ws_handler=fake_ws_handler)
    try:
        with patch.dict(
            sys.modules,
            {"src.api.websocket_handler": fake_websocket_module},
        ), patch(
            "src.core.transcriber_dispatch.resolve_transcriber",
            return_value=transcriber,
        ), patch("src.core.task_manager.db_manager") as database, patch(
            "src.utils.notification.send_wework_notification",
            new=slow_webhook,
        ):
            database.get_cached_result = AsyncMock(return_value=None)
            database.save_result = AsyncMock()
            await manager.start()
            await _submit(manager, "first", tmp_path)
            await _submit(manager, "second", tmp_path)

            await asyncio.wait_for(webhook_started.wait(), timeout=1.0)
            started_at = time.monotonic()
            try:
                await asyncio.wait_for(transcriber.second_started.wait(), timeout=0.5)
                second_started_without_waiting = True
            except asyncio.TimeoutError:
                second_started_without_waiting = False
            elapsed = time.monotonic() - started_at
            assert second_started_without_waiting, (
                "第二个任务直到 webhook 的两秒 sleep 结束后才开始，"
                f"worker 被同步通知阻塞（等待 {elapsed:.3f}s）"
            )
            assert elapsed < 1.0

            await asyncio.wait_for(both_notifications_finished.wait(), timeout=3.0)
            assert sent_events == ["first", "second"]
    finally:
        await manager.stop()
        transcription.max_concurrent_tasks = old_concurrency
        transcription.max_queue_size = old_queue_size
        transcription.delete_after_transcription = old_delete_after
        notification.enabled = old_enabled
        notification.webhook_url = old_webhook_url
