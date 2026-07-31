"""终态失败通知契约：兼容 progress，并补充显式 error。"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.websocket_handler import WebSocketHandler
from src.core.task_manager import ErrorKind, TaskManager
from src.models.schemas import ErrorResponse, TaskStatus, TranscriptionTask


def make_task(task_id="task-1", status=TaskStatus.FAILED):
    """构造失败通知测试任务。"""
    return TranscriptionTask(
        task_id=task_id,
        file_name="audio.wav",
        file_path="/tmp/audio.wav",
        file_size=100,
        file_hash="hash-1",
        engine="qwen3",
        status=status,
        error="模型失败",
    )


@pytest.fixture
def handler():
    return WebSocketHandler()


def make_websocket():
    websocket = MagicMock()
    websocket.send = AsyncMock()
    return websocket


class TestNotifyTaskError:
    @pytest.mark.asyncio
    async def test_sends_explicit_error_to_all_registered_connections(self, handler):
        websocket_a = make_websocket()
        websocket_b = make_websocket()
        handler.connections = {"conn-a": websocket_a, "conn-b": websocket_b}
        handler.task_connections["task-1"] = {"conn-a", "conn-b"}

        with patch.object(handler, "_send_message", new=AsyncMock()) as send_message:
            await handler.notify_task_error(
                "task-1", "timeout", "任务处理超时", "timed_out"
            )

        assert send_message.await_count == 2
        for call in send_message.await_args_list:
            assert call.args[1] == "error"
            payload = call.args[2]
            assert payload["task_id"] == "task-1"
            assert payload["error"] == "timeout"
            assert payload["message"] == "任务处理超时"
            assert payload["details"]["status"] == "timed_out"

    @pytest.mark.asyncio
    async def test_unregistered_task_is_noop(self, handler):
        with patch.object(handler, "_send_message", new=AsyncMock()) as send_message:
            await handler.notify_task_error(
                "missing-task", "engine_error", "失败", "failed"
            )

        send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_connection_failure_does_not_stop_other_connections(self, handler):
        websocket_bad = make_websocket()
        websocket_good = make_websocket()
        handler.connections = {"bad": websocket_bad, "good": websocket_good}
        handler.task_connections["task-1"] = {"bad", "good"}
        sent_websockets = []

        async def send_message(websocket, _msg_type, _payload):
            if websocket is websocket_bad:
                raise RuntimeError("连接已关闭")
            sent_websockets.append(websocket)

        with patch.object(handler, "_send_message", side_effect=send_message):
            await handler.notify_task_error(
                "task-1", "engine_error", "失败", "failed"
            )

        assert sent_websockets == [websocket_good]
        assert "bad" not in handler.connections

    @pytest.mark.asyncio
    async def test_send_error_keeps_connection_level_task_id_empty(self, handler):
        websocket = make_websocket()
        with patch.object(handler, "_send_message", new=AsyncMock()) as send_message:
            await handler._send_error(websocket, "connection_error", "连接错误")

        payload = send_message.await_args.args[2]
        assert ErrorResponse(error="connection_error", message="连接错误").task_id is None
        assert payload.get("task_id") is None


class TestTaskManagerTerminalFailureNotification:
    @pytest.mark.asyncio
    async def test_notifies_progress_then_error_with_kind_and_status(self):
        manager = TaskManager()
        task = make_task(status=TaskStatus.TIMED_OUT)
        events = []
        websocket_handler = MagicMock()

        async def notify_progress(**_kwargs):
            events.append("progress")

        async def notify_error(**kwargs):
            events.append("error")
            assert kwargs["error_type"] == ErrorKind.TIMEOUT.value
            assert kwargs["status"] == TaskStatus.TIMED_OUT.value

        websocket_handler.notify_task_progress = notify_progress
        websocket_handler.notify_task_error = notify_error
        with patch("src.api.websocket_handler.ws_handler", websocket_handler):
            await manager._notify_task_failed(task, kind=ErrorKind.TIMEOUT.value)

        assert events == ["progress", "error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failed_notification", ["progress", "error"])
    async def test_failure_in_one_notification_does_not_skip_the_other(
        self, failed_notification
    ):
        manager = TaskManager()
        task = make_task()
        websocket_handler = MagicMock()
        websocket_handler.notify_task_progress = AsyncMock(
            side_effect=RuntimeError("progress failed")
            if failed_notification == "progress"
            else None
        )
        websocket_handler.notify_task_error = AsyncMock(
            side_effect=RuntimeError("error failed")
            if failed_notification == "error"
            else None
        )

        with patch("src.api.websocket_handler.ws_handler", websocket_handler):
            await manager._notify_task_failed(task, kind=ErrorKind.ENGINE_ERROR.value)

        websocket_handler.notify_task_progress.assert_awaited_once()
        websocket_handler.notify_task_error.assert_awaited_once()


class TestMaintenanceWatchdogNotification:
    @pytest.mark.asyncio
    async def test_watchdog_notifies_after_releasing_queue_lock(self, monkeypatch):
        from src.core.config import config

        old_timeout = config.transcription.task_max_processing_seconds
        config.transcription.task_max_processing_seconds = 100
        try:
            manager = TaskManager()
            manager.tasks["stuck"] = make_task(
                task_id="stuck", status=TaskStatus.PROCESSING
            )
            manager.tasks["stuck"].started_at = datetime.now() - timedelta(seconds=500)
            manager.is_running = True
            lock_was_available = False
            notification_kinds = []

            async def notify_task_failed(task, *, kind=None):
                nonlocal lock_was_available
                notification_kinds.append(kind)
                manager.is_running = False
                try:
                    await asyncio.wait_for(manager._queue_lock.acquire(), timeout=0.1)
                except asyncio.TimeoutError:
                    return
                lock_was_available = True
                manager._queue_lock.release()

            manager._notify_task_failed = notify_task_failed
            def stop_after_locked_maintenance():
                # 让旧实现（还没有锁外通知）也能结束单轮测试；新实现会在
                # 此后继续消费 timed_out_tasks 并完成通知。
                manager.is_running = False

            manager._evict_terminal_tasks = MagicMock(
                side_effect=stop_after_locked_maintenance
            )
            manager._sweep_orphan_upload_files = AsyncMock()
            monkeypatch.setattr(
                "src.core.task_manager.asyncio.sleep", AsyncMock()
            )

            await manager._maintenance_loop()

            assert manager.tasks["stuck"].status == TaskStatus.TIMED_OUT
            assert notification_kinds == [ErrorKind.TIMEOUT.value]
            assert lock_was_available is True
        finally:
            config.transcription.task_max_processing_seconds = old_timeout
