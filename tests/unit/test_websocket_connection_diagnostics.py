import asyncio
from contextlib import contextmanager
import hashlib
import io
import json
from unittest.mock import AsyncMock

import pytest
import websockets
from loguru import logger

from src.api.websocket_handler import WebSocketHandler


@contextmanager
def _captured_websocket_logs():
    from src.core.config import config

    stream = io.StringIO()
    sink_id = logger.add(stream, format=config.logging.format, level="DEBUG", colorize=False)
    try:
        yield stream
    finally:
        logger.remove(sink_id)


class _FailingSocket:
    def __init__(self, marker):
        self.send = AsyncMock(side_effect=RuntimeError(marker))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("notification", "stage", "message_type"),
    [
        ("progress", "progress_notification", "task_progress"),
        ("complete", "completion_notification", "task_complete"),
        ("failure", "failure_notification", "error"),
    ],
)
async def test_notification_errors_log_safe_connection_and_task_context(
    notification, stage, message_type,
):
    marker = "SECRET_TRANSCRIPT_OR_CLOSE_REASON"
    handler = WebSocketHandler()
    healthy = AsyncMock()
    handler.connections["conn-safe-17"] = _FailingSocket(marker)
    handler.connections["conn-safe-18"] = healthy
    handler.task_connections["task-safe-29"] = {"conn-safe-17", "conn-safe-18"}

    with _captured_websocket_logs() as stream:
        if notification == "progress":
            await handler.notify_task_progress("task-safe-29", 55.0, "processing")
        elif notification == "complete":
            await handler.notify_task_complete("task-safe-29", {"text": marker})
        else:
            await handler.notify_task_error(
                "task-safe-29", "transcription_error", marker, "failed"
            )

    output = stream.getvalue()
    assert marker not in output
    connection_ref = hashlib.sha256(b"conn-safe-17").hexdigest()[:16]
    assert f"connection_ref={connection_ref}" in output
    assert "task-safe-29" in output
    assert "RuntimeError" in output
    assert stage in output
    assert message_type in output
    assert "conn-safe-17" not in handler.connections
    healthy.send.assert_awaited_once()
    assert "conn-safe-18" in handler.connections


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_message_type"),
    [
        ({"type": "task_status_batch", "data": {"task_ids": []}}, "task_status_batch"),
        ({"type": [], "data": {"task_ids": []}}, "unknown"),
        ({"type": "task_status_batch", "data": {"task_id": []}}, "task_status_batch"),
    ],
)
async def test_message_logs_safe_fields_for_bad_shapes(monkeypatch, payload, expected_message_type):
    from src.core.config import config

    monkeypatch.setattr(config.auth, "enabled", False)
    handler = WebSocketHandler()
    handler._build_capabilities = lambda: {}

    class OneMessageSocket:
        remote_address = ("127.0.0.1", 43210)

        def __init__(self):
            self.messages = iter([json.dumps(payload)])
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise StopAsyncIteration

        async def send(self, payload):
            self.sent.append(json.loads(payload))

    websocket = OneMessageSocket()
    handler._handle_message = AsyncMock(side_effect=RuntimeError("SECRET_MESSAGE_BODY"))

    with _captured_websocket_logs() as stream:
        await handler.handle_connection(websocket, "/")

    output = stream.getvalue()
    assert "SECRET_MESSAGE_BODY" not in output
    assert "RuntimeError" in output
    assert f"message_type={expected_message_type}" in output
    connected = websocket.sent[0]
    connection_id = connected["data"]["connection_id"]
    connection_ref = hashlib.sha256(connection_id.encode("utf-8")).hexdigest()[:16]
    assert f"connection_ref={connection_ref}" in output
    assert websocket.sent[-1]["data"]["message"] != "SECRET_MESSAGE_BODY"


@pytest.mark.asyncio
async def test_remote_close_reason_is_not_logged_and_close_code_is_visible(monkeypatch):
    from src.core.config import config

    monkeypatch.setattr(config.auth, "enabled", False)
    handler = WebSocketHandler()
    handler._build_capabilities = lambda: {}
    cleanups = asyncio.Queue()
    cleanup = handler._cleanup_connection

    def mark_cleaned(connection_id):
        cleanup(connection_id)
        cleanups.put_nowait(connection_id)

    handler._cleanup_connection = mark_cleaned
    server = await websockets.serve(handler.handle_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reason = "SECRET_REMOTE_CLOSE_REASON"
    try:
        with _captured_websocket_logs() as stream:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as client:
                await client.recv()
                await client.close(code=4000, reason=reason)
            await asyncio.wait_for(cleanups.get(), timeout=2)
        output = stream.getvalue()
    finally:
        server.close()
        await server.wait_closed()

    assert reason not in output
    assert "ConnectionClosedError" in output
    assert "close_code=4000" in output
    assert "abnormal_close" in output
    assert "WARNING" in output
