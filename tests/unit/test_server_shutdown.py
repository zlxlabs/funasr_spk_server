"""服务停止必须等待任务管理器后清理池，再关闭 websocket。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_server_stop_closes_websocket_before_pool_cleanup():
    import src.main as main

    events = []
    server = main.FunASRServer()
    server.transcriber = MagicMock()
    server.transcriber.cleanup = AsyncMock(side_effect=lambda: events.append("pool_cleanup"))
    server.server = MagicMock()
    server.server.close.side_effect = lambda: events.append("ws_close")
    server.server.wait_closed = AsyncMock(side_effect=lambda: events.append("ws_wait"))
    with patch.object(main.task_manager, "stop", AsyncMock(side_effect=lambda: events.append("task_stop"))), \
         patch.object(main, "send_custom_notification", AsyncMock(side_effect=lambda *args: events.append("notify"))):
        await server.stop()

    assert events == ["ws_close", "task_stop", "pool_cleanup", "ws_wait", "notify"]
    assert server.is_running is False
