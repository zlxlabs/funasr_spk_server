"""真实 WebSocket 服务的连接数上限契约。"""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
import websockets
from loguru import logger

_original_notification_enabled = os.environ.get("FUNASR_NOTIFICATION_ENABLED")
os.environ["FUNASR_NOTIFICATION_ENABLED"] = "false"
import src.main as server_main
if _original_notification_enabled is None:
    os.environ.pop("FUNASR_NOTIFICATION_ENABLED", None)
else:
    os.environ["FUNASR_NOTIFICATION_ENABLED"] = _original_notification_enabled


@pytest.mark.asyncio
async def test_live_server_passes_websocket_path_to_real_handler(monkeypatch):
    """真实 handler 经 websockets.serve 接入时应收到连接欢迎消息。"""
    monkeypatch.setattr(server_main.config.server, "host", "127.0.0.1")
    monkeypatch.setattr(server_main.config.server, "port", 0)
    monkeypatch.setattr(server_main.config.server, "max_connections", 2)
    monkeypatch.setattr(server_main.db_manager, "init_db", _async_noop)
    monkeypatch.setattr(server_main.task_manager, "start", _async_noop)
    monkeypatch.setattr(server_main.task_manager, "stop", _async_noop)
    monkeypatch.setattr(server_main, "send_custom_notification", _async_noop)

    transcriber = SimpleNamespace(initialize=_async_noop, cleanup=_async_noop)
    monkeypatch.setattr(server_main, "resolve_transcriber", lambda _engine: transcriber)

    server = server_main.FunASRServer()
    server._periodic_cleanup = _async_noop
    client = None
    started = False
    try:
        await server.start()
        started = True
        port = server.server.sockets[0].getsockname()[1]
        client = await websockets.connect(f"ws://127.0.0.1:{port}/")

        welcome = await asyncio.wait_for(client.recv(), timeout=1)
        assert json.loads(welcome)["type"] == "connected"
    finally:
        if client is not None:
            await client.close()
        if started:
            await server.stop()


@pytest.mark.asyncio
async def test_live_server_rejects_over_limit_and_releases_slot(monkeypatch):
    """启动真实随机端口服务，验证拒绝码、断连归还名额和可观测日志。"""
    monkeypatch.setattr(server_main.config.server, "host", "127.0.0.1")
    monkeypatch.setattr(server_main.config.server, "port", 0)
    monkeypatch.setattr(server_main.config.server, "max_connections", 2)
    monkeypatch.setattr(server_main.db_manager, "init_db", _async_noop)
    monkeypatch.setattr(server_main.task_manager, "start", _async_noop)
    monkeypatch.setattr(server_main.task_manager, "stop", _async_noop)
    monkeypatch.setattr(server_main, "send_custom_notification", _async_noop)

    transcriber = SimpleNamespace(initialize=_async_noop, cleanup=_async_noop)
    monkeypatch.setattr(server_main, "resolve_transcriber", lambda _engine: transcriber)

    accepted = 0
    third_acceptance = asyncio.Event()

    async def hold_connection(websocket, *_args):
        nonlocal accepted
        accepted += 1
        if accepted >= 3:
            third_acceptance.set()
        await websocket.wait_closed()

    monkeypatch.setattr(server_main.ws_handler, "handle_connection", hold_connection)
    log_messages = []
    sink_id = logger.add(lambda message: log_messages.append(str(message)), format="{message}")
    server = server_main.FunASRServer()

    async def cleanup_noop():
        return None

    server._periodic_cleanup = cleanup_noop
    clients = []
    started = False
    try:
        await server.start()
        started = True
        port = server.server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}/"

        first = await websockets.connect(uri)
        second = await websockets.connect(uri)
        clients.extend((first, second))

        third = await websockets.connect(uri)
        clients.append(third)
        try:
            await asyncio.wait_for(third.recv(), timeout=1)
        except websockets.ConnectionClosed as exc:
            rejection_code = exc.code
        except asyncio.TimeoutError:
            rejection_code = None
        assert rejection_code == 1013

        await first.close()
        replacement = await websockets.connect(uri)
        clients.append(replacement)
        await asyncio.wait_for(third_acceptance.wait(), timeout=3)

        assert any("WS-CONN-LIMIT" in message for message in log_messages)
    finally:
        for client in clients:
            await client.close()
        if started:
            await server.stop()
        logger.remove(sink_id)


async def _async_noop(*_args, **_kwargs):
    """替代启动流程中的数据库、引擎和任务管理器外部工作。"""
    return None
