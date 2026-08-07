"""可观测性 (P1) — 真 websockets.serve + process_request 活体测试 (codex #17/#4).

不 mock websockets: 起真 server 在 ephemeral 端口, httpx GET /health+/metrics,
再开真 ws 连接验升级仍通. 这是"process_request 在 venv 实装 websockets 下确实
能拦 HTTP 路径并返 body"的唯一可信证明 (单元 mock 证不了).

#4 版本钉死: 断言实装 websockets 是 12.x (legacy API); bump 到 13+ 直接红,
逼人重写 process_request (新 asyncio API 签名不同), 而非运行时静默坏.
"""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock

import httpx
import pytest
import websockets

from src.api.http_endpoints import HttpEndpoints


def test_websockets_version_is_12x():
    """#4: 端点实现钉在 websockets 12.0 legacy process_request 签名上。

    bump 到 13+ 这套 API 变 → 本断言红 → 强制重写 http_endpoints.process_request。
    """
    assert websockets.__version__.startswith("12."), (
        f"websockets={websockets.__version__}: http_endpoints 假设 12.0 legacy "
        f"process_request(path, request_headers) 签名; 13+ 是新 asyncio API, 需重写。"
    )


def _make_real_endpoints(host="127.0.0.1", metrics_token=None):
    tm = MagicMock()
    tm.is_running = True
    tm._maintenance_task = MagicMock()
    tm._maintenance_task.done.return_value = False
    tm.get_metrics_snapshot.return_value = {
        "queue_size": 0, "max_queue_size": 20, "pending": 0, "processing": 0,
        "tasks_in_memory": 0, "terminal_total": {"completed": 1}, "errors_total": {},
        "task_seconds_ema": 90.0, "engine": "funasr", "pool_size": 2,
    }
    db = MagicMock()
    db.get_cache_stats = AsyncMock(return_value={"total_cached": 0, "hits": 0, "misses": 0, "projected_serves": 0})
    cfg = MagicMock()
    cfg.observability.metrics_enabled = True
    cfg.observability.metrics_token = metrics_token
    cfg.server.host = host
    cfg.transcription.default_engine = "funasr"
    return HttpEndpoints(task_manager=tm, db_manager=db, config=cfg)


async def _dummy_ws(websocket, *args):
    """最小 ws handler (兼容 12.0 单/双参签名)。"""
    await websocket.send("hi")


@pytest.mark.asyncio
async def test_live_health_metrics_and_ws_upgrade():
    ep = _make_real_endpoints(host="127.0.0.1")
    server = await websockets.serve(
        _dummy_ws, "127.0.0.1", 0, process_request=ep.process_request
    )
    port = server.sockets[0].getsockname()[1]
    try:
        async with httpx.AsyncClient() as client:
            # /health → 200 JSON healthy
            r = await client.get(f"http://127.0.0.1:{port}/health")
            assert r.status_code == 200
            assert r.json()["status"] == "healthy"
            capabilities = await client.get(f"http://127.0.0.1:{port}/capabilities?probe=live")
            assert capabilities.status_code == 200
            assert capabilities.json()["features"] == {"terms": True}

            # /metrics → 200 Prometheus text
            r2 = await client.get(f"http://127.0.0.1:{port}/metrics")
            assert r2.status_code == 200
            assert "funasr_queue_size" in r2.text
            assert 'funasr_tasks_terminal_total{status="completed"} 1' in r2.text

            # / → 200 HTML 状态页
            r3 = await client.get(f"http://127.0.0.1:{port}/")
            assert r3.status_code == 200
            assert "text/html" in r3.headers.get("content-type", "")
            assert "FunASR 服务状态" in r3.text

        # 真 ws 升级仍通 — /ws 与**根路径 /** 都要通 (根路径回归: 生产事故 2026-06-17,
        # / 被 HTML 状态页劫持导致客户端无法连接; 现 Upgrade 头放行修复)
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            assert await ws.recv() == "hi"
        async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
            assert await ws.recv() == "hi"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_live_metrics_denied_without_token_on_wildcard():
    """A5: host=0.0.0.0 + 无 token → /metrics 403 (活体验证)。"""
    ep = _make_real_endpoints(host="0.0.0.0", metrics_token=None)
    server = await websockets.serve(
        _dummy_ws, "127.0.0.1", 0, process_request=ep.process_request
    )
    port = server.sockets[0].getsockname()[1]
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"http://127.0.0.1:{port}/metrics")
            assert r.status_code == 403
            assert "funasr_queue_size" not in r.text
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
