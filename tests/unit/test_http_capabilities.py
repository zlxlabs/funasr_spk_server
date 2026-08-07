"""I1 capabilities 协议：纯 helper 与 HTTP/WS 绑定。"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api import http_endpoints, websocket_handler
from src.api.http_endpoints import HttpEndpoints
from src.api.websocket_handler import WebSocketHandler
from src.core.capabilities import build_asr_capabilities


def _config(*, engine="funasr", metrics_enabled=True):
    cfg = MagicMock()
    cfg.observability.metrics_enabled = metrics_enabled
    cfg.observability.metrics_token = None
    cfg.server.host = "127.0.0.1"
    cfg.transcription.default_engine = engine
    cfg.auth.enabled = False
    return cfg


def _endpoints(cfg):
    tm = MagicMock(is_running=True)
    tm._maintenance_task = MagicMock()
    tm._maintenance_task.done.return_value = False
    return HttpEndpoints(tm, MagicMock(), cfg)


def test_capabilities_schema_and_canonical_id_are_minimal():
    result = build_asr_capabilities(
        schema_version=1,
        engine="funasr",
        runtime="cpu",
        features={"terms": True, "provider": True},
    )
    assert set(result) == {"schema_version", "engine", "runtime", "features", "capability_id"}
    assert result["features"] == {"terms": True}
    canonical = json.dumps(
        {key: result[key] for key in ("schema_version", "engine", "runtime", "features")},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert result["capability_id"] == hashlib.sha256(canonical.encode()).hexdigest()


def test_capabilities_id_changes_when_explicit_input_changes():
    base = dict(schema_version=1, engine="funasr", runtime="cpu", features={"terms": True})
    ids = {build_asr_capabilities(**{**base, key: value})["capability_id"] for key, value in (
        ("schema_version", 2), ("engine", "qwen3"), ("runtime", "cuda"),
        ("features", {"terms": False}),
    )}
    assert len(ids) == 4


@pytest.mark.asyncio
async def test_http_capabilities_route_precedes_metrics_flag_and_upgrade(monkeypatch):
    monkeypatch.setattr(http_endpoints, "detect_runtime", lambda: SimpleNamespace(name="cpu"))
    ep = _endpoints(_config(metrics_enabled=False))
    response = await ep.process_request("/capabilities?probe=1", {})
    assert response is not None and json.loads(response[2])['features'] == {"terms": True}
    assert await ep.process_request("/capabilities", {"Upgrade": "websocket"}) is None
    assert await ep.process_request("/health", {}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("engine", "terms"), (("funasr", True), ("qwen3", False)))
async def test_http_capabilities_engine_matrix(monkeypatch, engine, terms):
    monkeypatch.setattr(http_endpoints, "detect_runtime", lambda: SimpleNamespace(name="mac_ane"))
    ep = _endpoints(_config(engine=engine))
    response = await ep.process_request("/capabilities", {})
    body = json.loads(response[2])
    assert body["engine"] == engine
    assert body["runtime"] == "mac_ane"
    assert body["features"] == {"terms": terms}


class _ClosedWebSocket:
    remote_address = ("127.0.0.1", 12345)

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_websocket_connected_uses_same_capability_id(monkeypatch):
    cfg = _config()
    monkeypatch.setattr(websocket_handler, "config", cfg)
    monkeypatch.setattr(websocket_handler, "detect_runtime", lambda: SimpleNamespace(name="cpu"))
    ep = _endpoints(cfg)
    monkeypatch.setattr(http_endpoints, "detect_runtime", lambda: SimpleNamespace(name="cpu"))
    http_body = json.loads((await ep.process_request("/capabilities", {}))[2])

    handler = WebSocketHandler()
    handler._send_message = AsyncMock()
    await handler.handle_connection(_ClosedWebSocket(), "/")
    connected = handler._send_message.await_args.args[2]
    assert connected["capabilities"] == http_body
    assert connected["connection_id"]
    assert connected["message"] == "连接成功"
    assert connected["server_time"]
