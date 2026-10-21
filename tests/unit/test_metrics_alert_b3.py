"""运维告警脚本回归：原子状态、按服务配置探测、指标不可用显式失败。"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest
from loguru import logger

os.environ.setdefault("FUNASR_NOTIFICATION_ENABLED", "false")

ROOT = Path(__file__).resolve().parents[2]
_MODULE_NAME = "metrics_alert_b3_under_test"
_METRICS = (
    "funasr_queue_size 0\n"
    "funasr_queue_max 10\n"
    'funasr_errors_total{kind="worker"} 3\n'
)


def _load_metrics_alert():
    """通过脚本入口所在模块加载告警执行器。"""
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, ROOT / "scripts" / "metrics_alert.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, status_code: int, *, text: str = "", body: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._body = body or {}

    def json(self):
        return self._body


class _Client:
    def __init__(self, *args, **kwargs):
        self.urls: list[tuple[str, dict | None]] = []
        _clients.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url: str, params: dict | None = None):
        self.urls.append((url, params))
        if url.endswith("/health"):
            if _health_unreachable:
                raise httpx.ConnectError("unreachable")
            return _Response(200, body={"status": "healthy"})
        return _Response(_metrics_status, text=_METRICS)


_clients: list[_Client] = []
_metrics_status = 200
_health_unreachable = False


@pytest.fixture
def metrics_alert(monkeypatch, tmp_path):
    """把告警落盘隔离到临时目录，并默认模拟健康服务。"""
    global _metrics_status, _health_unreachable
    module = _load_metrics_alert()
    from src.core.config import config

    monkeypatch.setattr(config.observability, "alert_enabled", True)
    monkeypatch.setattr(config.observability, "metrics_token", None)
    monkeypatch.setenv("FUNASR_ALERT_STATE_PATH", str(tmp_path / "alert_state.json"))
    monkeypatch.setattr(module.httpx, "Client", _Client)
    _clients.clear()
    _metrics_status = 200
    _health_unreachable = False
    return module


def test_state_write_failure_keeps_previous_state_intact(monkeypatch, metrics_alert, tmp_path):
    """公开 main 写 state 中途失败时，旧 JSON 必须保持完整、字节不变。"""
    state_path = tmp_path / "alert_state.json"
    previous = b'{"active":{},"error_total_seen":3}\n'
    state_path.write_bytes(previous)

    def partial_dump(_state, file_obj, **kwargs):
        file_obj.write("{")
        raise OSError("simulated interrupted state write")

    monkeypatch.setattr(json, "dump", partial_dump)
    with pytest.raises(OSError, match="simulated interrupted state write"):
        metrics_alert.main([])

    assert state_path.read_bytes() == previous


def test_main_probes_configured_server_address(metrics_alert, monkeypatch):
    """公开 main 必须将 server.host 与 server.port 组合为探测地址。"""
    from src.core.config import config

    monkeypatch.setattr(config.server, "host", "192.0.2.10")
    monkeypatch.setattr(config.server, "port", 9876)

    assert metrics_alert.main([]) == 0
    assert [url for url, _ in _clients[-1].urls] == [
        "http://192.0.2.10:9876/health",
        "http://192.0.2.10:9876/metrics",
    ]


def test_metrics_403_fails_loud_with_unavailable_marker(
    metrics_alert, monkeypatch
):
    """公开 main 收到 /metrics 403 时必须非零退出并记录专用错误标记。"""
    global _metrics_status, _health_unreachable
    from unittest.mock import AsyncMock

    _metrics_status = 403
    _health_unreachable = True
    monkeypatch.setattr(
        metrics_alert, "send_custom_notification", AsyncMock(return_value=True)
    )
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="ERROR")
    try:
        result = metrics_alert.main([])
    finally:
        logger.remove(sink_id)

    assert result != 0
    assert any("ALERT-METRICS-UNAVAILABLE" in record for record in records)