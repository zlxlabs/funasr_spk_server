"""metrics_alert 发送路径: 先发后记、fail-loud、selftest。

公开入口是 scripts/metrics_alert.py 的 main(); 只 mock 探测/发送的最外层调用,
不直接喂 evaluate_alerts 的内部 helper。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

# 无 .env 时 NotificationConfig.enabled 默认 True，import config 会 sys.exit。
# setdefault 不覆盖已有值（CI / 本机若已配 webhook 保持原样）。
os.environ.setdefault("FUNASR_NOTIFICATION_ENABLED", "false")

import httpx
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
_MODULE_NAME = "metrics_alert_under_test"


def _load_metrics_alert():
    """按文件路径加载脚本模块 (scripts/ 不是 package)。"""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, ROOT / "scripts" / "metrics_alert.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _UnreachableClient:
    """httpx.Client 替身: /health 与 /metrics 都连不上 → server_down。"""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, params=None):
        raise httpx.ConnectError("unreachable")


@pytest.fixture
def metrics_alert():
    return _load_metrics_alert()


@pytest.fixture
def alert_env(monkeypatch, tmp_path, metrics_alert):
    """打开告警开关、把 state 指到临时文件、探测恒不可达。"""
    from src.core.config import config

    state_path = tmp_path / "alert_state.json"
    monkeypatch.setattr(config.observability, "alert_enabled", True)
    monkeypatch.setenv("FUNASR_ALERT_STATE_PATH", str(state_path))
    monkeypatch.setattr(metrics_alert.httpx, "Client", _UnreachableClient)
    return state_path


def _error_records():
    """收集 loguru ERROR 记录, 供断言 ALERT-SEND-FAILED。"""
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="ERROR")
    return records, sink_id


def test_send_failure_nonzero_and_state_not_recorded(monkeypatch, metrics_alert, alert_env):
    """mock send 失败 → 退出码非 0, state 不记录该告警, 打 ALERT-SEND-FAILED。"""
    from unittest.mock import AsyncMock

    state_path = alert_env
    monkeypatch.setattr(
        metrics_alert, "send_custom_notification", AsyncMock(return_value=False)
    )
    records, sink_id = _error_records()
    try:
        rc = metrics_alert.main([])
    finally:
        logger.remove(sink_id)

    assert rc != 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "server_down" not in state.get("active", {})
    assert any("ALERT-SEND-FAILED" in r for r in records)


def test_send_success_zero_and_state_recorded(monkeypatch, metrics_alert, alert_env):
    """mock send 成功 → 退出 0 且 state 记录; 发送时 state 还不能已记该告警。"""
    state_path = alert_env
    order: list[str] = []

    async def send_ok(title, content, msg_type="text"):
        order.append("send")
        if state_path.exists():
            st = json.loads(state_path.read_text(encoding="utf-8"))
            assert "server_down" not in st.get("active", {})
        return True

    monkeypatch.setattr(metrics_alert, "send_custom_notification", send_ok)
    rc = metrics_alert.main([])
    assert rc == 0
    assert order == ["send"]
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "server_down" in state.get("active", {})


def test_notification_disabled_with_alert_nonzero(monkeypatch, metrics_alert, alert_env):
    """通知未启用且有告警待发 → 非 0, 打 ALERT-SEND-FAILED, 不记 state。"""
    from src.core.config import config

    state_path = alert_env
    monkeypatch.setattr(config.notification, "enabled", False)
    records, sink_id = _error_records()
    try:
        rc = metrics_alert.main([])
    finally:
        logger.remove(sink_id)

    assert rc != 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "server_down" not in state.get("active", {})
    assert any("ALERT-SEND-FAILED" in r for r in records)


def test_selftest_failure_nonzero(monkeypatch, metrics_alert):
    """--selftest 发送失败 → 非 0, 打 ALERT-SEND-FAILED。"""
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        metrics_alert, "send_custom_notification", AsyncMock(return_value=False)
    )
    records, sink_id = _error_records()
    try:
        rc = metrics_alert.main(["--selftest"])
    finally:
        logger.remove(sink_id)

    assert rc != 0
    assert any("ALERT-SEND-FAILED" in r for r in records)
