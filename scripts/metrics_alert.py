#!/usr/bin/env python3
"""可观测性告警: 定时拉 /health + /metrics, 超阈值发企微 (launchd 周期拉起)。

与 server 进程**解耦** —— server 整个挂了, 本脚本拉不到 /health 也能告警 (server
内嵌方案做不到)。纯逻辑在 src/observability/alerting.py (已单测), 本文件只是薄壳:
HTTP 抓取 + 复用 src/utils/notification.send_custom_notification 发送 + state 落盘。

用法 (单次执行):
    venv/bin/python scripts/metrics_alert.py
    venv/bin/python scripts/metrics_alert.py --selftest
launchd 用 StartInterval 周期拉起即可 (见 docs/部署.md 告警节)。

退出码: 0=无待发或全部送达; 非 0=有告警未送达 / 通知未启用 / selftest 失败。
发送失败不写入已告警 state, 下轮继续重发 (fail-loud, 不靠静默期吞掉)。

开关: config.observability.alert_enabled (env FUNASR_ALERT_ENABLED=true)。未启用直接退出。
阈值: alert_queue_saturation_ratio / alert_error_surge_threshold / alert_cooldown_seconds。
state: 默认落在 DB 同目录 alert_state.json, env FUNASR_ALERT_STATE_PATH 可覆盖。
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import tempfile
import time

# 保证可从项目根 import src.*
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import httpx
from loguru import logger

from src.core.config import config
from src.observability.alerting import (
    AlertThresholds,
    build_probe,
    evaluate_alerts,
    load_state,
)
from src.utils.notification import send_custom_notification


def _scrape_base() -> str:
    """从 server 配置构造探测地址；通配绑定地址转换为本机回环地址。"""
    host = config.server.host.strip()
    if host in {"", "0.0.0.0"}:
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{config.server.port}"


def _save_state_atomically(path: str, state: dict) -> None:
    """在 state 同目录写临时文件，完整序列化后原子替换旧文件。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=parent,
        prefix=f".{os.path.basename(path)}.",
        delete=False,
    ) as temp_file:
        json.dump(state, temp_file, ensure_ascii=False, indent=2)
    os.replace(temp_file.name, path)


def _state_path() -> str:
    env = os.getenv("FUNASR_ALERT_STATE_PATH")
    if env:
        return env
    db_dir = os.path.dirname(config.database.path) or "."
    return os.path.join(db_dir, "alert_state.json")


def probe_once(base: str, token, timeout: float) -> dict:
    """抓 /health + /metrics, 组装成 evaluate_alerts 的 probe。

    reachable 看"是否拿到任何 HTTP 响应"(503 degraded 也算可达), 仅连接错误 → 不可达。
    /metrics 仅 200 才取文本；取不到时由 main 标记 ALERT-METRICS-UNAVAILABLE 并非零退出。
    """
    health_ok = False
    health_body = None
    metrics_text = None
    metrics_status_code = None
    try:
        with httpx.Client(timeout=timeout) as client:
            try:
                r = client.get(base + "/health")
                health_ok = True  # 拿到响应即可达 (200 或 503)
                try:
                    health_body = r.json()
                except Exception:
                    health_body = None
            except httpx.RequestError as exc:
                logger.warning(f"/health 抓取失败 (视为不可达): {exc}")

            try:
                params = {"token": token} if token else None
                r = client.get(base + "/metrics", params=params)
                metrics_status_code = r.status_code
                if r.status_code == 200:
                    metrics_text = r.text
                else:
                    logger.warning(f"/metrics 返回 HTTP {r.status_code}")
            except httpx.RequestError as exc:
                logger.warning("/metrics 抓取失败 (请求异常)")
    except Exception as exc:  # httpx.Client 构造等极端情况
        logger.error(f"探测异常: {exc}")

    probe = build_probe(health_ok=health_ok, health_body=health_body, metrics_text=metrics_text)
    probe["metrics_status_code"] = metrics_status_code
    return probe


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析命令行。argv=None 时用 sys.argv[1:]。"""
    parser = argparse.ArgumentParser(
        description="可观测性告警: 拉 /health+/metrics, 超阈值发企微"
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="发一条测试通知后退出; 失败退出码非 0",
    )
    return parser.parse_args(argv)


async def _selftest_send() -> int:
    """发一条 selftest 通知。成功返回 0, 未送达返回非 0。"""
    ok = await send_custom_notification(
        "🔔 FunASR 告警 selftest",
        "metrics_alert --selftest",
    )
    if not ok:
        logger.error("ALERT-SEND-FAILED selftest 通知未送达")
        return 1
    logger.info("selftest 通知已送达")
    return 0


async def _send_all(alerts: list[dict]) -> set[str]:
    """逐条发企微, 返回成功送达的 key。失败打 ALERT-SEND-FAILED, 不吞异常。"""
    sent: set[str] = set()
    for a in alerts:
        ok = await send_custom_notification(a["title"], a["body"])
        if ok:
            sent.add(a["key"])
        else:
            logger.error(f"ALERT-SEND-FAILED key={a['key']} title={a['title']}")
    return sent


def _state_with_only_sent(
    prev_state: dict,
    evaluated_state: dict,
    alerts: list[dict],
    sent_keys: set[str],
) -> dict:
    """evaluated_state 是「假设全部送达」的下一拍; 未送达的 key 回退 prev 以便下轮重发。"""
    new_state = copy.deepcopy(evaluated_state)
    prev_active = prev_state.get("active", {})
    new_active = new_state.setdefault("active", {})
    for a in alerts:
        key = a["key"]
        if key in sent_keys:
            continue
        if key in prev_active:
            new_active[key] = copy.deepcopy(prev_active[key])
        else:
            new_active.pop(key, None)
        # 事件型 error_surge 靠 baseline 判定增量; 未送达则不推进, 否则下轮不再重发。
        if key == "error_surge":
            if "error_total_seen" in prev_state:
                new_state["error_total_seen"] = prev_state["error_total_seen"]
            else:
                new_state.pop("error_total_seen", None)
    return new_state


def main(argv: list[str] | None = None) -> int:
    """入口。先发送, 只把送达成功的告警写入 state; 任一失败则非 0。"""
    args = _parse_args(argv)
    if args.selftest:
        return asyncio.run(_selftest_send())

    obs = config.observability
    if not obs.alert_enabled:
        logger.info("告警未启用 (alert_enabled=false), 跳过。")
        return 0

    thresholds = AlertThresholds(
        queue_saturation_ratio=obs.alert_queue_saturation_ratio,
        error_surge_threshold=obs.alert_error_surge_threshold,
        cooldown_seconds=obs.alert_cooldown_seconds,
    )
    base = _scrape_base()
    probe = probe_once(base, obs.metrics_token, timeout=10.0)
    # 整个服务不可达时由 server_down 规则告警；单独的指标端点不可用则必须显式失败。
    metrics_unavailable = probe["metrics"] is None and (
        probe["reachable"] or probe["metrics_status_code"] is not None
    )
    if metrics_unavailable:
        logger.error("ALERT-METRICS-UNAVAILABLE /metrics 未提供可用指标")

    state_path = _state_path()
    prev_state = load_state(state_path)
    result = evaluate_alerts(
        probe=probe, prev_state=prev_state, thresholds=thresholds, now_ts=time.time()
    )

    alerts = result["alerts"]
    sent_keys: set[str] = set()
    if alerts:
        logger.warning(f"触发 {len(alerts)} 条告警: {[a['key'] for a in alerts]}")
        sent_keys = asyncio.run(_send_all(alerts))
    else:
        logger.debug("本轮无告警。")

    _save_state_atomically(
        state_path,
        _state_with_only_sent(prev_state, result["state"], alerts, sent_keys),
    )

    missed = [a["key"] for a in alerts if a["key"] not in sent_keys]
    if missed:
        logger.error(
            f"ALERT-SEND-FAILED {len(missed)}/{len(alerts)} 条告警未送达, 本轮不记已告警"
        )
        return 1
    return 1 if metrics_unavailable else 0


if __name__ == "__main__":
    sys.exit(main())
