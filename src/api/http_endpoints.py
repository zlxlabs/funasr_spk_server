"""可观测性仪表盘 HTTP 端点 (P1) — websockets 同端口 process_request 钩子.

设计定案: docs/开发/2026-06-16-可观测性仪表盘与测试加固-设计定案与落地计划.md

架构 (D3): 不新起 HTTP server / 不多开端口, 复用 websockets.serve 的 process_request
回调. websockets==12.0 是 **legacy** API, 回调签名:
    async def process_request(path, request_headers) -> Optional[(status, headers, body)]
返回元组 → 直接 HTTP 响应 (不进 ws 升级); 返回 None → 照常 ws 握手.

  HTTP GET ─► process_request ─┬─ /health,/metrics 命中 ─► (status, headers, body)
                              └─ 其它 (ws 升级) ─► None ─► ws handler

注意 (codex 评审):
  #1 只处理 GET: legacy process_request 在 ws 握手前触发, 握手必是 GET. HEAD/POST
     到不了这里 (协议层先拒). 探针**必须用 GET** (文档化要求).
  #2 path 含 query string: 用 urlsplit 取 path 路由 + parse_qs 取 token, 禁止直接
     `path == "/metrics"` 等值比较 (否则 /metrics?token=x 漏路由 + 断鉴权).
  #11 VRAM 探针 (nvidia-smi 同步 subprocess) 绝不在事件循环里直接调: 走
     run_in_executor 移到线程 + TTL 缓存, 不冻所有 ws 处理.
  A5 安全: /metrics 只出聚合计数; metrics_token 未设 + host=0.0.0.0 → 拒绝;
     /metrics 响应绝不回显 token.

A3-final: /health 只做 liveness (维护循环活 + 能接任务). 模型 eager 加载
  (main.py:47-49), 故无 readiness gate; 冷启动盲区记 TODO #23.
"""
from __future__ import annotations

import asyncio
import http
import json
import time
from typing import Optional
from urllib.parse import urlsplit, parse_qs

from loguru import logger

from src.core.capabilities import build_asr_capabilities
from src.core import gpu_mem
from src.core.runtime import detect_runtime


# /metrics 指标前缀
_NS = "funasr"
# 绑定到这些 host 视为"裸暴露"(非 LAN 限定), 无 token 时 /metrics 拒绝 (A5)
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})
# VRAM 缓存 TTL (秒): nvidia-smi fork 较贵, 限频; scrape 15s 完全够新鲜
_VRAM_CACHE_TTL_SEC = 10.0


# ==================== 纯函数 (易测) ====================

def render_prometheus(
    snapshot: dict,
    cache_stats: Optional[dict] = None,
    vram_mib: Optional[int] = None,
    uptime_s: Optional[float] = None,
) -> str:
    """把指标快照渲染成 Prometheus 文本格式 (事实标准, curl 可读, 零新依赖).

    snapshot 来自 task_manager.get_metrics_snapshot(). cache_stats / vram / uptime
    探不到时省略对应指标 (Prometheus 容忍缺失, 不输出比输出 0 更诚实).
    """
    lines: list[str] = []

    def gauge(name: str, value, help_text: str):
        lines.append(f"# HELP {_NS}_{name} {help_text}")
        lines.append(f"# TYPE {_NS}_{name} gauge")
        lines.append(f"{_NS}_{name} {value}")

    def counter_labeled(name: str, mapping: dict, label: str, help_text: str):
        lines.append(f"# HELP {_NS}_{name} {help_text}")
        lines.append(f"# TYPE {_NS}_{name} counter")
        for k, v in sorted(mapping.items()):
            lines.append(f'{_NS}_{name}{{{label}="{k}"}} {v}')

    # ---- 瞬时 gauge ----
    gauge("queue_size", snapshot["queue_size"], "当前任务队列深度 (准入控制)")
    gauge("queue_max", snapshot["max_queue_size"], "任务队列上限")
    gauge("tasks_pending", snapshot["pending"], "PENDING 任务数")
    gauge("tasks_inflight", snapshot["processing"], "在途 (PROCESSING) 任务数 = 真实并发占用")
    gauge("tasks_in_memory", snapshot["tasks_in_memory"], "self.tasks 当前驻留任务数")
    gauge("task_seconds_ema", round(float(snapshot["task_seconds_ema"]), 3), "单任务处理时长 EMA (秒)")
    gauge("pool_size", snapshot["pool_size"], "有效并发度 (引擎池大小)")

    # ---- engine info (label gauge, 值恒 1) ----
    lines.append(f"# HELP {_NS}_engine_info 当前引擎 (label=engine, 值恒 1)")
    lines.append(f"# TYPE {_NS}_engine_info gauge")
    lines.append(f'{_NS}_engine_info{{engine="{snapshot["engine"]}"}} 1')

    # ---- 单调 counter ----
    counter_labeled("tasks_terminal_total", snapshot.get("terminal_total", {}),
                    "status", "终态任务累计数 (单调, 活过 TTL 淘汰)")
    counter_labeled("errors_total", snapshot.get("errors_total", {}),
                    "kind", "错误累计数 (单调, 按 kind)")

    # ---- 缓存 (探到才出) ----
    if cache_stats:
        for key, help_text in (
            ("total_cached", "缓存总行数"),
            ("hits", "缓存命中数"),
            ("misses", "缓存未命中数"),
            ("projected_serves", "diarize 投影命中数"),
        ):
            if key in cache_stats:
                gauge(f"cache_{key}", cache_stats[key], help_text)

    # ---- VRAM (仅 cuda 探到才出, #11 已 off-loop) ----
    if vram_mib is not None:
        gauge("vram_free_mib", vram_mib, "当前 CUDA 卡空闲显存 (MiB)")

    # ---- uptime ----
    if uptime_s is not None:
        gauge("uptime_seconds", round(float(uptime_s), 1), "服务运行时长 (秒)")

    return "\n".join(lines) + "\n"


def evaluate_health(is_running: bool, maintenance_alive: bool) -> tuple[int, dict]:
    """A3-final liveness 判定: 维护循环活 + 能接任务 → 200, 否则 503.

    不查"模型加载完"(eager load 下几乎恒真, 用模型当 gate 会误报). 不查池
    worker (懒实例化可能尚未建, 池真死由 errors_total + inflight 卡住体现).
    """
    checks = {
        "maintenance_loop": bool(maintenance_alive),
        "accepting_tasks": bool(is_running),
    }
    healthy = all(checks.values())
    return (200 if healthy else 503), {
        "status": "healthy" if healthy else "degraded",
        "checks": checks,
    }


def authorize_metrics(
    path: str,
    auth_header: Optional[str],
    configured_token: Optional[str],
    host: str,
) -> Optional[int]:
    """/metrics 鉴权 (A5). 返回 None=放行, 否则返回拒绝的 HTTP 状态码 (403).

    - configured_token 设了: 校验 ?token= (query, #2) 或 Authorization: [Bearer] X.
    - configured_token 未设: host 是 0.0.0.0/:: (裸暴露) → 拒绝; 绑具体 LAN/loopback → 放行.
    """
    if configured_token:
        supplied = _extract_token(path, auth_header)
        return None if supplied == configured_token else 403
    # 未设 token: 仅当显式绑非通配 host 才放行 (避免 0.0.0.0 全网段裸暴露)
    if (host or "").strip() in _WILDCARD_HOSTS:
        return 403
    return None


def _extract_token(path: str, auth_header: Optional[str]) -> Optional[str]:
    """从 ?token= query 或 Authorization header (支持 Bearer 前缀) 取 token."""
    q = parse_qs(urlsplit(path).query)
    if "token" in q and q["token"]:
        return q["token"][0]
    if auth_header:
        h = auth_header.strip()
        if h.lower().startswith("bearer "):
            return h[7:].strip()
        return h
    return None


# ==================== 端点装配 ====================

class HttpEndpoints:
    """持有依赖引用, 提供 websockets process_request 回调.

    用法 (main.py): ep = HttpEndpoints(task_manager, db_manager, config, started_at=...)
                    websockets.serve(handler, ..., process_request=ep.process_request)
    """

    def __init__(self, task_manager, db_manager, config, started_at: Optional[float] = None):
        self._tm = task_manager
        self._db = db_manager
        self._cfg = config
        self._started_at = started_at
        # VRAM 缓存 (#11): (monotonic 时刻, 值) — 限频 nvidia-smi fork
        self._vram_cache: tuple[float, Optional[int]] = (0.0, None)

    async def process_request(self, path, request_headers):
        """websockets 12.0 legacy 回调. 命中 /health|/metrics 返响应, 否则 None 走 ws."""
        # ⚠️ ws 握手请求 (Upgrade: websocket) 一律放行 (返 None), 不被 HTTP 端点劫持。
        # 否则客户端连 ws://host:port/ (根路径) 会被 HTML 状态页拦成 HTTP 200 → 握手失败
        # "无法连接到服务器" (生产事故 2026-06-17)。不管什么路径, 有 Upgrade 头就是 ws。
        upgrade = _header_get(request_headers, "Upgrade")
        if upgrade and upgrade.strip().lower() == "websocket":
            return None

        route = urlsplit(path).path
        if route == "/capabilities":
            return _json_response(200, self._build_capabilities())

        obs = self._cfg.observability
        if not obs.metrics_enabled:
            return None  # 端点总开关关 → 退回纯 ws

        if route == "/":
            # 极简状态页: 静态 HTML/JS, 不嵌 secret. token 由用户在 URL ?token= 提供,
            # 页面 JS 读自己的 location.search 去 fetch /metrics (localhost 无 token 直开 /).
            return _html_response(200, _STATUS_HTML)

        if route == "/health":
            code, body = evaluate_health(
                is_running=getattr(self._tm, "is_running", False),
                maintenance_alive=self._maintenance_alive(),
            )
            return _json_response(code, body)

        if route == "/metrics":
            auth_header = _header_get(request_headers, "Authorization")
            denied = authorize_metrics(path, auth_header, obs.metrics_token, self._cfg.server.host)
            if denied is not None:
                # 拒绝响应绝不回显 token (A5)
                return _text_response(denied, "forbidden\n")
            text = await self.build_metrics_text()
            return _text_response(200, text)

        return None  # 非端点路径 → ws 升级

    def _build_capabilities(self) -> dict[str, object]:
        """Compose capabilities from the configured engine and effective runtime."""
        engine = self._cfg.transcription.default_engine
        return build_asr_capabilities(
            schema_version=1,
            engine=engine,
            runtime=detect_runtime().name,
            features={"terms": engine in ("funasr", "qwen3")},
        )

    def _maintenance_alive(self) -> bool:
        t = getattr(self._tm, "_maintenance_task", None)
        return t is not None and not t.done()

    async def build_metrics_text(self) -> str:
        """组装 /metrics 文本: snapshot(同步) + cache_stats(await) + vram(off-loop) + uptime."""
        snapshot = self._tm.get_metrics_snapshot()

        cache_stats = None
        try:
            cache_stats = await self._db.get_cache_stats()
        except Exception as exc:  # 缓存统计失败不拖垮 /metrics
            logger.debug(f"/metrics 读 cache_stats 失败 (省略): {exc}")

        vram = await self._read_vram_cached()

        uptime = (time.monotonic() - self._started_at) if self._started_at is not None else None
        return render_prometheus(snapshot, cache_stats=cache_stats, vram_mib=vram, uptime_s=uptime)

    async def _read_vram_cached(self) -> Optional[int]:
        """读 VRAM, off-loop (#11) + TTL 缓存. 非 cuda 机 free_vram_mib 返 None → 省略."""
        now = time.monotonic()
        ts, val = self._vram_cache
        if now - ts < _VRAM_CACHE_TTL_SEC:
            return val
        try:
            loop = asyncio.get_running_loop()
            # nvidia-smi 同步 subprocess 移到线程池, 不冻事件循环
            val = await loop.run_in_executor(None, gpu_mem.free_vram_mib)
        except Exception as exc:
            logger.debug(f"/metrics 读 VRAM 失败 (省略): {exc}")
            val = None
        self._vram_cache = (now, val)
        return val


# ==================== 响应构造 (websockets 12.0 tuple) ====================

_JSON_HEADERS = [("Content-Type", "application/json; charset=utf-8")]
_TEXT_HEADERS = [("Content-Type", "text/plain; version=0.0.4; charset=utf-8")]
_HTML_HEADERS = [("Content-Type", "text/html; charset=utf-8")]


def _json_response(code: int, body: dict):
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return (http.HTTPStatus(code), _JSON_HEADERS, payload)


def _text_response(code: int, text: str):
    return (http.HTTPStatus(code), _TEXT_HEADERS, text.encode("utf-8"))


def _html_response(code: int, html: str):
    return (http.HTTPStatus(code), _HTML_HEADERS, html.encode("utf-8"))


def _header_get(request_headers, name: str) -> Optional[str]:
    """从 websockets Headers 或 dict 取 header (大小写不敏感, best-effort)."""
    if request_headers is None:
        return None
    try:
        getter = getattr(request_headers, "get", None)
        if getter is not None:
            return getter(name)
    except Exception:
        pass
    return None


# ==================== 极简状态页 (静态 HTML/JS) ====================
# 一页够用: fetch /health + /metrics 渲染成人读表格, 每 3s 自刷. 零外部依赖.
# token: 页面读自己 URL 的 ?token= 传给 /metrics fetch (不嵌服务端 HTML, codex A5).
# localhost 无 token 直开 /; 0.0.0.0 设了 token 则开 /?token=xxx.
_STATUS_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FunASR 服务状态</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px; background:#0f1115; color:#d7dbe0; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color:#8a93a0; font-size:12px; margin-bottom:16px; }
  .banner { display:inline-block; padding:8px 16px; border-radius:8px; font-weight:700; font-size:16px; margin-bottom:16px; }
  .ok { background:#13361f; color:#46d17e; border:1px solid #1f6b3c; }
  .bad { background:#3a1618; color:#ff6b72; border:1px solid #7a2a2e; }
  table { border-collapse:collapse; margin:8px 0 20px; min-width:340px; }
  th,td { text-align:left; padding:6px 14px 6px 0; font-size:14px; }
  td.v { font-variant-numeric:tabular-nums; font-weight:600; color:#fff; }
  h2 { font-size:13px; color:#8a93a0; text-transform:uppercase; letter-spacing:.05em; margin:18px 0 4px; }
  .err { color:#ff6b72; }
</style>
</head>
<body>
  <h1>FunASR 服务状态</h1>
  <div class="sub">每 3 秒自动刷新 · 数据源 /health + /metrics</div>
  <div id="banner" class="banner">加载中…</div>
  <div id="content"></div>
<script>
const TOKEN = new URLSearchParams(location.search).get('token');
const qs = TOKEN ? ('?token=' + encodeURIComponent(TOKEN)) : '';

function parseMetrics(text) {
  const flat = {}, labeled = {};
  for (const line of text.split('\\n')) {
    if (!line || line[0] === '#') continue;
    const m = line.match(/^([a-z0-9_]+)(\\{[^}]*\\})?\\s+(.+)$/i);
    if (!m) continue;
    const [, name, labels, val] = m;
    if (labels) { (labeled[name] = labeled[name] || []).push([labels, val]); }
    else { flat[name] = val; }
  }
  return { flat, labeled };
}

function row(k, v) { return '<tr><td>' + k + '</td><td class="v">' + v + '</td></tr>'; }

function render(health, metricsText, metricsErr) {
  const banner = document.getElementById('banner');
  if (health && health.status === 'healthy') {
    banner.className = 'banner ok'; banner.textContent = '● HEALTHY';
  } else {
    banner.className = 'banner bad';
    banner.textContent = '● ' + (health ? String(health.status).toUpperCase() : 'UNREACHABLE');
  }
  let html = '';
  if (health && health.checks) {
    html += '<h2>健康检查</h2><table>';
    for (const [k, v] of Object.entries(health.checks)) html += row(k, v ? '✓' : '✗');
    html += '</table>';
  }
  if (metricsErr) {
    html += '<h2>指标</h2><p class="err">/metrics 读取失败: ' + metricsErr +
            '（host=0.0.0.0 需在网址加 ?token=你的token）</p>';
  } else if (metricsText) {
    const { flat, labeled } = parseMetrics(metricsText);
    const g = (n) => flat['funasr_' + n] !== undefined ? flat['funasr_' + n] : '—';
    html += '<h2>队列 / 并发</h2><table>'
      + row('队列深度', g('queue_size') + ' / ' + g('queue_max'))
      + row('在途 (inflight)', g('tasks_inflight'))
      + row('等待 (pending)', g('tasks_pending'))
      + row('驻留任务', g('tasks_in_memory'))
      + row('并发度', g('pool_size'))
      + row('单任务 EMA(秒)', g('task_seconds_ema'))
      + '</table>';
    const eng = (labeled['funasr_engine_info'] || []).map(([l]) => (l.match(/engine="([^"]*)"/) || [])[1]).join(',');
    let cacheRows = '';
    ['cache_total_cached','cache_hits','cache_misses','cache_projected_serves','vram_free_mib','uptime_seconds'].forEach(n => {
      if (flat['funasr_' + n] !== undefined) cacheRows += row(n.replace('cache_',''), flat['funasr_' + n]);
    });
    html += '<h2>引擎 / 缓存</h2><table>' + row('引擎', eng || '—') + cacheRows + '</table>';
    const term = labeled['funasr_tasks_terminal_total'] || [];
    const errs = labeled['funasr_errors_total'] || [];
    if (term.length) {
      html += '<h2>终态累计 (单调)</h2><table>';
      term.forEach(([l, v]) => html += row((l.match(/status="([^"]*)"/) || [])[1], v));
      html += '</table>';
    }
    if (errs.length) {
      html += '<h2>错误累计</h2><table>';
      errs.forEach(([l, v]) => html += row((l.match(/kind="([^"]*)"/) || [])[1], v));
      html += '</table>';
    }
  }
  document.getElementById('content').innerHTML = html;
}

async function tick() {
  let health = null, metricsText = null, metricsErr = null;
  try { health = await (await fetch('/health')).json(); } catch (e) { health = null; }
  try {
    const r = await fetch('/metrics' + qs);
    if (r.ok) metricsText = await r.text(); else metricsErr = 'HTTP ' + r.status;
  } catch (e) { metricsErr = String(e); }
  render(health, metricsText, metricsErr);
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""
