"""真 server + 真 websocket client 端到端测试的共享 helper（引擎无关）。

被 test_funasr_server_websocket_e2e.py 使用。下划线前缀 → pytest 不收集本文件。

NOTE: test_qwen3_server_websocket_e2e.py 目前仍有一份自带的平行 helper（历史），
后续可迁移到本模块统一（DRY）。本模块先服务 funasr e2e，不动既有 qwen3 e2e（避免回归既有测试）。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import time
from pathlib import Path

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None


# 可被调用方覆盖的超时常量（默认值覆盖 handshake / 长音频结果等待）
HANDSHAKE_TIMEOUT_SEC = 30.0   # connected / upload_request 应答 / 分片 ack
RESULT_TIMEOUT_SEC = 900.0     # 等转录结果（长音频用例可能跑十几分钟）

# 服务端失败双发：先 task_progress(status=failed|timed_out|cancelled)，再补一条
# type="error" 的显式终态消息（notify_task_error）。两条都是终态信号，本 helper 先命中
# task_progress 分支即返回，error 分支（下方 while 循环里）是兜底。
# 保留 task_progress 判定是为了兼容只发进度消息的老服务端。
TERMINAL_FAILURE_STATUSES = {"failed", "timed_out", "cancelled"}


def free_port() -> int:
    """让 OS 分配一个空闲端口。"""
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for_port(host: str, port: int, timeout: float = 120.0) -> bool:
    """轮询直到 TCP 端口可连或超时。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket()
            s.settimeout(1.0)
            s.connect((host, port))
            s.close()
            return True
        except (OSError, socket.timeout):
            time.sleep(0.5)
    return False


def file_hash_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


async def _recv_json(ws, timeout: float, what: str) -> dict:
    """带超时的 recv + json 解析。超时抛 AssertionError 而非永久挂起。"""
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    except asyncio.TimeoutError:
        raise AssertionError(
            f"等待 {what} 超时 ({timeout}s) —— 服务端未发终态消息"
        ) from None
    return json.loads(raw)


async def client_upload_and_wait(
    ws_url: str,
    audio: Path,
    *,
    force_refresh: bool = True,
    output_format: str = "json",
    extra_request: dict | None = None,
) -> dict:
    """完整 client 流程: connect → upload_request → upload_data → 等 task_complete。

    走真 server → task_manager → _process_task 结果处理层（这正是 parity 直调
    transcribe 覆盖不到的那段）。

    Returns dict: {ok, task_id, wall_time, cached, result, error?}。
    """
    t0 = time.time()
    data = audio.read_bytes()
    digest = file_hash_md5(audio)
    b64 = base64.b64encode(data).decode("utf-8")

    request_data = {
        "file_name": audio.name,
        "file_size": len(data),
        "file_hash": digest,
        "force_refresh": force_refresh,
        "output_format": output_format,
    }
    if extra_request:
        request_data.update(extra_request)

    async with websockets.connect(ws_url, max_size=200 * 1024 * 1024) as ws:
        connected = await _recv_json(ws, HANDSHAKE_TIMEOUT_SEC, "connected")
        assert connected.get("type") == "connected", f"unexpected: {connected}"

        await ws.send(json.dumps({"type": "upload_request", "data": request_data}))
        resp = await _recv_json(ws, HANDSHAKE_TIMEOUT_SEC, "upload_request 应答")
        if resp["type"] == "error":
            return {"ok": False, "error": resp["data"]["message"], "wall_time": time.time() - t0}
        if resp["type"] == "task_complete":
            return {
                "ok": True, "cached": True,
                "task_id": resp["data"].get("task_id", "cached"),
                "wall_time": time.time() - t0,
                "result": resp["data"]["result"],
            }
        task_id = resp["data"].get("task_id")
        assert task_id, f"无 task_id: {resp}"

        await ws.send(json.dumps({
            "type": "upload_data",
            "data": {"task_id": task_id, "file_data": b64},
        }))

        while True:
            msg = await _recv_json(ws, RESULT_TIMEOUT_SEC, f"task {task_id} 终态")
            if msg["type"] == "task_progress":
                status = (msg.get("data") or {}).get("status")
                if status in TERMINAL_FAILURE_STATUSES:
                    return {
                        "ok": False,
                        "task_id": task_id,
                        "error": (msg.get("data") or {}).get("message") or status,
                        "wall_time": time.time() - t0,
                    }
                continue
            if msg["type"] == "task_complete":
                return {
                    "ok": True, "cached": False,
                    "task_id": task_id,
                    "wall_time": time.time() - t0,
                    "result": msg["data"]["result"],
                }
            if msg["type"] == "error":
                return {"ok": False, "task_id": task_id, "error": msg["data"]["message"],
                        "wall_time": time.time() - t0}


def _split_chunks(data: bytes, chunk_size: int) -> list[bytes]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)] or [b""]


async def client_chunked_upload_and_wait(
    ws_url: str,
    audio: Path,
    *,
    chunk_size: int = 512 * 1024,
    force_refresh: bool = True,
    output_format: str = "json",
    extra_request: dict | None = None,
) -> dict:
    """完整分片 client 流程: connect → chunked_upload_request → 逐帧 upload_chunk →
    (收齐自动 finalize) → 等 task_complete。

    走真 server 的**分片重组 + offset 写盘 + finalize → task_manager → _process_task**
    全链路 (单文件 upload_data 路径覆盖不到分片重组那段)。

    Returns dict: {ok, task_id, wall_time, cached, num_chunks, result, error?}。
    """
    t0 = time.time()
    data = audio.read_bytes()
    digest = file_hash_md5(audio)
    chunks = _split_chunks(data, chunk_size)
    total_chunks = len(chunks)

    # 分片上传走 upload_request + upload_mode=chunked (非独立消息类型),
    # server 据此路由到 _handle_chunked_upload_request, 回 upload_ready
    request_data = {
        "file_name": audio.name,
        "file_size": len(data),
        "file_hash": digest,
        "upload_mode": "chunked",
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "force_refresh": force_refresh,
        "output_format": output_format,
    }
    if extra_request:
        request_data.update(extra_request)

    async with websockets.connect(ws_url, max_size=200 * 1024 * 1024) as ws:
        connected = await _recv_json(ws, HANDSHAKE_TIMEOUT_SEC, "connected")
        assert connected.get("type") == "connected", f"unexpected: {connected}"

        # 1. 发起分片会话 (upload_request + upload_mode=chunked)
        await ws.send(json.dumps({"type": "upload_request", "data": request_data}))
        resp = await _recv_json(ws, HANDSHAKE_TIMEOUT_SEC, "chunked upload_request 应答")
        if resp["type"] == "error":
            return {"ok": False, "error": resp["data"]["message"],
                    "wall_time": time.time() - t0}
        assert resp["type"] == "upload_ready", f"期待 upload_ready, 实得: {resp}"
        task_id = resp["data"]["task_id"]

        # 2. 逐帧发分片 (带 md5 chunk_hash); 每帧读一条响应 (通常 chunk_received)
        for idx, chunk in enumerate(chunks):
            await ws.send(json.dumps({
                "type": "upload_chunk",
                "data": {
                    "task_id": task_id,
                    "chunk_index": idx,
                    "chunk_data": base64.b64encode(chunk).decode("utf-8"),
                    "chunk_hash": hashlib.md5(chunk).hexdigest(),
                },
            }))
            ack = await _recv_json(
                ws, HANDSHAKE_TIMEOUT_SEC, f"chunk {idx}/{total_chunks} ack"
            )
            if ack["type"] == "error":
                return {"ok": False, "task_id": task_id, "error": ack["data"]["message"],
                        "num_chunks": total_chunks, "wall_time": time.time() - t0}

        # 3. 收齐后等终态。cached 判定: finalize 命中缓存时直发 task_complete,
        #    不经 upload_complete/task_queued; fresh 则先有 upload_complete/task_queued
        #    再(稍后)推 task_complete。据此区分 (chunk_received 是逐片 ack, 忽略)。
        saw_progress = False
        while True:
            msg = await _recv_json(ws, RESULT_TIMEOUT_SEC, f"task {task_id} 终态")
            mtype = msg["type"]
            if mtype == "chunk_received":
                continue
            if mtype == "task_progress":
                status = (msg.get("data") or {}).get("status")
                if status in TERMINAL_FAILURE_STATUSES:
                    return {
                        "ok": False,
                        "task_id": task_id,
                        "error": (msg.get("data") or {}).get("message") or status,
                        "num_chunks": total_chunks,
                        "wall_time": time.time() - t0,
                    }
                continue
            if mtype in ("upload_complete", "task_queued"):
                saw_progress = True
                continue
            if mtype == "task_complete":
                return {"ok": True, "cached": not saw_progress, "task_id": task_id,
                        "num_chunks": total_chunks, "wall_time": time.time() - t0,
                        "result": msg["data"]["result"]}
            if mtype == "error":
                return {"ok": False, "task_id": task_id, "error": msg["data"]["message"],
                        "num_chunks": total_chunks, "wall_time": time.time() - t0}
