"""Ad-hoc: 真 server + 真并发 2 client 跑长音频 (1人 16min + 4人 44min).

不入 git, 一次性 perf 测试. 跑完打印总耗时 / 各 client wall / 检测 speaker 数.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import websockets


# 仓库根目录，由本文件位置推导，避免硬编码某台机器的家目录路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL = PROJECT_ROOT / "tmp_long_audio" / "eval_set"
AUDIO_1SPK = EVAL / "audio_1spk_real.m4a"          # 16min 1 人
AUDIO_4SPK = EVAL / "audio_4spk.m4a"               # 44min 4 人

SERVER_PORT = 18867
SERVER_URL = f"ws://127.0.0.1:{SERVER_PORT}"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def wait_port(host: str, port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket(); s.settimeout(1.0); s.connect((host, port)); s.close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


async def one_client(name: str, audio: Path):
    t0 = time.time()
    data = audio.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    digest = md5(audio)
    async with websockets.connect(SERVER_URL, max_size=500 * 1024 * 1024) as ws:
        connected = json.loads(await ws.recv())
        assert connected["type"] == "connected"
        await ws.send(json.dumps({
            "type": "upload_request",
            "data": {
                "file_name": audio.name, "file_size": len(data),
                "file_hash": digest, "force_refresh": True,
                "output_format": "json",
            }
        }))
        resp = json.loads(await ws.recv())
        if resp["type"] == "error":
            return {"name": name, "ok": False, "err": resp["data"]["message"]}
        task_id = resp["data"]["task_id"]
        await ws.send(json.dumps({
            "type": "upload_data",
            "data": {"task_id": task_id, "file_data": b64}
        }))
        last_progress_print = 0
        while True:
            msg = json.loads(await ws.recv())
            t = msg["type"]
            if t == "task_progress":
                p = msg["data"].get("progress", 0)
                now = time.time()
                if now - last_progress_print > 30:
                    print(f"  [{name}] progress {p:.0f}% @ wall {now - t0:.0f}s", flush=True)
                    last_progress_print = now
                continue
            if t == "task_complete":
                r = msg["data"]["result"]
                wall = time.time() - t0
                return {
                    "name": name, "ok": True, "wall": wall,
                    "duration": r.get("duration", 0),
                    "processing_time": r.get("processing_time", 0),
                    "speakers": r.get("speakers", []),
                    "segments_count": len(r.get("segments", [])),
                    "task_id": task_id,
                }
            if t == "error":
                return {"name": name, "ok": False, "err": msg["data"]["message"], "task_id": task_id}


async def main():
    assert AUDIO_1SPK.exists(), AUDIO_1SPK
    assert AUDIO_4SPK.exists(), AUDIO_4SPK

    # 启 server
    env = os.environ.copy()
    env.update({
        "FUNASR_DEFAULT_ENGINE": "qwen3",
        "FUNASR_QWEN3_POOL_SIZE": "2",
        "FUNASR_SERVER_PORT": str(SERVER_PORT),
        "FUNASR_SERVER_HOST": "127.0.0.1",
        "TMPDIR": "/tmp",
        "DYLD_LIBRARY_PATH": str(PROJECT_ROOT / "src/core/vendor/qwen_asr_gguf/inference/bin"),
    })
    log_path = Path("/tmp/qwen3_long_concurrent_server.log")
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "run_server.py"],
        cwd=str(PROJECT_ROOT), env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    print(f"[main] server 启动中 (PID {proc.pid}, port {SERVER_PORT}, log {log_path})...", flush=True)

    try:
        if not wait_port("127.0.0.1", SERVER_PORT, 120.0):
            raise RuntimeError(f"server 120s 内未起来, 看 {log_path}")
        print(f"[main] server port ready, sleep 15s 等 worker pool 加载...", flush=True)
        await asyncio.sleep(15)
        print(f"[main] 启动 2 个并发 client: 1spk={AUDIO_1SPK.name} (16min), 4spk={AUDIO_4SPK.name} (44min)", flush=True)

        t0 = time.time()
        r1, r2 = await asyncio.gather(
            one_client("1spk-16min", AUDIO_1SPK),
            one_client("4spk-44min", AUDIO_4SPK),
        )
        total = time.time() - t0

        print(f"\n========== 真并发 e2e 结果 ==========")
        print(f"并发 2 task wall: {total:.1f}s ({total/60:.1f}min)")
        for r in (r1, r2):
            if not r["ok"]:
                print(f"  [{r['name']}] FAIL: {r.get('err')}")
                continue
            rtf = r["processing_time"] / r["duration"] if r["duration"] else 0
            print(
                f"  [{r['name']}] wall={r['wall']:.1f}s ({r['wall']/60:.1f}min) "
                f"server_time={r['processing_time']:.1f}s RTF={rtf:.3f} "
                f"speakers={len(r['speakers'])} segments={r['segments_count']}"
            )
        # 串行 baseline 推算
        if r1["ok"] and r2["ok"]:
            est_serial = r1["wall"] + r2["wall"]
            ratio = total / max(r1["wall"], r2["wall"])
            print(f"\n并发 ratio = {ratio:.2f}x slowest serial (理论上限 == 单 task max wall)")
            print(f"估算节省 = {est_serial - total:.1f}s ({(est_serial - total)/est_serial*100:.0f}%)")
    finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
        log_fh.close()
        print(f"[main] server 已停, 日志保留 in {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
