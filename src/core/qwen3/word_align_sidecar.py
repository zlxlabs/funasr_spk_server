"""word_align CUDA sidecar (TODOS #18) — 长驻独立进程, idle TTL 自杀真正释放 VRAM.

为什么要 sidecar (评审定案):
  word_align 的 CUDA ONNX session 一旦加载, 显存被 ORT BFCArena 高水位缓着不还,
  唯进程退出可释放. 把 CUDA word_align 拆到独立长驻进程, 空闲 idle_ttl 后自杀,
  主 ASR 服务长期 baseline 不被偶发的词级时间戳请求顶高.

架构 (锁定决策):
  - A1: 长驻进程 + idle TTL + **Unix domain socket** request/response (stdlib, 零依赖).
    传 audio_path + chunks JSON (不传字节, sidecar 自己读文件).
  - A3: sidecar **只做 CUDA** (单一职责); CPU 兜底留主进程. 超时主进程杀 sidecar 再 CPU.
  - A4: 仅 cuda runtime 启用 (Mac worker 即退 / CPU 不碰显存, 不需要).
  - codex #6: client 是**进程全局单例** (pool>1 也只一个 CUDA session).
  - codex #8: CUDA OOM → sidecar **退休** (退出进程释放 VRAM), 不永久 poison 主进程.
  - codex #9: sidecar **瘦入口** — 只 import qwen3/word_align.py + 极简音频加载,
    不碰 qwen3_transcriber.py (避免拖入 ASR/diarize 机器).
  - codex #15: socket 路径 **per-PID** (避免 PM2 多实例 / 测试 / 重启撞车).

⚠️ preflight 不替代 OOM fallback (codex #11): sidecar 内 CUDA OOM 仍可能发生 (TOCTOU),
  退休 + 主进程 CPU 兜底永远保留.

本模块分三层 (各自单测):
  1. 协议: send_msg / recv_msg (4 字节大端长度前缀 + JSON body).
  2. server: handle_conn (单请求) + serve (bind UDS + accept loop + idle TTL).
  3. client: WordAlignSidecarClient (主进程侧单例 manager, 见同文件下半).
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

from loguru import logger

# 4 字节大端无符号长度前缀: UDS 是字节流, 必须自带 framing 防半包/粘包.
_LEN = struct.Struct(">I")


# ════════════════════════════════════════════════════════════════════════
# 1. 协议 (length-prefixed JSON over UDS)
# ════════════════════════════════════════════════════════════════════════
def _encode(obj: dict) -> bytes:
    """obj → length-prefixed JSON 字节 (utf-8)."""
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return _LEN.pack(len(body)) + body


def send_msg(sock: socket.socket, obj: dict) -> None:
    """发一条消息 (length-prefixed JSON). sendall 保证整帧发出."""
    sock.sendall(_encode(obj))


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """阻塞读够 n 字节; 对端 EOF (返回空) → None."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    """收一条消息; 对端关闭 / 半截 → None (不抛). socket.timeout 由调用方处理."""
    header = _recv_exact(sock, _LEN.size)
    if header is None:
        return None
    (n,) = _LEN.unpack(header)
    body = _recv_exact(sock, n)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


# ════════════════════════════════════════════════════════════════════════
# 2. server (sidecar 进程侧)
# ════════════════════════════════════════════════════════════════════════
def handle_conn(
    conn: socket.socket,
    *,
    aligner: Any,
    audio_loader: Callable[[str], Any],
) -> bool:
    """处理一个连接上的单个请求, 回响应. 返回 retire 标志 (True=该退休/退出进程).

    请求形态:
      - {"cmd": "ping"} → {"ok": True, "pong": True} (健康检查, 不调 aligner).
      - {"audio_path", "chunks", "language"} → 对齐, 回 {"ok": True, "words", "stats"}.

    错误:
      - CUDA 资源错误 (is_resource_error) → {"ok": False, "resource_error": True, ...}
        + 返回 retire=True (codex #8: OOM 退休, 主进程会杀掉/不再用本 sidecar).
      - 普通对齐错误 → {"ok": False, "resource_error": False, ...}, retire=False.
    """
    from src.core.qwen3.word_align import is_resource_error

    req = recv_msg(conn)
    if req is None:
        return False
    if req.get("cmd") == "ping":
        send_msg(conn, {"ok": True, "pong": True})
        return False
    try:
        audio = audio_loader(req["audio_path"])
        words, stats = aligner.align_chunks(
            audio, req["chunks"], language=req.get("language")
        )
        send_msg(conn, {"ok": True, "words": words, "stats": stats})
        return False
    except Exception as exc:  # 对齐失败: 资源类退休, 普通类继续服务
        resource = is_resource_error(exc)
        send_msg(
            conn,
            {"ok": False, "error": f"{type(exc).__name__}: {exc}", "resource_error": resource},
        )
        if resource:
            logger.warning(f"sidecar CUDA 资源错误, 退休退出释放 VRAM: {exc}")
        return resource


def serve(
    sock_path: str,
    *,
    aligner_factory: Callable[[], Any],
    audio_loader: Callable[[str], Any],
    idle_ttl_sec: float,
) -> None:
    """bind UDS + accept loop. 空闲 idle_ttl_sec 无连接 → 退出 (释放 VRAM).

    aligner 惰性 build (首个真实对齐请求才 factory()), 无请求时不占显存.
    idle TTL 用 accept() 超时实现 (codex #5): 服务中 socket busy, 仅 idle 等连接
    时才计时, 超时即退. lease-free — 客户端连不上 = 进程已退, 自行 respawn.
    """
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            os.unlink(sock_path)  # 清残留 socket 文件
        except FileNotFoundError:
            pass
        srv.bind(sock_path)
        srv.listen(8)  # backlog: 容忍 readiness 探测 + 真实请求短暂并排 (单线程顺序处理)
        srv.settimeout(idle_ttl_sec)
        logger.info(f"word_align sidecar 就绪: {sock_path} (idle_ttl={idle_ttl_sec}s)")

        aligner: Optional[Any] = None
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                logger.info(f"word_align sidecar idle {idle_ttl_sec}s 无请求, 退出释放 VRAM")
                return
            with conn:
                # ping 不需要 aligner; 真实对齐请求才 lazy build
                if aligner is None:
                    # peek 不方便, 直接 build 前先看是不是 ping: handle_conn 内部已分流,
                    # 但 build 在 handle_conn 外做才能保证 ping 不触发 build → 传 lazy factory.
                    aligner = _LazyAligner(aligner_factory)
                retire = handle_conn(conn, aligner=aligner, audio_loader=audio_loader)
            if retire:
                logger.warning("word_align sidecar 退休 (CUDA 资源错误), 退出")
                return
    finally:
        srv.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass


class _LazyAligner:
    """惰性包装: 首个 align_chunks 才真 build CUDA aligner (ping 不触发 build)."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._inner: Optional[Any] = None

    def align_chunks(self, audio, chunks, language=None):
        if self._inner is None:
            self._inner = self._factory()
        return self._inner.align_chunks(audio, chunks, language=language)


# UDS sun_path 上限 (macOS 104 / Linux 108 字节), socket 必须放短路径.
# 固定用 /tmp (macOS /tmp→/private/tmp 仍短; Linux /tmp 短), per-PID 命名避免
# PM2 多实例 / 测试 / 重启撞车 (codex #15). 不用 tempfile.gettempdir() —
# macOS TMPDIR 是 /var/folders/... 深路径, 容易超 104.
_SOCKET_DIR = "/tmp"


def default_socket_path(owner_pid: Optional[int] = None) -> str:
    """主进程的 word_align sidecar UDS 路径 (per-owner-PID, 短路径).

    owner_pid 缺省取当前进程 PID — 主 ASR 服务进程拥有一个 sidecar.
    """
    pid = owner_pid if owner_pid is not None else os.getpid()
    return os.path.join(_SOCKET_DIR, f"funasr_wa_sidecar_{pid}.sock")


def ping(sock_path: str, timeout: float = 0.5) -> bool:
    """对 sidecar 发一次 ping, 收到 pong 返回 True (健康检查). 任何失败 → False."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout)
        s.connect(sock_path)
        send_msg(s, {"cmd": "ping"})
        resp = recv_msg(s)
        return bool(resp and resp.get("pong"))
    except OSError:
        return False
    finally:
        s.close()


def wait_for_socket(sock_path: str, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """轮询等 sidecar 就绪 (ping 通). 完整 ping/pong 握手, 返回时 sidecar 已回到
    accept 等待态 (单线程顺序处理), 避免 readiness 探测占着半个连接. 超时 → False.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(sock_path) and ping(sock_path, timeout=0.5):
            return True
        time.sleep(interval)
    return False


# ════════════════════════════════════════════════════════════════════════
# 3. client (主进程侧 manager, 进程全局单例)
# ════════════════════════════════════════════════════════════════════════
class SidecarUnavailable(Exception):
    """sidecar 起不来 / 连不上 / 重试耗尽 — 调用方降级主进程 CPU."""


class SidecarConnectError(Exception):
    """连接 sidecar 失败 (refused/missing) — 视为 sidecar 已退出, 触发 respawn."""


class SidecarTimeout(Exception):
    """对齐超时 (sidecar 已被杀, 杜绝 CUDA+CPU 双跑 codex #4) — 降级 CPU."""


class SidecarResourceError(Exception):
    """sidecar CUDA 资源错误 (已退休 codex #8) — 降级 CPU."""


class SidecarAlignError(Exception):
    """sidecar 普通对齐错误 (非资源) — sidecar 仍健康, 调用方按无词处理."""


def request(
    sock_path: str,
    req: dict,
    *,
    connect_timeout: float,
    recv_timeout: float,
) -> Optional[dict]:
    """连 sidecar 发一个请求收响应. connect 失败 → SidecarConnectError;
    recv 超时 → socket.timeout (= 对齐太慢). 对端半截 → None.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(connect_timeout)
        try:
            s.connect(sock_path)
        except OSError as exc:  # refused / 文件不存在 / connect 超时 → sidecar 不可达
            raise SidecarConnectError(str(exc)) from exc
        send_msg(s, req)
        s.settimeout(recv_timeout)
        return recv_msg(s)  # socket.timeout 在此抛出 = 对齐超时
    finally:
        s.close()


class WordAlignSidecarClient:
    """word_align CUDA sidecar 的主进程侧 manager (进程全局单例, codex #6).

    生命周期: lazy spawn (首个请求) → 复用 (idle TTL 内) → 死了 respawn →
    超时/资源错误杀掉. 串行化 (一把锁): pool>1 多 worker 也只一个 CUDA sidecar.

    降级 (调用方据异常转主进程 CPU):
      SidecarTimeout / SidecarResourceError / SidecarUnavailable → CPU 兜底.
      SidecarAlignError → 无词 (sidecar 健康, 不必杀).
    """

    def __init__(
        self,
        sock_path: str,
        spawn_cmd: List[str],
        *,
        align_timeout_sec: float = 120.0,
        connect_timeout_sec: float = 2.0,
        ready_timeout_sec: float = 30.0,
    ):
        self._sock_path = sock_path
        self._spawn_cmd = spawn_cmd
        self._align_timeout = align_timeout_sec
        self._connect_timeout = connect_timeout_sec
        self._ready_timeout = ready_timeout_sec
        self._proc: Optional[Any] = None
        self._lock = threading.RLock()

    @property
    def running_pid(self) -> Optional[int]:
        """存活 sidecar 子进程的 PID; 未启动或已退出时返回 None."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return None
        return proc.pid

    def _alive(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
            and ping(self._sock_path, timeout=self._connect_timeout)
        )

    def _kill(self) -> None:
        """杀 sidecar 进程 + 清 socket 文件 (超时/资源错误/respawn/shutdown)."""
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception as exc:  # 杀不掉不致命, 下次 respawn 用新 socket 文件
                logger.warning(f"word_align sidecar kill 失败: {exc}")
        try:
            os.unlink(self._sock_path)
        except OSError:
            pass

    def _ensure_running(self) -> None:
        """sidecar 活着就复用, 否则 (re)spawn 并等就绪. 不就绪 → SidecarUnavailable."""
        if self._alive():
            return
        self._kill()  # 清残留进程/socket
        logger.info(f"启动 word_align sidecar: {' '.join(self._spawn_cmd)}")
        self._proc = subprocess.Popen(self._spawn_cmd)
        if not wait_for_socket(self._sock_path, timeout=self._ready_timeout):
            self._kill()
            raise SidecarUnavailable("sidecar 启动后 socket 未在超时内就绪")

    def align(
        self, audio_path: str, chunks: List[Any], language: Optional[str]
    ) -> Tuple[List[dict], dict]:
        """经 sidecar 出词级时间戳. 失败抛具体异常供调用方降级 (见类 docstring)."""
        req = {"audio_path": audio_path, "chunks": chunks, "language": language}
        with self._lock:
            last_exc: Optional[Exception] = None
            for attempt in range(2):  # 初次 + sidecar 死了 respawn 一次
                self._ensure_running()
                try:
                    resp = request(
                        self._sock_path, req,
                        connect_timeout=self._connect_timeout,
                        recv_timeout=self._align_timeout,
                    )
                except socket.timeout as exc:  # 对齐超时 → 杀 sidecar (codex #4)
                    self._kill()
                    raise SidecarTimeout(f"word_align sidecar 对齐超时 {self._align_timeout}s") from exc
                except (SidecarConnectError, ConnectionError, FileNotFoundError) as exc:
                    last_exc = exc  # sidecar 死了 → respawn 重试
                    self._kill()
                    continue
                if resp is None:  # 半截响应 → 视为死, respawn
                    last_exc = SidecarUnavailable("sidecar 响应半截")
                    self._kill()
                    continue
                if not resp.get("ok"):
                    if resp.get("resource_error"):
                        self._kill()  # OOM 退休 (codex #8), 确保进程已死
                        raise SidecarResourceError(resp.get("error", "cuda resource error"))
                    raise SidecarAlignError(resp.get("error", "align failed"))
                return resp["words"], resp["stats"]
            raise SidecarUnavailable(f"word_align sidecar 重试耗尽: {last_exc}")

    def shutdown(self) -> None:
        """服务停时杀 sidecar 进程 (清理)."""
        with self._lock:
            self._kill()


# ── 进程全局单例 ──────────────────────────────────────────────────────────
_client_singleton: Optional[WordAlignSidecarClient] = None
_client_lock = threading.Lock()


def _build_default_client() -> WordAlignSidecarClient:
    """按 config + runtime 构造默认 sidecar client (spawn_cmd 指向本模块 entry).

    Step 3 (entry point) 落地后补全参数透传; 此处先组 spawn_cmd 骨架.
    """
    import sys

    from src.core.config import config

    cfg = config.qwen3
    sock_path = default_socket_path()
    spawn_cmd = [
        sys.executable, "-m", "src.core.qwen3.word_align_sidecar",
        "--socket", sock_path,
        "--model-path", cfg.word_align_model_path,
        "--language", cfg.word_align_language,
        "--cuda-batch-size", str(cfg.word_align_cuda_batch_size),
        "--idle-ttl", str(cfg.word_align_sidecar_idle_ttl_sec),
    ]
    return WordAlignSidecarClient(
        sock_path, spawn_cmd,
        align_timeout_sec=cfg.word_align_sidecar_align_timeout_sec,
    )


def get_word_align_sidecar_client() -> WordAlignSidecarClient:
    """进程全局单例 (codex #6): pool>1 多 worker 共用一个 CUDA sidecar.

    首次创建时注册 atexit: 主进程退出 → 杀 sidecar 子进程 (避免孤儿占 VRAM;
    idle TTL 已兜底 ≤TTL 秒, atexit 让正常退出立即清). 无 sidecar 时 no-op.
    """
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            import atexit

            _client_singleton = _build_default_client()
            atexit.register(reset_word_align_sidecar_client)
        return _client_singleton


def reset_word_align_sidecar_client() -> None:
    """清单例 (测试用; 先 shutdown 杀进程)."""
    global _client_singleton
    with _client_lock:
        if _client_singleton is not None:
            try:
                _client_singleton.shutdown()
            except Exception:
                pass
        _client_singleton = None


# ════════════════════════════════════════════════════════════════════════
# 4. entry point (sidecar 子进程入口 — 瘦入口, codex #9)
# ════════════════════════════════════════════════════════════════════════
def run_sidecar(argv: Optional[List[str]] = None) -> None:
    """sidecar 进程主入口: 解析 argv → 组 CUDA aligner factory + lean audio_loader → serve.

    **瘦入口** (codex #9): 只 import word_align.py (aligner) + audio_io.py (加载),
    **不碰 qwen3_transcriber / diarize / sherpa**, 避免拖进 ASR/diarize 机器 + 多占显存.
    """
    import argparse

    parser = argparse.ArgumentParser(description="word_align CUDA sidecar")
    parser.add_argument("--socket", required=True, help="UDS socket 路径")
    parser.add_argument("--model-path", required=True, help="MMS CTC-FA ONNX 路径")
    parser.add_argument("--language", default="chi", help="兜底语言 ISO 码")
    parser.add_argument("--cuda-batch-size", type=int, default=1)
    parser.add_argument("--idle-ttl", type=float, default=90.0, help="空闲 TTL 秒, 到点退出释放 VRAM")
    args = parser.parse_args(argv)

    def aligner_factory():
        # CUDA aligner: provider 锁 cuda (sidecar 只做 CUDA, A3)
        from src.core.qwen3.word_align import WordAligner

        return WordAligner(
            model_path=args.model_path,
            provider="cuda",
            language=args.language,
            cuda_batch_size=args.cuda_batch_size,
        )

    def audio_loader(path: str):
        from src.core.qwen3.audio_io import load_audio_mono_16k

        audio_16k, _sr = load_audio_mono_16k(path)
        return audio_16k  # 1D 波形, 剥掉 sample_rate

    logger.info(
        f"word_align sidecar 进程启动 pid={os.getpid()} socket={args.socket} "
        f"model={args.model_path} lang={args.language} idle_ttl={args.idle_ttl}s"
    )
    serve(
        args.socket,
        aligner_factory=aligner_factory,
        audio_loader=audio_loader,
        idle_ttl_sec=args.idle_ttl,
    )


if __name__ == "__main__":
    run_sidecar()
