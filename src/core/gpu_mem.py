"""GPU 显存探针 — word_align 显存安全课题 (TODOS #17/#18) 的共享原语.

Lane 1 (#17) preflight gate 与 Lane 2 (#18) sidecar router 共用本模块:
- free_vram_mib(): best-effort 读「当前进程使用的 CUDA 卡」剩余显存 (MiB).
- has_headroom(): 纯函数门控, probe 不到时不误杀 (交给 OOM fallback).

设计定案见 docs/开发/gpu加速/2026-06-16-word-align显存安全-评审定案与落地计划.md.

为什么只用 nvidia-smi 不引 NVML (决策 A2):
  探针是冷路径 (每 word_align 请求才调一次, 后面对齐要跑多秒), 子进程
  ~50-100ms 开销可忽略. NVML 是为零收益花一个依赖创新额度, 且要维护
  「import 失败退化 nvidia-smi」双路径. nvidia-smi 已在用 (旧
  _gpu_mem_used_mib 迁来此处), 精度对 GB 级阈值绰绰有余.

为什么尊重 CUDA_VISIBLE_DEVICES (codex #12):
  nvidia-smi 不设 -i 时返回所有物理卡 (多行), 读死第一行会在多卡机
  量到错卡. CUDA_VISIBLE_DEVICES 的首个 token 是进程 CUDA device 0
  对应的物理卡, 用它做 nvidia-smi -i 才量到进程真正用的那张卡.

⚠️ preflight 不替代 OOM fallback (codex #11, TOCTOU): 探完到建 session
  之间, 同卡其它进程 (CapsWriter / ASR) 仍可能吃掉显存. 本探针只挡
  「明显不够」, OOM → poison → CPU 兜底永远保留.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from loguru import logger

# nvidia-smi 查询超时 (秒). 卡死时宁可返回 None 走兜底, 不阻塞请求.
_NVIDIA_SMI_TIMEOUT_SEC = 5


def _resolve_device_index() -> str:
    """解析 nvidia-smi -i 的目标卡索引 (尊重 CUDA_VISIBLE_DEVICES, codex #12).

    - CUDA_VISIBLE_DEVICES 设了非空 → 取首个 token (索引或 GPU-UUID):
      进程的 CUDA device 0 = 该物理卡, 量它才对.
    - 未设 / 空 → 默认 "0" (单卡机 / 显式 GPU 0).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        first = cvd.split(",")[0].strip()
        if first:
            return first
    return "0"


def free_vram_mib(device: Optional[str] = None) -> Optional[int]:
    """读当前进程使用的 CUDA 卡剩余显存 (MiB); 探不到返回 None (不抛).

    Args:
        device: 显式覆盖 nvidia-smi -i 目标 (索引或 GPU-UUID); 缺省按
            CUDA_VISIBLE_DEVICES 解析.

    Returns:
        free MiB (int), 或 None — 当 nvidia-smi 不可用 (非 CUDA 机 / 不在
        PATH) / 超时 / 非零退出 / 输出非数字时. None 语义是「未测」, 调用方
        据此「不误杀」(has_headroom).
    """
    idx = device if device is not None else _resolve_device_index()
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(idx),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception as exc:  # 非 CUDA / 无 nvidia-smi / 超时 / 解析失败 → 未测
        logger.debug(f"free_vram_mib 探测失败 (返回 None, 走不误杀): {exc}")
        return None


def process_vram_mib(pid: int) -> Optional[int]:
    """读指定进程在当前 CUDA 卡上占用的显存 (MiB); 探不到返回 None.

    PID 成功查询但不在 nvidia-smi 占用列表中时返回 0. 设备选择、5 秒超时
    和失败语义与 free_vram_mib 保持一致, 供 sidecar preflight 加回其自身占用.
    """
    idx = _resolve_device_index()
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(idx),
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC,
            check=True,
        )
        usage_mib = 0
        for row in out.stdout.splitlines():
            if not row.strip():
                continue
            row_pid, used_mib = row.split(",", maxsplit=1)
            if int(row_pid.strip()) == pid:
                usage_mib += int(used_mib.strip())
        return usage_mib
    except Exception as exc:  # 探针失败 → 未测, 调用方按既定门控语义处理
        logger.debug(f"[GPU-MEM-PID-PROBE-FAILED] process_vram_mib 探测失败 (返回 None): {exc}")
        return None


def used_vram_mib(device: Optional[str] = None) -> Optional[int]:
    """读当前进程使用的 CUDA 卡已用显存 (MiB); 探不到返回 None (不抛).

    与 free_vram_mib 同源 (nvidia-smi + 设备选择), 只查 memory.used. 给
    word_align poison 时 dispose 前后打 VRAM delta 观测用 (不当显存回收保证).
    """
    idx = device if device is not None else _resolve_device_index()
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(idx),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception as exc:
        logger.debug(f"used_vram_mib 探测失败 (返回 None): {exc}")
        return None


def has_headroom(free_mib: Optional[int], required_mib: int) -> bool:
    """门控: 当前显存是否够加载/路由一个 CUDA word_align session.

    Args:
        free_mib: free_vram_mib() 的结果 (None = 探不到).
        required_mib: 要求的最小空闲显存阈值 (MiB).

    Returns:
        - free_mib is None (探不到) → True: 不误杀, 让请求照常试 CUDA, 失败
          再走 OOM poison + CPU fallback (codex #11).
        - free_mib >= required_mib → True.
        - free_mib < required_mib → False: 显存明显不够, 走 CPU (不等 OOM).
    """
    if free_mib is None:
        return True
    return free_mib >= required_mib
