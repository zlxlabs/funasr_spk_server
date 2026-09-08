"""跨进程结果发布：先完整写入临时文件，再原子替换目标文件。"""
from __future__ import annotations
import os
import json
import pickle
import uuid
from pathlib import Path
from typing import Any
def publish_pickle_result(result_path: str | Path, payload: Any) -> None:
    """发布完整 pickle bytes；消费者永远只观察最终路径。"""
    destination = Path(result_path)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
def publish_json_result(result_path: str | Path, payload: Any) -> None:
    """发布完整 JSON bytes；保留 FunASR 的非 pickle 传输契约。"""
    destination = Path(result_path)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
def publish_text_marker(marker_path: str | Path, content: str) -> None:
    """原子发布 ready marker，避免消费者看到空内容。"""
    destination = Path(marker_path)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
