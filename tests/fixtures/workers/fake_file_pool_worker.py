"""真实子进程 fixture：一次只消费一个任务，用于验证文件池生命周期。"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path


def _write_capture(task_dir: Path, task_file: Path, task: dict) -> None:
    capture_path = os.environ.get("FAKE_POOL_CAPTURE")
    if not capture_path:
        return
    capture = {
        "argv": sys.argv,
        "pid": os.getpid(),
        "task_bytes": task_file.read_bytes().hex(),
        "task": task,
        "darwin_suffix": os.environ.get("DIRHELPER_USER_DIR_SUFFIX"),
    }
    Path(capture_path).write_text(json.dumps(capture), encoding="utf-8")


def _publish_result(result_file: Path, data: dict) -> None:
    temporary = result_file.with_name(
        f".{result_file.name}.{os.getpid()}.tmp"
    )
    with temporary.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, result_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--task-dir", required=True)
    args = parser.parse_args()
    task_dir = Path(args.task_dir)
    ready_file = task_dir / f"worker_{args.worker_id}.ready"
    ready_file.write_text(str(os.getpid()), encoding="utf-8")
    mode = os.environ.get("FAKE_POOL_MODE", "success")
    delays = json.loads(os.environ.get("FAKE_POOL_DELAYS", "{}"))

    while True:
        if (task_dir / f"worker_{args.worker_id}.stop").exists():
            ready_file.unlink(missing_ok=True)
            return
        task_files = sorted(task_dir.glob(f"worker_{args.worker_id}_*.task"))
        if not task_files:
            time.sleep(0.01)
            continue

        task_file = task_files[0]
        task = json.loads(task_file.read_bytes())
        _write_capture(task_dir, task_file, task)
        if mode == "hang":
            while True:
                time.sleep(1)
        if mode == "crash":
            os._exit(17)

        source_name = Path(task.get("source_audio_path", "")).name
        time.sleep(float(delays.get(source_name, 0)))
        result_task_id = task["task_id"]
        result_pid = os.getpid()
        if mode == "wrong_task":
            result_task_id = "wrong-task-id"
        if mode == "wrong_pid":
            result_pid = os.getpid() + 1
        result_file = task_file.with_suffix(".pkl")
        _publish_result(
            result_file,
            {
                "task_id": result_task_id,
                "success": True,
                "result": {
                    "task_id": task["task_id"],
                    "worker_id": args.worker_id,
                    "worker_pid": result_pid,
                    "source_basename": source_name,
                },
                "worker_pid": result_pid,
            },
        )
        task_file.unlink(missing_ok=True)
        ready_file.unlink(missing_ok=True)
        return


if __name__ == "__main__":
    main()
