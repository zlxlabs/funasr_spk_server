"""隔离加载真实 Config 的只读 doctor 子进程入口。"""
from __future__ import annotations

import json
import os
import sys
from typing import Sequence


sys.dont_write_bytecode = True
os.environ["FUNASR_DOCTOR_CONFIG_PROBE"] = "1"
os.environ["FUNASR_WORKER_MODE"] = "1"


def run_config_probe(config_path: str) -> int:
    """加载并校验真实 Config，只返回不含原始输入的安全 JSON。"""
    try:
        from src.core.config import Config, ConfigFileUnavailableError
        from src.core.doctor_diagnostics import doctor_config_snapshot
    except Exception:
        print(json.dumps({"status": "error", "error": "configuration_unavailable"}))
        return 2

    try:
        config = Config.load_from_file(config_path, strict=True)
        snapshot = doctor_config_snapshot(config)
    except ConfigFileUnavailableError:
        print(json.dumps({"status": "error", "error": "configuration_unavailable"}))
        return 2
    except SystemExit:
        print(json.dumps({"status": "error", "error": "configuration_invalid"}))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error": "configuration_invalid"}))
        return 2

    print(json.dumps({"status": "ok", **snapshot}, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """执行内部 probe；参数只接受一个配置文件路径。"""
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print(json.dumps({"status": "error", "error": "configuration_unavailable"}))
        return 2
    return run_config_probe(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
