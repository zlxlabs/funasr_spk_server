"""隔离加载真实 Config 的只读 doctor 子进程入口。"""
from __future__ import annotations

import json
import sys
from typing import Sequence


sys.dont_write_bytecode = True


def run_config_probe(config_path: str) -> int:
    """加载并校验真实 Config，只返回不含原始输入的安全 JSON。"""
    try:
        from pydantic import ValidationError
        from src.core.config import Config, ConfigFileUnavailableError
        from src.core.doctor_diagnostics import doctor_config_snapshot
    except (ImportError, OSError):
        print(json.dumps({"status": "error", "error": "configuration_unavailable"}))
        return 2

    try:
        config = Config.load_from_file(config_path, strict=True)
        snapshot = doctor_config_snapshot(config)
    except ConfigFileUnavailableError:
        print(json.dumps({"status": "error", "error": "configuration_unavailable"}))
        return 2
    except (OSError, UnicodeDecodeError):
        print(json.dumps({"status": "error", "error": "configuration_unavailable"}))
        return 2
    except SystemExit:
        print(json.dumps({"status": "error", "error": "configuration_invalid"}))
        return 2
    except ValidationError:
        print(json.dumps({"status": "error", "error": "configuration_invalid"}))
        return 2
    except (TypeError, ValueError):
        print(json.dumps({"status": "error", "error": "configuration_invalid"}))
        return 2

    directory_metadata = snapshot["directories"]
    assert isinstance(directory_metadata, dict)
    has_directory_error = any(not metadata["usable"] for metadata in directory_metadata.values())
    status = "error" if has_directory_error else "ok"
    error = "directory_unavailable" if has_directory_error else None
    print(
        json.dumps(
            {"status": status, "error": error, **snapshot},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if has_directory_error:
        return 2
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
