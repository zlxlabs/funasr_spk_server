#!/usr/bin/env python3
"""Strict read-only provider and artifact diagnostics for this repository."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from src.core.doctor_diagnostics import (  # noqa: E402
    build_doctor_report,
    describe_doctor_artifact,
)


def _probe_config() -> tuple[dict[str, object], list[str]]:
    """在隔离子进程复用真实 Config，父进程只接受白名单字段。"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["FUNASR_DOCTOR_CONFIG_PROBE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.core.doctor_config_probe", str(ROOT / "config.json")],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"engine": "unknown", "provider": "unknown", "qwen_paths": {}, "runtime_override": ""}, [
            "configuration_unavailable"
        ]

    try:
        lines = result.stdout.splitlines()
        payload = json.loads(lines[0]) if len(lines) == 1 else {}
    except (json.JSONDecodeError, IndexError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if result.returncode == 0 and payload.get("status") == "ok":
        paths = payload.get("qwen_paths")
        if (
            isinstance(payload.get("engine"), str)
            and isinstance(payload.get("provider"), str)
            and isinstance(paths, dict)
        ):
            return {
                "engine": payload["engine"],
                "provider": payload["provider"],
                "qwen_paths": paths,
                "runtime_override": payload.get("runtime_override", ""),
            }, []
    error = payload.get("error")
    safe_error = (
        error if error in {"configuration_unavailable", "configuration_invalid"} else "configuration_unavailable"
    )
    return {"engine": "unknown", "provider": "unknown", "qwen_paths": {}, "runtime_override": ""}, [safe_error]


def _runtime(available: list[str], forced: str) -> str:
    if forced in {"cpu", "cuda", "mac_ane"}:
        return forced
    return (
        "cuda"
        if "CUDAExecutionProvider" in available and platform.system() == "Linux"
        else "mac_ane"
        if platform.system() == "Darwin"
        else "cpu"
    )


def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except (ImportError, OSError, RuntimeError):
        return []


def _path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _diagnose() -> tuple[dict[str, object], int]:
    effective, errors = _probe_config()
    warnings: list[str] = []
    available = _available_providers()
    paths = effective["qwen_paths"]
    assert isinstance(paths, dict)
    qwen_artifacts = (
        {
            name: describe_doctor_artifact(_path(value))
            for name, value in paths.items()
            if name != "word_align_model_path"
        }
        if effective["engine"] == "qwen3"
        else {}
    )
    provider = str(effective["provider"])
    if effective["engine"] == "qwen3" and provider == "coreml_ane_full":
        model_dir = _path(paths.get("asr_model_dir"))
        qwen_artifacts["backend_mlpackage"] = describe_doctor_artifact(
            model_dir / "qwen3_asr_encoder_backend.mlpackage" if model_dir is not None else None
        )
    report = build_doctor_report(
        engine=str(effective["engine"]),
        runtime=_runtime(available, str(effective.get("runtime_override", ""))),
        configured_provider=provider,
        available_providers=available,
        qwen_artifacts=qwen_artifacts,
        funasr_dynamic_cache="deferred",
        word_align_artifact=(
            describe_doctor_artifact(_path(paths.get("word_align_model_path")))
            if effective["engine"] == "qwen3"
            else {"exists": False, "type": "inactive", "size": 0}
        ),
        platform_name=platform.system().lower(),
        config_errors=errors,
        config_warnings=warnings,
    )
    code = {"ok": 0, "warning": 1, "error": 2}[str(report["status"])]
    return report, code


def main(argv: list[str] | None = None) -> int:
    if (argv if argv is not None else sys.argv[1:]) != ["--json"]:
        print("doctor: expected --json", file=sys.stderr)
        return 2
    report, code = _diagnose()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(f"doctor: status={report['status']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
