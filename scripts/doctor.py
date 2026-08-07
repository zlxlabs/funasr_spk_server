#!/usr/bin/env python3
"""Strict read-only provider and artifact diagnostics for this repository."""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from src.core.doctor_diagnostics import (  # noqa: E402
    build_doctor_report,
    describe_doctor_artifact,
    resolve_doctor_config,
)


def _read_config() -> tuple[object, list[str]]:
    try:
        with (ROOT / "config.json").open(encoding="utf-8") as stream:
            return json.load(stream), []
    except (OSError, json.JSONDecodeError):
        return {}, ["configuration_unavailable"]


def _read_dotenv() -> tuple[dict[str, object], list[str]]:
    try:
        env_path = ROOT / ".env"
        if not env_path.exists():
            return {}, []
        return dict(dotenv_values(env_path)), []
    except (OSError, ValueError):
        return {}, ["dotenv_unavailable"]


def _runtime(available: list[str], forced: str) -> str:
    if forced in {"cpu", "cuda", "mac_ane"}:
        return forced
    return "cuda" if "CUDAExecutionProvider" in available and platform.system() == "Linux" else "mac_ane" if platform.system() == "Darwin" else "cpu"


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
    config, config_errors = _read_config()
    dotenv_data, dotenv_errors = _read_dotenv()
    effective = resolve_doctor_config(config, dotenv_data, os.environ)
    errors = config_errors + dotenv_errors + list(effective["errors"])
    warnings = list(effective["warnings"])
    available = _available_providers()
    paths = effective["qwen_paths"]
    assert isinstance(paths, dict)
    qwen_artifacts = {name: describe_doctor_artifact(_path(value)) for name, value in paths.items() if name != "word_align_model_path"}
    provider = str(effective["provider"])
    if provider == "coreml_ane_full":
        model_dir = _path(paths.get("asr_model_dir"))
        qwen_artifacts["backend_mlpackage"] = describe_doctor_artifact(
            model_dir / "qwen3_asr_encoder_backend.mlpackage" if model_dir is not None else None
        )
    report = build_doctor_report(
        engine=str(effective["engine"]),
        runtime=_runtime(available, str(effective["runtime_override"])),
        configured_provider=provider,
        available_providers=available,
        qwen_artifacts=qwen_artifacts,
        funasr_dynamic_cache="deferred",
        word_align_artifact=describe_doctor_artifact(_path(paths.get("word_align_model_path"))),
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
