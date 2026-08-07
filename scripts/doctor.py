#!/usr/bin/env python3
"""Strict read-only provider and artifact diagnostics for this repository."""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from src.core.doctor_diagnostics import build_doctor_report, describe_doctor_artifact


def _read_config() -> tuple[dict, list[str]]:
    try:
        with (ROOT / "config.json").open(encoding="utf-8") as stream:
            return json.load(stream), []
    except (OSError, json.JSONDecodeError):
        return {}, ["configuration_unavailable"]


def _section(config: dict, name: str) -> dict:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _setting(section: dict, key: str, env_name: str, default=None):
    return os.environ.get(env_name, section.get(key, default))


def _runtime(available: list[str]) -> str:
    forced = os.environ.get("FUNASR_RUNTIME", "").strip().lower()
    if forced in {"cpu", "cuda", "mac_ane"}:
        return forced
    return "cuda" if "CUDAExecutionProvider" in available and platform.system() == "Linux" else "mac_ane" if platform.system() == "Darwin" else "cpu"


def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except (ImportError, OSError, RuntimeError):
        return []


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _diagnose() -> tuple[dict[str, object], int]:
    config, errors = _read_config()
    transcription = _section(config, "transcription")
    qwen3 = _section(config, "qwen3")
    profile = os.environ.get("FUNASR_PROFILE", "").strip().lower()
    profile_engine = {"cuda_prod": "qwen3", "cuda_dev": "qwen3", "mac_prod": "funasr", "mac_dev": "funasr"}
    engine = os.environ.get("FUNASR_DEFAULT_ENGINE", profile_engine.get(profile, transcription.get("default_engine", "funasr")))
    provider = "cpu" if engine != "qwen3" else _setting(qwen3, "asr_encoder_provider", "FUNASR_QWEN3_ASR_ENCODER_PROVIDER", {"cuda_prod": "cuda", "cuda_dev": "cuda"}.get(profile, "auto"))
    qwen_paths = {
        "asr_model_dir": _setting(qwen3, "asr_model_dir", "FUNASR_QWEN3_ASR_MODEL_DIR", "./models/qwen3_diarize/Qwen3-ASR-1.7B"),
        "segmentation_model": _setting(qwen3, "segmentation_model", "FUNASR_QWEN3_SEGMENTATION_MODEL", "./models/qwen3_diarize/sherpa/pyannote-segmentation-3.0/model.onnx"),
        "embedding_model": _setting(qwen3, "embedding_model", "FUNASR_QWEN3_EMBEDDING_MODEL", "./models/qwen3_diarize/sherpa/nemo-titanet-small/embedding.onnx"),
    }
    word_align = _setting(qwen3, "word_align_model_path", "FUNASR_QWEN3_WORD_ALIGN_MODEL_PATH", "./models/qwen3_diarize/ctc_forced_aligner/model.onnx")
    available = _available_providers()
    report = build_doctor_report(
        engine=str(engine).strip().lower(),
        runtime=_runtime(available),
        configured_provider=str(provider),
        available_providers=available,
        qwen_artifacts={name: describe_doctor_artifact(_path(value)) for name, value in qwen_paths.items()},
        funasr_dynamic_cache="deferred",
        word_align_artifact=describe_doctor_artifact(_path(word_align)),
    )
    report["errors"] = list(report["errors"]) + errors
    if errors:
        report["status"] = "error"
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
