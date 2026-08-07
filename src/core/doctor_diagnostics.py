"""Pure, explicit-input diagnostics used by the read-only doctor CLI."""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from src.core.config import Config


_SUPPORTED_ENGINES = {"funasr", "qwen3"}
_PROVIDER_NAMES = {
    "auto",
    "cpu",
    "cuda",
    "tensorrt",
    "trt",
    "coreml_ane_fe",
    "coreml_ane_full",
}


def describe_doctor_artifact(path: object) -> dict[str, object]:
    """Describe one local artifact without exposing its path or reading content."""
    if not isinstance(path, (str, Path)) or (isinstance(path, str) and not path.strip()):
        return {"exists": False, "type": "invalid", "size": 0}
    try:
        candidate = Path(path)
        info = candidate.stat()
    except (FileNotFoundError, NotADirectoryError):
        return {"exists": False, "type": "missing", "size": 0}
    except (OSError, TypeError, ValueError):
        return {"exists": False, "type": "unreadable", "size": 0}
    if stat.S_ISREG(info.st_mode):
        kind = "file"
        size = info.st_size
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
        try:
            nonempty = next(candidate.iterdir(), None) is not None
        except OSError:
            nonempty = False
        size = info.st_size if nonempty else 0
    else:
        kind = "other"
        size = 0
    return {"exists": True, "type": kind, "size": size}


def inspect_doctor_directory_target(path: object) -> dict[str, object]:
    """只读判断目录目标是否可由 setup_directories 成功处理。"""
    if not isinstance(path, (str, Path)) or (isinstance(path, str) and not path.strip()):
        return {"exists": False, "type": "invalid", "size": 0, "usable": False, "error": "target_invalid"}

    candidate = Path(path)
    try:
        if os.path.lexists(candidate) and not candidate.exists():
            return {"exists": True, "type": "other", "size": 0, "usable": False, "error": "target_not_directory"}
        if candidate.exists():
            artifact = describe_doctor_artifact(candidate)
            usable = artifact["exists"] and artifact["type"] == "directory"
            return {
                **artifact,
                "usable": usable,
                "error": None if usable else "target_not_directory",
            }
    except OSError:
        return {"exists": False, "type": "unreadable", "size": 0, "usable": False, "error": "target_unreadable"}

    ancestor = candidate.parent
    while True:
        try:
            if ancestor.exists():
                break
        except OSError:
            return {"exists": False, "type": "unreadable", "size": 0, "usable": False, "error": "ancestor_unreadable"}
        parent = ancestor.parent
        if parent == ancestor:
            return {"exists": False, "type": "missing", "size": 0, "usable": False, "error": "ancestor_unavailable"}
        ancestor = parent

    try:
        if not ancestor.is_dir():
            error = "ancestor_not_directory"
        elif not os.access(ancestor, os.W_OK):
            error = "ancestor_not_writable"
        elif not os.access(ancestor, os.X_OK):
            error = "ancestor_not_searchable"
        else:
            error = None
    except OSError:
        error = "ancestor_unreadable"
    return {"exists": False, "type": "missing", "size": 0, "usable": error is None, "error": error}


def doctor_config_snapshot(config: "Config") -> dict[str, object]:
    """提取真实 Config 的安全 provider、工件和目录元数据，不回传原始路径。"""
    artifact_paths = {
        "asr_model_dir": config.qwen3.asr_model_dir,
        "segmentation_model": config.qwen3.segmentation_model,
        "embedding_model": config.qwen3.embedding_model,
    }
    qwen_artifacts = {
        name: describe_doctor_artifact(value) for name, value in artifact_paths.items()
    }
    if config.qwen3.asr_encoder_provider == "coreml_ane_full":
        qwen_artifacts["backend_mlpackage"] = describe_doctor_artifact(
            Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_backend.mlpackage"
        )
    directories = {
        name: inspect_doctor_directory_target(path)
        for name, path in config.config_directory_targets()
    }
    return {
        "engine": config.transcription.default_engine,
        "provider": config.qwen3.asr_encoder_provider or "auto",
        "runtime_override": os.getenv("FUNASR_RUNTIME", "").strip().lower(),
        "artifacts": {
            "qwen": qwen_artifacts,
            "word_align": describe_doctor_artifact(config.qwen3.word_align_model_path),
        },
        "directories": directories,
    }


def _provider_name(
    configured: str,
    runtime: str,
    platform_name: str,
    available: Sequence[str],
) -> tuple[str, str, bool, bool]:
    """Return target, actual EP, fallback flag, and unsupported-config flag."""
    del runtime
    value = configured.strip().lower() or "auto"
    platform_default = "CoreMLExecutionProvider" if platform_name == "darwin" else "CPUExecutionProvider"
    target = {
        "auto": platform_default,
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "trt": "TensorrtExecutionProvider",
        "coreml_ane_fe": "CoreMLExecutionProvider",
        "coreml_ane_full": "CoreMLExecutionProvider",
    }.get(value, platform_default)
    target_available = target in available
    effective = target if target_available else "CPUExecutionProvider" if "CPUExecutionProvider" in available else "unavailable"
    unsupported = value not in _PROVIDER_NAMES
    return target, effective, unsupported or not target_available, unsupported


def _artifact_usable(artifact: Mapping[str, object], expected_type: str) -> bool:
    return bool(
        artifact.get("exists")
        and artifact.get("type") == expected_type
        and int(artifact.get("size", 0)) > 0
    )


def build_doctor_report(
    *,
    engine: str,
    runtime: str,
    configured_provider: str,
    available_providers: Sequence[str],
    qwen_artifacts: Mapping[str, Mapping[str, object]],
    funasr_dynamic_cache: str,
    word_align_artifact: Mapping[str, object],
    platform_name: str = "linux",
    config_errors: Sequence[str] = (),
    config_warnings: Sequence[str] = (),
) -> dict[str, object]:
    """Build safe diagnostics from explicit values; never reads global config."""
    available = list(dict.fromkeys(str(item) for item in available_providers))
    errors = list(config_errors)
    warnings = list(config_warnings)
    qwen = {name: dict(value) for name, value in qwen_artifacts.items()}
    if engine not in _SUPPORTED_ENGINES:
        if "unsupported_engine" not in errors:
            errors.append("unsupported_engine")
        provider = {"configured": "unknown", "effective": "unavailable", "available": available, "fallback_would_occur": False}
        provider_fallback = False
    elif engine == "funasr":
        warnings.append(f"funasr_dynamic_cache_{funasr_dynamic_cache if funasr_dynamic_cache in {'unknown', 'deferred'} else 'unknown'}")
        provider = {"configured": "not_applicable", "effective": "not_applicable", "available": available, "fallback_would_occur": False}
        provider_fallback = False
    else:
        target, effective, provider_fallback, unsupported = _provider_name(
            configured_provider, runtime, platform_name, available
        )
        configured_value = configured_provider.strip().lower()
        safe_configured = configured_value if configured_value in _PROVIDER_NAMES | {"dml", "coreml"} else "unknown"
        if unsupported:
            errors.append("unsupported_provider")
        if target not in available:
            errors.append("configured_provider_unavailable")
        provider = {"configured": safe_configured, "effective": effective, "available": available, "fallback_would_occur": provider_fallback}
        expected = {"asr_model_dir": "directory", "segmentation_model": "file", "embedding_model": "file"}
        if configured_value == "coreml_ane_full":
            expected["backend_mlpackage"] = "directory"
        for name, kind in expected.items():
            artifact = qwen.get(name, {"exists": False, "type": "missing", "size": 0})
            if not _artifact_usable(artifact, kind):
                errors.append(f"qwen_artifact_unavailable:{name}")
                if name == "backend_mlpackage":
                    provider_fallback = True
                    provider["fallback_would_occur"] = True

    word_align = dict(word_align_artifact)
    if engine == "qwen3" and not _artifact_usable(word_align, "file"):
        warnings.append("optional_word-align_unavailable")
    status = "error" if errors else "warning" if warnings else "ok"
    return {
        "status": status,
        "engine": engine,
        "runtime": runtime,
        "provider": provider,
        "fallback_would_occur": provider_fallback,
        "artifacts": {
            "qwen": qwen,
            "funasr": {"dynamic_cache": funasr_dynamic_cache if funasr_dynamic_cache in {"unknown", "deferred"} else "unknown"},
            "word_align": word_align,
        },
        "warnings": list(dict.fromkeys(warnings)),
        "errors": list(dict.fromkeys(errors)),
    }
