"""Pure, explicit-input diagnostics used by the read-only doctor CLI."""
from __future__ import annotations

import copy
import stat
from pathlib import Path
from typing import Mapping, Sequence

from src.core.config_profiles import PROFILES


_DEFAULTS = {
    "transcription": {"default_engine": "funasr"},
    "qwen3": {
        "asr_model_dir": "./models/qwen3_diarize/Qwen3-ASR-1.7B",
        "segmentation_model": "./models/qwen3_diarize/sherpa/pyannote-segmentation-3.0/model.onnx",
        "embedding_model": "./models/qwen3_diarize/sherpa/nemo-titanet-small/embedding.onnx",
        "word_align_model_path": "./models/qwen3_diarize/ctc_forced_aligner/model.onnx",
        "asr_encoder_provider": "auto",
    },
}
_ENV_FIELDS = {
    "FUNASR_DEFAULT_ENGINE": ("transcription", "default_engine"),
    "FUNASR_QWEN3_ASR_ENCODER_PROVIDER": ("qwen3", "asr_encoder_provider"),
    "FUNASR_QWEN3_ASR_MODEL_DIR": ("qwen3", "asr_model_dir"),
    "FUNASR_QWEN3_SEGMENTATION_MODEL": ("qwen3", "segmentation_model"),
    "FUNASR_QWEN3_EMBEDDING_MODEL": ("qwen3", "embedding_model"),
    "FUNASR_QWEN3_WORD_ALIGN_MODEL_PATH": ("qwen3", "word_align_model_path"),
}
_SUPPORTED_ENGINES = {"funasr", "qwen3"}
_SUPPORTED_RUNTIMES = {"cpu", "cuda", "mac_ane"}


def _merge_mapping(base: dict[str, object], override: Mapping[str, object]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge_mapping(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = copy.deepcopy(value)


def resolve_doctor_config(
    config_data: object,
    dotenv_values: Mapping[str, object],
    process_env: Mapping[str, object],
) -> dict[str, object]:
    """Resolve doctor inputs using defaults < config < profile < environment."""
    errors: list[str] = []
    warnings: list[str] = []
    resolved = copy.deepcopy(_DEFAULTS)
    if isinstance(config_data, Mapping):
        for section_name in ("transcription", "qwen3"):
            section = config_data.get(section_name, {})
            if not isinstance(section, Mapping):
                errors.append(f"configuration_section_invalid:{section_name}")
            else:
                _merge_mapping(resolved[section_name], section)  # type: ignore[arg-type]
    else:
        errors.append("configuration_root_invalid")

    environment = dict(dotenv_values)
    environment.update(process_env)
    profile_name = environment.get("FUNASR_PROFILE", "")
    profile_text = profile_name.strip().lower() if isinstance(profile_name, str) else ""
    if profile_text:
        profile = PROFILES.get(profile_text)
        if profile is None:
            warnings.append("unknown_profile")
        else:
            _merge_mapping(resolved, profile)
    for env_name, (section_name, field_name) in _ENV_FIELDS.items():
        if env_name in environment:
            resolved[section_name][field_name] = environment[env_name]  # type: ignore[index]

    transcription = resolved.get("transcription")
    qwen3 = resolved.get("qwen3")
    transcription = transcription if isinstance(transcription, Mapping) else {}
    qwen3 = qwen3 if isinstance(qwen3, Mapping) else {}
    engine_value = transcription.get("default_engine")
    if not isinstance(engine_value, str) or not engine_value.strip():
        errors.append("invalid_engine_type")
        engine = "unknown"
    else:
        engine = engine_value.strip().lower()
        if engine not in _SUPPORTED_ENGINES:
            errors.append("unsupported_engine")
            engine = "unknown"

    provider_value = qwen3.get("asr_encoder_provider")
    if engine == "qwen3":
        if not isinstance(provider_value, str) or not provider_value.strip():
            errors.append("invalid_provider_type")
            provider = "unknown"
        else:
            provider = provider_value.strip().lower()
    else:
        provider = "not_applicable"

    qwen_paths: dict[str, object] = {}
    for field_name in ("asr_model_dir", "segmentation_model", "embedding_model", "word_align_model_path"):
        value = qwen3.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"invalid_artifact_path:{field_name}")
            qwen_paths[field_name] = ""
        else:
            qwen_paths[field_name] = value

    runtime_value = environment.get("FUNASR_RUNTIME", "")
    runtime_override = runtime_value.strip().lower() if isinstance(runtime_value, str) else ""
    if runtime_override and runtime_override not in _SUPPORTED_RUNTIMES:
        errors.append("unsupported_runtime")
        runtime_override = ""
    return {
        "engine": engine,
        "provider": provider,
        "runtime_override": runtime_override,
        "qwen_paths": qwen_paths,
        "errors": errors,
        "warnings": warnings,
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


def _provider_name(configured: str, runtime: str, platform_name: str) -> str | None:
    value = configured.strip().lower()
    if value == "auto":
        value = "coreml_ane_fe" if platform_name == "darwin" else "cpu"
    return {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "coreml_ane_fe": "CoreMLExecutionProvider",
        "coreml_ane_full": "CoreMLExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "trt": "TensorrtExecutionProvider",
        "dml": "DmlExecutionProvider",
    }.get(value)


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
        effective = _provider_name(configured_provider, runtime, platform_name)
        safe_configured = configured_provider.strip().lower() if configured_provider.strip() else "unknown"
        provider_fallback = effective is None or effective not in available
        if provider_fallback:
            errors.append("configured_provider_unavailable")
        provider = {"configured": safe_configured if effective is not None else "unknown", "effective": effective or "unavailable", "available": available, "fallback_would_occur": provider_fallback}
        expected = {"asr_model_dir": "directory", "segmentation_model": "file", "embedding_model": "file"}
        if configured_provider.strip().lower() == "coreml_ane_full":
            expected["backend_mlpackage"] = "directory"
        for name, kind in expected.items():
            artifact = qwen.get(name, {"exists": False, "type": "missing", "size": 0})
            if not _artifact_usable(artifact, kind):
                errors.append(f"qwen_artifact_unavailable:{name}")
                if name == "backend_mlpackage":
                    provider_fallback = True
                    provider["fallback_would_occur"] = True

    word_align = dict(word_align_artifact)
    if word_align.get("type") == "invalid":
        errors.append("invalid_artifact_path:word_align_model_path")
    elif not _artifact_usable(word_align, "file"):
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
