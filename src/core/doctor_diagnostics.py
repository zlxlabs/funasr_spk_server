"""Pure, explicit-input diagnostics used by the read-only doctor CLI."""
from __future__ import annotations

import stat
from pathlib import Path
from typing import Mapping, Sequence


def describe_doctor_artifact(path: str | Path) -> dict[str, object]:
    """Describe one local artifact without exposing its path or reading content."""
    try:
        info = Path(path).stat()
    except (FileNotFoundError, NotADirectoryError):
        return {"exists": False, "type": "missing", "size": 0}
    except OSError:
        return {"exists": False, "type": "unreadable", "size": 0}
    if stat.S_ISREG(info.st_mode):
        kind = "file"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    else:
        kind = "other"
    return {"exists": True, "type": kind, "size": info.st_size}


def _provider_name(configured: str, runtime: str) -> str | None:
    value = configured.strip().lower()
    if value == "auto":
        value = {"cuda": "cuda", "mac_ane": "coreml"}.get(runtime, "cpu")
    return {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "coreml_ane_fe": "CoreMLExecutionProvider",
        "coreml_ane_full": "CoreMLExecutionProvider",
    }.get(value)


def _artifact_usable(artifact: Mapping[str, object]) -> bool:
    return bool(
        artifact.get("exists")
        and artifact.get("type") in {"file", "directory"}
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
) -> dict[str, object]:
    """Build safe diagnostics from explicit values; never reads global config."""
    available = list(dict.fromkeys(str(item) for item in available_providers))
    effective = _provider_name(configured_provider, runtime)
    provider_fallback = effective is None or effective not in available
    errors: list[str] = []
    warnings: list[str] = []
    if provider_fallback:
        errors.append("configured_provider_unavailable")

    qwen = {name: dict(value) for name, value in qwen_artifacts.items()}
    if engine == "qwen3":
        for name, artifact in qwen.items():
            if not _artifact_usable(artifact):
                errors.append(f"qwen_artifact_unavailable:{name}")
    dynamic = funasr_dynamic_cache if funasr_dynamic_cache in {"unknown", "deferred"} else "unknown"
    if engine == "funasr":
        warnings.append(f"funasr_dynamic_cache_{dynamic}")
    word_align = dict(word_align_artifact)
    if not _artifact_usable(word_align):
        warnings.append("optional_word-align_unavailable")
    status = "error" if errors else "warning" if warnings else "ok"
    return {
        "status": status,
        "engine": engine,
        "runtime": runtime,
        "provider": {
            "configured": configured_provider,
            "effective": effective or "unavailable",
            "available": available,
            "fallback_would_occur": provider_fallback,
        },
        "fallback_would_occur": provider_fallback,
        "artifacts": {
            "qwen": qwen,
            "funasr": {"dynamic_cache": dynamic},
            "word_align": word_align,
        },
        "warnings": warnings,
        "errors": errors,
    }
