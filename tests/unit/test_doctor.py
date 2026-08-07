"""I4 doctor contract: strict read-only diagnostics and safe JSON output."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.doctor_diagnostics import (
    build_doctor_report,
    describe_doctor_artifact,
    resolve_doctor_config,
)


ROOT = Path(__file__).resolve().parents[2]
DOCTOR = ROOT / "scripts" / "doctor.py"


def _run_doctor(cwd: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "FUNASR_NOTIFICATION_ENABLED": "false",
            "FUNASR_RUNTIME": "cpu",
            "FUNASR_QWEN3_ASR_ENCODER_PROVIDER": "cpu",
            **env_overrides,
        }
    )
    return subprocess.run(
        [sys.executable, str(DOCTOR), "--json"],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )


def _doctor_fixture(tmp_path, config_data, dotenv_text="", process_env=None):
    root = tmp_path / "fixture-repo"
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "core").mkdir(parents=True)
    (root / "src" / "__init__.py").touch()
    (root / "src" / "core" / "__init__.py").touch()
    shutil.copy(DOCTOR, root / "scripts" / "doctor.py")
    shutil.copy(ROOT / "src" / "core" / "doctor_diagnostics.py", root / "src" / "core" / "doctor_diagnostics.py")
    shutil.copy(ROOT / "src" / "core" / "config_profiles.py", root / "src" / "core" / "config_profiles.py")
    (root / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
    (root / ".env").write_text(dotenv_text, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("FUNASR_")}
    env.update({"FUNASR_NOTIFICATION_ENABLED": "false", "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(process_env or {})
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "doctor.py"), "--json"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    return root, result
def test_doctor_artifact_reports_safe_type_and_size(tmp_path):
    missing = describe_doctor_artifact(tmp_path / "missing.onnx")
    assert missing == {"exists": False, "type": "missing", "size": 0}

    empty_file = tmp_path / "empty.onnx"
    empty_file.touch()
    assert describe_doctor_artifact(empty_file) == {
        "exists": True,
        "type": "file",
        "size": 0,
    }

    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"model")
    assert describe_doctor_artifact(model_file) == {
        "exists": True,
        "type": "file",
        "size": 5,
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert describe_doctor_artifact(model_dir)["type"] == "directory"

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "model.pipe"
        os.mkfifo(fifo)
        assert describe_doctor_artifact(fifo) == {
            "exists": True,
            "type": "other",
            "size": 0,
        }


def test_doctor_report_maps_provider_fallback_and_does_not_echo_paths(tmp_path):
    secret = "doctor-secret-token"
    report = build_doctor_report(
        engine="qwen3",
        runtime="cpu",
        configured_provider="cuda",
        available_providers=["CPUExecutionProvider"],
        qwen_artifacts={
            "asr_model_dir": describe_doctor_artifact(tmp_path / secret),
            "segmentation_model": {"exists": True, "type": "file", "size": 4},
            "embedding_model": {"exists": True, "type": "file", "size": 4},
        },
        funasr_dynamic_cache="deferred",
        word_align_artifact={"exists": False, "type": "missing", "size": 0},
    )
    assert report["provider"]["configured"] == "cuda"
    assert report["provider"]["available"] == ["CPUExecutionProvider"]
    assert report["provider"]["fallback_would_occur"] is True
    assert report["fallback_would_occur"] is True
    assert report["status"] == "error"
    assert secret not in json.dumps(report)


def test_doctor_report_equal_provider_has_no_fallback(tmp_path):
    artifact = {"exists": True, "type": "file", "size": 1}
    report = build_doctor_report(
        engine="qwen3",
        runtime="cpu",
        configured_provider="cpu",
        available_providers=["CPUExecutionProvider"],
        qwen_artifacts={"asr": artifact, "seg": artifact, "embed": artifact},
        funasr_dynamic_cache="unknown",
        word_align_artifact=artifact,
    )
    assert report["provider"]["fallback_would_occur"] is False
    assert report["fallback_would_occur"] is False
    assert report["errors"] == []
    assert report["status"] == "ok"


def test_doctor_effective_config_uses_dotenv_profile_then_process_env():
    result = resolve_doctor_config(
        {"transcription": {"default_engine": "funasr"}},
        {"FUNASR_PROFILE": "cuda_dev", "FUNASR_DEFAULT_ENGINE": "funasr"},
        {"FUNASR_DEFAULT_ENGINE": "qwen3"},
    )
    assert result["engine"] == "qwen3"
    assert result["provider"] == "cuda"
    assert result["errors"] == []


@pytest.mark.parametrize(
    ("configured", "available", "effective"),
    [
        ("auto", ["CPUExecutionProvider", "CUDAExecutionProvider"], "CPUExecutionProvider"),
        ("tensorrt", ["TensorrtExecutionProvider"], "TensorrtExecutionProvider"),
        ("trt", ["TensorrtExecutionProvider"], "TensorrtExecutionProvider"),
        ("dml", ["DmlExecutionProvider"], "DmlExecutionProvider"),
    ],
)
def test_doctor_provider_resolution_matches_qwen_supported_paths(configured, available, effective):
    artifact = {"exists": True, "type": "file", "size": 1}
    report = build_doctor_report(
        engine="qwen3",
        runtime="cuda",
        platform_name="linux",
        configured_provider=configured,
        available_providers=available,
        qwen_artifacts={
            "asr_model_dir": {"exists": True, "type": "directory", "size": 1},
            "segmentation_model": artifact,
            "embedding_model": artifact,
        },
        funasr_dynamic_cache="deferred",
        word_align_artifact=artifact,
    )
    assert report["provider"]["effective"] == effective
    assert report["fallback_would_occur"] is False


@pytest.mark.parametrize(
    "config_data",
    [
        [],
        {"transcription": {"default_engine": "bogus"}},
        {"transcription": {"default_engine": ""}},
        {"qwen3": {"word_align_model_path": None}},
    ],
)
def test_doctor_invalid_config_returns_json_exit_two(tmp_path, config_data):
    _, result = _doctor_fixture(tmp_path, config_data)
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert len(result.stdout.splitlines()) == 1
    assert report["status"] == "error"
    assert report["errors"]
    assert "Traceback" not in result.stderr


def test_doctor_dotenv_profile_is_applied_in_subprocess(tmp_path):
    _, result = _doctor_fixture(tmp_path, {}, "FUNASR_PROFILE=cuda_dev\n")
    report = json.loads(result.stdout)
    assert report["engine"] == "qwen3"
    assert report["provider"]["configured"] == "cuda"
    assert result.returncode == 2


def test_doctor_null_provider_is_overridden_by_profile(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {"qwen3": {"asr_encoder_provider": None}},
        "FUNASR_PROFILE=cuda_dev\n",
    )
    report = json.loads(result.stdout)
    assert report["provider"]["configured"] == "cuda"
    assert "Traceback" not in result.stderr


def test_doctor_expected_artifact_types_and_empty_directory_are_errors(tmp_path):
    model_dir = tmp_path / "empty-model"
    model_dir.mkdir()
    onnx = tmp_path / "segmentation.onnx"
    onnx.write_bytes(b"onnx")
    report = build_doctor_report(
        engine="qwen3",
        runtime="cpu",
        platform_name="linux",
        configured_provider="cpu",
        available_providers=["CPUExecutionProvider"],
        qwen_artifacts={
            "asr_model_dir": describe_doctor_artifact(model_dir),
            "segmentation_model": describe_doctor_artifact(model_dir),
            "embedding_model": describe_doctor_artifact(onnx),
        },
        funasr_dynamic_cache="deferred",
        word_align_artifact=describe_doctor_artifact(onnx),
    )
    assert report["status"] == "error"
    assert report["fallback_would_occur"] is False
    assert "qwen_artifact_unavailable:asr_model_dir" in report["errors"]
    assert "qwen_artifact_unavailable:segmentation_model" in report["errors"]


def test_doctor_coreml_full_requires_backend_mlpackage(tmp_path):
    artifact = {"exists": True, "type": "file", "size": 1}
    report = build_doctor_report(
        engine="qwen3",
        runtime="mac_ane",
        platform_name="darwin",
        configured_provider="coreml_ane_full",
        available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        qwen_artifacts={
            "asr_model_dir": {"exists": True, "type": "directory", "size": 1},
            "segmentation_model": artifact,
            "embedding_model": artifact,
            "backend_mlpackage": {"exists": False, "type": "missing", "size": 0},
        },
        funasr_dynamic_cache="deferred",
        word_align_artifact=artifact,
    )
    assert report["fallback_would_occur"] is True
    assert report["status"] == "error"
    assert "qwen_artifact_unavailable:backend_mlpackage" in report["errors"]


@pytest.mark.parametrize("cwd_kind", ["root", "script", "temporary"])
def test_doctor_cli_is_cwd_independent_and_single_json(tmp_path, cwd_kind):
    cwd = {"root": ROOT, "script": DOCTOR.parent, "temporary": tmp_path}[cwd_kind]
    result = _run_doctor(cwd)
    assert result.returncode in {0, 1, 2}
    report = json.loads(result.stdout)
    assert len(result.stdout.splitlines()) == 1
    assert "provider" in report
    assert "fallback_would_occur" in report
    assert "artifacts" in report
    assert "doctor-secret-token" not in result.stdout
    assert result.stderr == "" or "doctor" in result.stderr.lower()


def test_doctor_cli_qwen_artifact_matrix_and_optional_word_align(tmp_path):
    asr_dir = tmp_path / "asr"
    asr_dir.mkdir()
    segmentation = tmp_path / "segmentation.onnx"
    segmentation.write_bytes(b"seg")
    embedding = tmp_path / "embedding.onnx"
    embedding.write_bytes(b"embed")
    result = _run_doctor(
        tmp_path,
        FUNASR_DEFAULT_ENGINE="qwen3",
        FUNASR_QWEN3_ASR_MODEL_DIR=str(asr_dir),
        FUNASR_QWEN3_SEGMENTATION_MODEL=str(segmentation),
        FUNASR_QWEN3_EMBEDDING_MODEL=str(embedding),
        FUNASR_QWEN3_WORD_ALIGN_MODEL_PATH=str(tmp_path / "optional.onnx"),
    )
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["status"] == "warning"
    assert report["artifacts"]["qwen"]["asr_model_dir"]["type"] == "directory"
    assert report["artifacts"]["qwen"]["segmentation_model"]["size"] == 3
    assert report["artifacts"]["word_align"]["exists"] is False
    assert any("word-align" in warning for warning in report["warnings"])

    segmentation.write_bytes(b"")
    failed = _run_doctor(
        tmp_path,
        FUNASR_DEFAULT_ENGINE="qwen3",
        FUNASR_QWEN3_ASR_MODEL_DIR=str(asr_dir),
        FUNASR_QWEN3_SEGMENTATION_MODEL=str(segmentation),
        FUNASR_QWEN3_EMBEDDING_MODEL=str(embedding),
    )
    failed_report = json.loads(failed.stdout)
    assert failed.returncode == 2
    assert failed_report["status"] == "error"
    assert "qwen_artifact_unavailable:segmentation_model" in failed_report["errors"]


def test_doctor_cli_does_not_create_files_or_directories(tmp_path):
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    result = _run_doctor(tmp_path)
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert result.returncode in {0, 1, 2}
    assert before == after
    json.loads(result.stdout)


def test_doctor_cli_does_not_echo_secret_or_artifact_paths(tmp_path):
    secret = "doctor-secret-path-token"
    secret_path = tmp_path / secret
    secret_path.write_bytes(b"artifact")
    result = _run_doctor(
        tmp_path,
        FUNASR_DEFAULT_ENGINE="qwen3",
        FUNASR_QWEN3_ASR_ENCODER_PROVIDER=secret,
        FUNASR_QWEN3_ASR_MODEL_DIR=str(secret_path),
        FUNASR_QWEN3_SEGMENTATION_MODEL=str(secret_path),
        FUNASR_QWEN3_EMBEDDING_MODEL=str(secret_path),
    )
    assert result.returncode == 2
    assert secret not in result.stdout
    assert secret not in result.stderr


def test_doctor_funasr_does_not_require_qwen_provider(tmp_path):
    result = _run_doctor(
        tmp_path,
        FUNASR_DEFAULT_ENGINE="funasr",
        FUNASR_QWEN3_ASR_ENCODER_PROVIDER="unavailable-provider",
    )
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["provider"]["fallback_would_occur"] is False
