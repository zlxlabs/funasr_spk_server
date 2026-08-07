"""I4 doctor contract: strict read-only diagnostics and safe JSON output."""
from __future__ import annotations

import importlib.util
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
    inspect_doctor_directory_target,
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
    shutil.copy(ROOT / "src" / "core" / "config.py", root / "src" / "core" / "config.py")
    shutil.copy(ROOT / "src" / "core" / "runtime.py", root / "src" / "core" / "runtime.py")
    shutil.copy(ROOT / "src" / "core" / "doctor_config_probe.py", root / "src" / "core" / "doctor_config_probe.py")
    if isinstance(config_data, bytes):
        (root / "config.json").write_bytes(config_data)
    elif isinstance(config_data, str):
        (root / "config.json").write_text(config_data, encoding="utf-8")
    else:
        (root / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
    (root / ".env").write_text(dotenv_text, encoding="utf-8")
    result = _run_fixture_doctor(root, process_env)
    return root, result


def _run_fixture_doctor(root: Path, process_env=None):
    """在隔离 fixture 中执行 doctor，避免继承本机 FUNASR_* 污染。"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("FUNASR_")}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.update({"FUNASR_NOTIFICATION_ENABLED": "false"})
    env.update(process_env or {})
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "doctor.py"), "--json"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )


def _run_fixture_probe(root: Path, process_env=None):
    """直接执行 -m probe，验证 IPC stdout 只含安全元数据。"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("FUNASR_")}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.update(
        {
            "FUNASR_NOTIFICATION_ENABLED": "false",
            "FUNASR_RUNTIME": "cpu",
            "FUNASR_QWEN3_ASR_ENCODER_PROVIDER": "cpu",
        }
    )
    env.update(process_env or {})
    return subprocess.run(
        [sys.executable, "-m", "src.core.doctor_config_probe", str(root / "config.json")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )


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
        qwen_artifacts={
            "asr_model_dir": {"exists": True, "type": "directory", "size": 1},
            "segmentation_model": artifact,
            "embedding_model": artifact,
        },
        funasr_dynamic_cache="unknown",
        word_align_artifact=artifact,
    )
    assert report["provider"]["fallback_would_occur"] is False
    assert report["fallback_would_occur"] is False
    assert report["errors"] == []
    assert report["status"] == "ok"


@pytest.mark.parametrize(
    ("configured", "platform_name", "available", "effective", "fallback", "error"),
    [
        ("auto", "linux", ["CPUExecutionProvider"], "CPUExecutionProvider", False, None),
        ("auto", "darwin", ["CoreMLExecutionProvider"], "CoreMLExecutionProvider", False, None),
        ("cpu", "linux", ["CPUExecutionProvider"], "CPUExecutionProvider", False, None),
        ("cuda", "linux", ["CUDAExecutionProvider"], "CUDAExecutionProvider", False, None),
        ("cuda", "linux", ["CPUExecutionProvider"], "CPUExecutionProvider", True, "configured_provider_unavailable"),
        ("cuda", "linux", [], "unavailable", True, "configured_provider_unavailable"),
        ("cuda", "linux", ["TensorrtExecutionProvider"], "unavailable", True, "configured_provider_unavailable"),
        ("tensorrt", "linux", ["TensorrtExecutionProvider"], "TensorrtExecutionProvider", False, None),
        ("trt", "linux", ["TensorrtExecutionProvider"], "TensorrtExecutionProvider", False, None),
        ("coreml_ane_fe", "darwin", ["CoreMLExecutionProvider"], "CoreMLExecutionProvider", False, None),
        ("coreml_ane_full", "darwin", ["CoreMLExecutionProvider"], "CoreMLExecutionProvider", False, None),
        ("dml", "linux", ["CPUExecutionProvider"], "CPUExecutionProvider", True, "unsupported_provider"),
        ("coreml", "darwin", ["CoreMLExecutionProvider"], "CoreMLExecutionProvider", True, "unsupported_provider"),
        ("foo", "darwin", ["CoreMLExecutionProvider"], "CoreMLExecutionProvider", True, "unsupported_provider"),
    ],
)
def test_doctor_provider_resolution_matches_qwen_supported_paths(
    configured, platform_name, available, effective, fallback, error
):
    artifact = {"exists": True, "type": "file", "size": 1}
    qwen_artifacts = {
        "asr_model_dir": {"exists": True, "type": "directory", "size": 1},
        "segmentation_model": artifact,
        "embedding_model": artifact,
    }
    if configured == "coreml_ane_full":
        qwen_artifacts["backend_mlpackage"] = {
            "exists": True,
            "type": "directory",
            "size": 1,
        }
    report = build_doctor_report(
        engine="qwen3",
        runtime="cuda",
        platform_name=platform_name,
        configured_provider=configured,
        available_providers=available,
        qwen_artifacts=qwen_artifacts,
        funasr_dynamic_cache="deferred",
        word_align_artifact=artifact,
    )
    assert report["provider"]["effective"] == effective
    assert report["fallback_would_occur"] is fallback
    if error:
        assert error in report["errors"]
    else:
        assert report["errors"] == []


def test_doctor_invalid_qwen_fields_are_fatal_even_for_funasr(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {
            "transcription": {"default_engine": "funasr"},
            "qwen3": {"backend_mlpackage_units": "not-a-valid-unit"},
        },
    )
    report = json.loads(result.stdout)
    assert result.returncode == 2
    assert report["status"] == "error"
    assert "configuration_invalid" in report["errors"]


@pytest.mark.parametrize("word_align_artifact", [
    {"exists": False, "type": "invalid", "size": 0},
    {"exists": False, "type": "missing", "size": 0},
    {"exists": True, "type": "file", "size": 0},
    {"exists": True, "type": "directory", "size": 1},
])
def test_doctor_qwen_word_align_is_optional(word_align_artifact):
    artifact = {"exists": True, "type": "file", "size": 1}
    report = build_doctor_report(
        engine="qwen3",
        runtime="cpu",
        configured_provider="cpu",
        available_providers=["CPUExecutionProvider"],
        qwen_artifacts={
            "asr_model_dir": {"exists": True, "type": "directory", "size": 1},
            "segmentation_model": artifact,
            "embedding_model": artifact,
        },
        funasr_dynamic_cache="deferred",
        word_align_artifact=word_align_artifact,
    )
    assert report["status"] == "warning"
    assert report["errors"] == []
    assert report["warnings"] == ["optional_word-align_unavailable"]


def test_doctor_config_priority_matches_real_config(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {"transcription": {"default_engine": "funasr"}},
        "FUNASR_PROFILE=cuda_dev\nFUNASR_DEFAULT_ENGINE=funasr\n",
        {"FUNASR_DEFAULT_ENGINE": "qwen3"},
    )
    report = json.loads(result.stdout)
    assert report["engine"] == "qwen3"
    assert report["provider"]["configured"] == "cuda"


def test_doctor_empty_qwen_provider_uses_real_config_value(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {"transcription": {"default_engine": "qwen3"}},
        process_env={"FUNASR_QWEN3_ASR_ENCODER_PROVIDER": ""},
    )
    report = json.loads(result.stdout)
    assert report["provider"]["configured"] == "auto"


@pytest.mark.parametrize(
    "config_data",
    [
        [],
        {"transcription": {"default_engine": "bogus"}},
        {"transcription": {"default_engine": ""}},
        {"transcription": {"default_engine": "qwen3"}, "qwen3": {"asr_model_dir": None}},
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
    (asr_dir / "model.json").write_text("model", encoding="utf-8")
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


def test_doctor_invalid_utf8_config_is_safe_configuration_unavailable(tmp_path):
    _, result = _doctor_fixture(tmp_path, b"{\xff")
    assert result.returncode == 2
    assert len(result.stdout.splitlines()) == 1
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert "configuration_unavailable" in report["errors"]
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_doctor_invalid_json_config_is_safe_configuration_unavailable(tmp_path):
    _, result = _doctor_fixture(tmp_path, "{not-json")
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert "configuration_unavailable" in report["errors"]
    assert "not-json" not in result.stdout + result.stderr


def test_doctor_missing_config_is_safe_configuration_unavailable(tmp_path):
    root, _ = _doctor_fixture(tmp_path, {})
    (root / "config.json").unlink()
    result = _run_fixture_doctor(root)
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert "configuration_unavailable" in report["errors"]


def test_doctor_unreadable_config_is_safe_configuration_unavailable(tmp_path):
    root, _ = _doctor_fixture(tmp_path, {})
    (root / "config.json").unlink()
    (root / "config.json").mkdir()
    result = _run_fixture_doctor(root)
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert "configuration_unavailable" in report["errors"]


def test_doctor_real_config_rejects_transcription_type_error(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {"transcription": {"max_concurrent_tasks": "bad"}},
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert "configuration_invalid" in report["errors"]
    assert "bad" not in result.stdout + result.stderr


def test_doctor_real_config_rejects_qwen_enum_error(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {"qwen3": {"backend_mlpackage_units": "not-a-valid-unit"}},
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert "configuration_invalid" in report["errors"]
    assert "not-a-valid-unit" not in result.stdout + result.stderr


def test_doctor_real_config_rejects_qwen_type_error(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {"qwen3": {"num_threads": "not-an-integer"}},
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert "configuration_invalid" in report["errors"]
    assert "not-an-integer" not in result.stdout + result.stderr


def test_doctor_env_repairs_lower_priority_bad_config_like_real_config(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {"transcription": {"max_concurrent_tasks": "bad"}},
        process_env={"FUNASR_MAX_CONCURRENT_TASKS": "4"},
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["status"] == "warning"
    assert "configuration_invalid" not in report["errors"]


def test_doctor_rejects_invalid_inactive_section_like_real_config(tmp_path):
    _, result = _doctor_fixture(
        tmp_path,
        {"qwen3": {"backend_mlpackage_units": "not-a-valid-unit"}},
    )
    report = json.loads(result.stdout)
    assert result.returncode == 2
    assert report["status"] == "error"
    assert "configuration_invalid" in report["errors"]


def test_doctor_probe_does_not_create_bytecode_or_directories(tmp_path):
    root, _ = _doctor_fixture(tmp_path, {})
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    result = _run_doctor(root)
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert result.returncode in {0, 1, 2}
    assert before == after
    assert not list(root.rglob("*.pyc"))
    assert not list(root.rglob("__pycache__"))


def test_doctor_main_does_not_mutate_calling_process_environment(monkeypatch, capsys):
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    before = dict(os.environ)
    spec = importlib.util.spec_from_file_location("doctor_environment_test", DOCTOR)
    assert spec is not None and spec.loader is not None
    doctor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doctor)
    monkeypatch.setattr(doctor, "_diagnose", lambda: ({"status": "ok"}, 0))
    assert doctor.main(["--json"]) == 0
    assert dict(os.environ) == before
    capsys.readouterr()


@pytest.mark.parametrize("probe_source", ["process", "dotenv"])
def test_probe_env_name_cannot_disable_normal_service_config(tmp_path, probe_source):
    probe_line = "" if probe_source == "process" else "FUNASR_DOCTOR_CONFIG_PROBE=1\n"
    (tmp_path / ".env").write_text(
        probe_line
        + "FUNASR_NOTIFICATION_ENABLED=false\n"
        + "FUNASR_DEFAULT_ENGINE=funasr\n"
        f"FUNASR_TEMP_DIR={tmp_path / 'temp'}\n"
        f"FUNASR_UPLOAD_DIR={tmp_path / 'uploads'}\n"
        f"FUNASR_MODEL_DIR={tmp_path / 'models'}\n"
        f"FUNASR_DATA_DIR={tmp_path / 'data'}\n"
        f"FUNASR_LOG_DIR={tmp_path / 'logs'}\n",
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("FUNASR_")}
    env["PYTHONPATH"] = str(ROOT)
    if probe_source == "process":
        env["FUNASR_DOCTOR_CONFIG_PROBE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", "import src.main"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "NoneType" not in result.stderr


def test_doctor_directory_file_target_is_fatal_without_writing(tmp_path):
    root, _ = _doctor_fixture(tmp_path, {})
    target = root / "upload-file"
    target.write_text("not a directory", encoding="utf-8")
    result = _run_fixture_doctor(root, {"FUNASR_UPLOAD_DIR": str(target)})
    report = json.loads(result.stdout)
    assert result.returncode == 2
    assert "directory_unavailable" in report["errors"]
    assert "upload-file" not in result.stdout


def test_doctor_directory_file_ancestor_is_fatal_without_writing(tmp_path):
    root, _ = _doctor_fixture(tmp_path, {})
    blocked = root / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    result = _run_fixture_doctor(root, {"FUNASR_UPLOAD_DIR": str(blocked / "uploads")})
    report = json.loads(result.stdout)
    assert result.returncode == 2
    assert "directory_unavailable" in report["errors"]


def test_doctor_directory_existing_and_creatable_targets_are_usable(tmp_path):
    root, _ = _doctor_fixture(tmp_path, {})
    existing = root / "existing"
    existing.mkdir()
    result = _run_fixture_doctor(
        root,
        {
            "FUNASR_TEMP_DIR": str(existing / "temp"),
            "FUNASR_UPLOAD_DIR": str(existing / "uploads"),
            "FUNASR_MODEL_DIR": str(existing / "models"),
            "FUNASR_DATA_DIR": str(existing / "data"),
            "FUNASR_LOG_DIR": str(existing / "logs"),
        },
    )
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert "directory_unavailable" not in report["errors"]


def test_doctor_directory_target_requires_writable_searchable_ancestor(tmp_path, monkeypatch):
    target = tmp_path / "missing" / "target"
    monkeypatch.setattr("src.core.doctor_diagnostics.os.access", lambda *_: False)
    result = inspect_doctor_directory_target(target)
    assert result["usable"] is False
    assert result["error"] == "ancestor_not_writable"


def test_probe_stdout_contains_artifact_metadata_but_no_raw_paths(tmp_path):
    secret = "doctor-secret-ipc-path"
    root, _ = _doctor_fixture(
        tmp_path,
        {
            "transcription": {"default_engine": "qwen3"},
            "qwen3": {
                "asr_model_dir": str(tmp_path / secret),
                "segmentation_model": str(tmp_path / (secret + ".seg")),
                "embedding_model": str(tmp_path / (secret + ".embed")),
                "word_align_model_path": str(tmp_path / (secret + ".words")),
            },
        },
    )
    result = _run_fixture_probe(root)
    assert result.returncode == 0
    assert len(result.stdout.splitlines()) == 1
    assert secret not in result.stdout
    payload = json.loads(result.stdout)
    assert set(payload["artifacts"]) == {"qwen", "word_align"}
    assert len(payload["directories"]) == 5


def test_probe_coreml_backend_and_word_align_are_safe_metadata(tmp_path):
    root = tmp_path / "fixture-repo"
    model_dir = root / "qwen-model"
    model_dir.mkdir(parents=True)
    (model_dir / "qwen3_asr_encoder_backend.mlpackage").mkdir()
    word_align = root / "word-align.onnx"
    word_align.write_bytes(b"align")
    root, _ = _doctor_fixture(
        tmp_path,
        {
            "transcription": {"default_engine": "qwen3"},
            "qwen3": {
                "asr_model_dir": str(model_dir),
                "segmentation_model": str(root / "segmentation.onnx"),
                "embedding_model": str(root / "embedding.onnx"),
                "word_align_model_path": str(word_align),
            },
        },
    )
    result = _run_fixture_probe(root, {"FUNASR_QWEN3_ASR_ENCODER_PROVIDER": "coreml_ane_full"})
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["artifacts"]["qwen"]["backend_mlpackage"]["type"] == "directory"
    assert payload["artifacts"]["word_align"] == {"exists": True, "type": "file", "size": 5}
    assert str(model_dir) not in result.stdout
