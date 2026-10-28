"""专用 CPU 环境中的全真入口测试；保留实际 stdout/stderr 作为证据。"""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("FUNASR_RUN_INTEGRATION") != "1",
    reason="设置 FUNASR_RUN_INTEGRATION=1 执行真实 CPU 模型",
)


def test_cpu_cli_real_model(tmp_path):
    """真实 60 秒双人音频必须产生文本、时间戳和说话人，不能借用测试桩。"""
    evidence = ROOT / ".venv-cpu/cli-evidence"
    evidence.mkdir(exist_ok=True)
    with (evidence / "stderr.log").open("wb") as stderr:
        result = subprocess.run(
            [ROOT / "scripts/funasr-transcribe",
             ROOT / "tests/fixtures/audio/podcast_2speakers_60s.wav"],
            cwd=tmp_path,
            env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")},
            stdout=subprocess.PIPE, stderr=stderr, timeout=1200,
        )
    (evidence / "result.json").write_bytes(result.stdout)
    assert result.returncode == 0, f"见 {evidence / 'stderr.log'}"
    data = json.loads(result.stdout.decode("utf-8"))
    assert 59 <= data["duration"] <= 61
    assert data["segments"] and data["speakers"]
    assert any(segment["text"].strip() for segment in data["segments"])
    for segment in data["segments"]:
        assert 0 <= segment["start_time"] < segment["end_time"] <= data["duration"] + 1
        assert segment["speaker"] in data["speakers"]
