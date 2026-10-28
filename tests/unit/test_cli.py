"""从实际 shell 入口验证 CPU CLI；只替换最外层 AutoModel。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/funasr-transcribe"


@pytest.fixture
def cli_clone(tmp_path):
    """搭一个独立 clone 布局，Python 仍用当前测试解释器。"""
    repo = tmp_path / "clone space"
    repo.mkdir()
    (repo / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (repo / "scripts").mkdir()
    shutil.copy(ENTRY, repo / "scripts/funasr-transcribe")
    (repo / ".venv-cpu/bin").mkdir(parents=True)
    (repo / ".venv-cpu/bin/python").write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    (repo / ".venv-cpu/bin/python").chmod(0o755)
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "funasr.py").write_text('''import json, os
print("MODEL IMPORT NOISE")
class AutoModel:
    def __init__(self, **kwargs):
        print("MODEL LOAD NOISE")
        with open(os.environ["MODEL_ARGS"], "w") as f:
            json.dump(kwargs, f)
    def generate(self, **kwargs):
        print("GENERATE NOISE")
        if os.environ.get("MODEL_FAIL"):
            raise RuntimeError("model deliberately failed")
        return [{"text": "你好。", "sentence_info": [
            {"start": 100, "end": 1200, "text": "你好。", "spk": 0}]}]
''')
    env = dict(os.environ, PYTHONPATH=str(fake), MODEL_ARGS=str(tmp_path / "args.json"))
    return repo / "scripts/funasr-transcribe", env


def test_json_real_entry(cli_clone, tmp_path):
    """模型噪声不能污染 JSON，真实内核投影保留 spk=0。"""
    entry, env = cli_clone
    audio = tmp_path / "audio space.wav"
    shutil.copy(ROOT / "tests/fixtures/audio/tts_1speaker_5s.wav", audio)
    result = subprocess.run([entry, audio.name], cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    data = json.loads(result.stdout.decode("utf-8"))
    assert data["segments"][0] == {"start_time": 0.1, "end_time": 1.2,
                                    "text": "你好。", "speaker": "Speaker1", "words": None}
    assert data["speakers"] == ["Speaker1"]
    assert b"MODEL LOAD NOISE" in result.stderr and b"GENERATE NOISE" in result.stderr
    args = json.loads(Path(env["MODEL_ARGS"]).read_text())
    assert args["device"] == "cpu" and args["ncpu"] == 2
    assert args["model"] == "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    for key in ("vad_kwargs", "punc_kwargs", "spk_kwargs"):
        assert args[key]["ncpu"] == 2
    assert not (tmp_path / "data").exists()


def test_early_errors_and_help(cli_clone, tmp_path):
    """参数和文件校验先于所有模型导入，help 无副作用。"""
    entry, env = cli_clone
    for argv, expected in [(["missing.wav"], b"FileNotFoundError"), ([], b"error"),
                           (["--bad"], b"error")]:
        result = subprocess.run([entry, *argv], cwd=tmp_path, env=env, capture_output=True)
        assert result.returncode != 0 and result.stdout == b""
        assert expected in result.stderr and b"MODEL" not in result.stderr
    result = subprocess.run([entry, "--help"], cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode == 0 and b"MODEL" not in result.stdout
    assert not Path(env["MODEL_ARGS"]).exists()


def test_model_failure_is_nonzero(cli_clone, tmp_path):
    """不把失败降级成空 JSON。"""
    entry, env = cli_clone
    env["MODEL_FAIL"] = "1"
    result = subprocess.run([entry, ROOT / "tests/fixtures/audio/tts_1speaker_5s.wav"],
                            cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode != 0 and result.stdout == b""
    assert b"model deliberately failed" in result.stderr


@pytest.mark.parametrize("cache", [None, "cache space"])
def test_installer_producer(tmp_path, cache):
    """截获安装脚本实际调用的 uv 参数和缓存环境，而非 grep 源码。"""
    repo = tmp_path / "repo space"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/install_cpu_cli.sh", repo / "scripts/install_cpu_cli.sh")
    shutil.copy(ROOT / "requirements-cpu.txt", repo / "requirements-cpu.txt")
    (repo / "scripts/funasr-transcribe").write_text('#!/bin/bash\nprintf "%s" "$MODELSCOPE_CACHE" > "$CAPTURE/cache"\nprintf "%s\\n" "$@" > "$CAPTURE/download"\n')
    (repo / "scripts/funasr-transcribe").chmod(0o755)
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "uv").write_text('#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ["CAPTURE"]+"/uv.jsonl", "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    (tools / "uv").chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + ":" + os.environ["PATH"],
               CAPTURE=str(tmp_path), HOME=str(tmp_path / "home"))
    env.pop("MODELSCOPE_CACHE", None)
    expected_cache = str(tmp_path / "home/.cache/funasr-cli/modelscope")
    if cache:
        expected_cache = env["MODELSCOPE_CACHE"] = str(tmp_path / cache)
    result = subprocess.run(["bash", repo / "scripts/install_cpu_cli.sh"], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    calls = [json.loads(line) for line in (tmp_path / "uv.jsonl").read_text().splitlines()]
    assert calls[0] == ["venv", "--python", "3.12", str(repo / ".venv-cpu")]
    assert calls[1] == ["pip", "install", "--python", str(repo / ".venv-cpu/bin/python"),
                        "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.8.0", "torchaudio==2.8.0"]
    assert calls[2] == ["pip", "install", "--python", str(repo / ".venv-cpu/bin/python"),
                        "-r", str(repo / "requirements-cpu.txt")]
    assert (tmp_path / "cache").read_text() == expected_cache
    assert (tmp_path / "download").read_text() == "--download-models\n"
