#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for tool in uv ffmpeg ffprobe; do
    command -v "$tool" >/dev/null || { printf '缺少前置命令: %s\n' "$tool" >&2; exit 1; }
done
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/funasr-cli/modelscope}"
uv venv --python 3.12 "$ROOT/.venv-cpu"
uv pip install --python "$ROOT/.venv-cpu/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0
uv pip install --python "$ROOT/.venv-cpu/bin/python" -r "$ROOT/requirements-cpu.txt"
"$ROOT/scripts/funasr-transcribe" --download-models
printf '安装完成。运行: "%s/scripts/funasr-transcribe" input.wav > result.json\n' "$ROOT"
