"""GPU 显存进程用量探针与 sidecar 预检公开接口回归测试。"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from src.core import gpu_mem


def _completed(stdout: str) -> subprocess.CompletedProcess:
    """构造 nvidia-smi 的成功输出。"""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_process_vram_mib_returns_requested_pid_usage(monkeypatch):
    """按指定 PID 解析占用，并沿用共享设备选择和 5 秒超时。"""
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed("1234, 512\n27182, 4096\n")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(gpu_mem.subprocess, "run", run)

    assert gpu_mem.process_vram_mib(27182) == 4096
    command, kwargs = calls[0]
    assert command == [
        "nvidia-smi", "-i", "2", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    assert kwargs == {"capture_output": True, "text": True, "timeout": 5, "check": True}


def test_process_vram_mib_returns_zero_when_pid_is_absent(monkeypatch):
    """查询成功但进程不在占用列表中，表示该 PID 当前占用 0 MiB。"""
    monkeypatch.setattr(
        gpu_mem.subprocess, "run", lambda *args, **kwargs: _completed("1234, 512\n")
    )
    assert gpu_mem.process_vram_mib(27182) == 0


@pytest.mark.parametrize(
    ("visible_devices", "expected_device"),
    [(None, "0"), ("", "0"), ("7", "7"), ("7,8", "7")],
)
def test_process_vram_mib_resolves_device_index(monkeypatch, visible_devices, expected_device):
    """未设置、空值、单卡和多卡可见设备都按共享规则解析。"""
    command_seen = []
    if visible_devices is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    def run(command, **kwargs):
        command_seen.append(command)
        return _completed("27182, 128\n")

    monkeypatch.setattr(gpu_mem.subprocess, "run", run)
    assert gpu_mem.process_vram_mib(27182) == 128
    assert command_seen[0][2] == expected_device


def test_process_vram_mib_timeout_returns_none_and_logs_marker(monkeypatch):
    """nvidia-smi 超时须传 timeout，并留下可检索的降级日志标记。"""
    calls = []
    messages = []
    handler_id = logger.add(messages.append, format="{message}", level="DEBUG")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    monkeypatch.setattr(gpu_mem.subprocess, "run", run)
    try:
        assert gpu_mem.process_vram_mib(27182) is None
    finally:
        logger.remove(handler_id)

    assert calls[0][1]["timeout"] == 5
    assert any("GPU-MEM-PID-PROBE-FAILED" in message for message in messages)


def test_sidecar_client_exposes_only_live_pid():
    """只读公开 PID 属性仅返回存活子进程的 PID。"""
    from src.core.qwen3.word_align_sidecar import WordAlignSidecarClient

    client = WordAlignSidecarClient("/tmp/test.sock", ["python", "-m", "sidecar"])
    assert client.running_pid is None
    client._proc = SimpleNamespace(pid=27182, poll=lambda: None)
    assert client.running_pid == 27182
    with pytest.raises(AttributeError):
        client.running_pid = 12345
    client._proc = SimpleNamespace(pid=27182, poll=lambda: 1)
    assert client.running_pid is None


@pytest.mark.asyncio
async def test_public_transcribe_preflight_uses_client_public_pid(monkeypatch, tmp_path):
    """公开 transcribe 入口经受限 client spec 获取 PID，不依赖 client._proc。"""
    from src.core.qwen3.asr import ASRChunkItem, ASRResult
    from src.core.qwen3.word_align_sidecar import WordAlignSidecarClient
    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    from src.models.schemas import TranscribeOptions

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    tx = Qwen3DiarizeTranscriber(
        asr_model_dir="/fake/asr", segmentation_model="/fake/seg.onnx",
        embedding_model="/fake/emb.onnx", cluster_merge_enabled=False,
        silence_align_enabled=False, nospk_split_enabled=False,
        word_align_enabled=True, word_align_provider="cuda", word_align_sidecar_enabled=True,
        word_align_preflight_free_mib=4608,
    )
    # spec_set 保证生产路径若直接读 client._proc 会立即失败。
    from unittest.mock import create_autospec

    client = create_autospec(WordAlignSidecarClient, instance=True, spec_set=True)
    client.running_pid = 27182
    client.align.return_value = (
        [{"text": "你好", "start": 0.2, "end": 1.0, "score": -0.1}],
        {"total_windows": 1, "failed_windows": 0, "total_words": 1, "failures": []},
    )
    asr_result = ASRResult(
        text="你好", items=[],
        chunks=[ASRChunkItem(text="你好", start=0.0, end=2.0, index=0)],
        duration=2.0, elapsed=0.1, rtf=0.05, peak_rss_mb=100.0, rss_delta_mb=1.0,
    )
    with patch("src.core.qwen3_transcriber.build_engine", return_value=object()), \
         patch("src.core.qwen3_transcriber.run_asr", return_value=asr_result), \
         patch(
             "src.core.qwen3_transcriber.calculate_file_hash",
             new=AsyncMock(return_value="audio-hash"),
         ), \
         patch.object(tx, "_word_align_primary_is_cuda", return_value=True) as is_cuda, \
         patch("src.core.qwen3_transcriber.get_word_align_sidecar_client", return_value=client), \
         patch("src.core.qwen3_transcriber.free_vram_mib", return_value=1200), \
         patch("src.core.qwen3_transcriber.process_vram_mib", return_value=4096) as process_probe:
        await tx.transcribe(
            str(audio_path), "task-sidecar-pid", output_format="json",
            options=TranscribeOptions(word_align=True, diarize=False, language="zh"),
        )

    is_cuda.assert_called_once()
    process_probe.assert_called_once_with(27182)
    client.align.assert_called_once()
