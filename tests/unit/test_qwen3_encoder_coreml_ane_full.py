"""
Phase 3 — vendor QwenAudioEncoder 加 COREML_ANE_FULL provider

COREML_ANE_FULL = COREML_ANE_FE (frontend ANE) + backend mlpackage (CoreML ANE)

PoC 实测 (M1 Max):
- mlpackage 583MB FP16, load 24s (cold ANE compile), warm 69 ms/run on (1,390,1024)
- CPU_AND_NE 比 CPU_ONLY 2.2x (151.5 ms), 24s load 证明 ANE plan 真实编译
- parity vs ONNX backend cos 0.999069 max_abs 4.58e-3
- ANE Power readout 0 mW 是 macOS 26 Beta bug, ANE 实际在跑

详见 spikes/qwen3_mac_hw_accel/phase3_backend/
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_runtimes(monkeypatch, tmp_path, mock_vendor_llama_native_module):
    """Mock onnxruntime + coremltools, 验证 provider 路由 & mlmodel 加载参数."""
    fake_ort = MagicMock()
    fake_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = "ORT_ENABLE_ALL"
    fake_ort.SessionOptions = MagicMock
    sessions_created: list[dict] = []

    def _fake_session(model_path, sess_options=None, providers=None):
        s = MagicMock()
        s.get_providers.return_value = [
            p[0] if isinstance(p, tuple) else p for p in (providers or [])
        ]
        inp = MagicMock()
        inp.type = "tensor(float)"
        s.get_inputs.return_value = [inp]
        sessions_created.append({"model_path": model_path, "providers": providers})
        return s

    fake_ort.InferenceSession = _fake_session
    fake_ort.get_available_providers = MagicMock(
        return_value=["CPUExecutionProvider", "CoreMLExecutionProvider"]
    )

    # CoreMLZeroCopyRunner mock (PyObjC zero-copy 路径, 替代 coremltools.MLModel)
    mlmodels_created: list[dict] = []

    def _fake_runner_init(self, path, compute_units="CPU_AND_NE", verbose=True):
        mlmodels_created.append({"path": path, "compute_units": compute_units})
        self.predict = lambda inputs: {"last_hidden_state": object()}

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    import importlib
    import src.core.vendor.qwen_asr_gguf.inference.coreml_runner as cr_mod
    monkeypatch.setattr(cr_mod.CoreMLZeroCopyRunner, "__init__", _fake_runner_init)

    import src.core.vendor.qwen_asr_gguf.inference.encoder as enc_mod
    importlib.reload(enc_mod)

    monkeypatch.setattr(enc_mod.QwenAudioEncoder, "encode", lambda self, *a, **kw: None)

    # 准备 fake mlpackage 目录
    mlp = tmp_path / "fake_be.mlpackage"
    mlp.mkdir()

    yield {
        "ort": fake_ort,
        "sessions": sessions_created,
        "mlmodels": mlmodels_created,
        "enc_mod": enc_mod,
        "mlpackage_path": str(mlp),
    }

    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    importlib.reload(enc_mod)


class TestCoremlAneFullOnMacos:
    def test_macos_fe_coreml_be_mlpackage(self, fake_runtimes, monkeypatch):
        """macOS 上 COREML_ANE_FULL: sess_fe 走 CoreML EP (MLProgram CPUAndNeuralEngine),
        backend 用 ct.models.MLModel(..., compute_units=CPU_AND_NE)"""
        fx = fake_runtimes
        monkeypatch.setattr(sys, "platform", "darwin")

        fx["enc_mod"].QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path=fx["mlpackage_path"],
            onnx_provider="COREML_ANE_FULL",
            verbose=False,
        )

        # frontend 仍是 ONNX InferenceSession (走 CoreML EP)
        assert len(fx["sessions"]) == 1, "只 sess_fe 是 ONNX, sess_be 是 mlmodel"
        fe = fx["sessions"][0]
        assert fe["providers"][0][0] == "CoreMLExecutionProvider"
        opts = fe["providers"][0][1]
        assert opts["ModelFormat"] == "MLProgram"
        assert opts["MLComputeUnits"] == "CPUAndNeuralEngine"
        assert "CPUExecutionProvider" in fe["providers"]

        # backend mlmodel 加载并配 CPU_AND_NE (CoreMLZeroCopyRunner)
        assert len(fx["mlmodels"]) == 1
        mlp = fx["mlmodels"][0]
        assert mlp["path"] == fx["mlpackage_path"]
        assert mlp["compute_units"] == "CPU_AND_NE"


class TestCoremlAneFullFallback:
    def test_linux_fallback_cpu_full(self, fake_runtimes, monkeypatch):
        """非 macOS 平台, COREML_ANE_FULL 自动 fallback ONNX CPU (含 backend)"""
        fx = fake_runtimes
        monkeypatch.setattr(sys, "platform", "linux")

        fx["enc_mod"].QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path="/fake/be.onnx",
            onnx_provider="COREML_ANE_FULL",
            verbose=False,
        )

        # 全部走 ONNX CPU
        assert len(fx["sessions"]) == 2
        assert fx["sessions"][0]["providers"] == ["CPUExecutionProvider"]
        assert fx["sessions"][1]["providers"] == ["CPUExecutionProvider"]
        # 不加载任何 mlmodel
        assert len(fx["mlmodels"]) == 0

    def test_macos_mlpackage_missing_fallback_to_ane_fe(self, fake_runtimes, monkeypatch, tmp_path):
        """macOS 但 mlpackage 不存在 (or backend_path 不是 .mlpackage), fallback COREML_ANE_FE 行为
        (frontend ANE + backend ONNX CPU)"""
        fx = fake_runtimes
        monkeypatch.setattr(sys, "platform", "darwin")
        missing_path = str(tmp_path / "not_a_mlpackage.onnx")

        fx["enc_mod"].QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path=missing_path,
            onnx_provider="COREML_ANE_FULL",
            verbose=False,
        )

        # frontend 仍走 ANE, backend 退回 ONNX CPU (不加载 mlmodel)
        assert len(fx["mlmodels"]) == 0, "mlpackage 不存在不应该尝试加载"
        assert len(fx["sessions"]) == 2, "fallback 必须创建 sess_fe + sess_be (都是 ONNX)"
        fe = fx["sessions"][0]
        assert fe["providers"][0][0] == "CoreMLExecutionProvider"  # frontend ANE
        be = fx["sessions"][1]
        assert be["providers"] == ["CPUExecutionProvider"]  # backend CPU

    def test_macos_no_coreml_ep_fallback_cpu(self, fake_runtimes, monkeypatch):
        """macOS 但 onnxruntime 没编 CoreMLExecutionProvider, 全部 fallback CPU"""
        fx = fake_runtimes
        monkeypatch.setattr(sys, "platform", "darwin")
        fx["ort"].get_available_providers.return_value = ["CPUExecutionProvider"]

        fx["enc_mod"].QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path=fx["mlpackage_path"],
            onnx_provider="COREML_ANE_FULL",
            verbose=False,
        )

        # 因为 frontend 都没法走 CoreML EP, full 路径整体 fallback CPU (含不加载 mlmodel)
        assert len(fx["mlmodels"]) == 0
        assert len(fx["sessions"]) == 2
        assert fx["sessions"][0]["providers"] == ["CPUExecutionProvider"]
        assert fx["sessions"][1]["providers"] == ["CPUExecutionProvider"]


class TestExistingBranchesUnchanged:
    """Phase 3 不影响 Phase 2 (COREML_ANE_FE) + 原 CPU 路径"""

    def test_coreml_ane_fe_unchanged(self, fake_runtimes, monkeypatch):
        fx = fake_runtimes
        monkeypatch.setattr(sys, "platform", "darwin")
        fx["enc_mod"].QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path="/fake/be.onnx",
            onnx_provider="COREML_ANE_FE",
            verbose=False,
        )
        assert len(fx["mlmodels"]) == 0, "COREML_ANE_FE 不加载 mlmodel"
        assert len(fx["sessions"]) == 2

    def test_cpu_unchanged(self, fake_runtimes, monkeypatch):
        fx = fake_runtimes
        monkeypatch.setattr(sys, "platform", "darwin")
        fx["enc_mod"].QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path="/fake/be.onnx",
            onnx_provider="CPU",
            verbose=False,
        )
        assert len(fx["mlmodels"]) == 0
        assert fx["sessions"][0]["providers"] == ["CPUExecutionProvider"]
        assert fx["sessions"][1]["providers"] == ["CPUExecutionProvider"]
