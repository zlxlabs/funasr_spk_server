"""
Phase 2 — vendor QwenAudioEncoder 加 COREML_ANE_FE provider

PoC 验证 (M1 Max, N=2 wall): 单变量 -7.5%, 跟 num_threads=4 组合 -16.1%
唯一 work 的组合: MLProgram + CPUAndNeuralEngine + only_frontend (backend 保 CPU)
反例: ALL units 抢 llama.cpp Metal (+73s); NeuralNetwork format silent fallback CPU 无收益;
      backend 走 CoreML 卡 axis 4 op 兼容报错

详见 spikes/qwen3_mac_hw_accel/coreml_asr_encoder.md
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_ort_module(monkeypatch, mock_vendor_llama_native_module):
    """Mock onnxruntime 以避免真模型加载, 仍允许验证 providers 配置传给 InferenceSession."""
    # 拦截 ort.SessionOptions / ort.GraphOptimizationLevel / ort.InferenceSession
    fake = MagicMock()
    fake.GraphOptimizationLevel.ORT_ENABLE_ALL = "ORT_ENABLE_ALL"
    fake.SessionOptions = MagicMock
    # 通过 'mark' 让我们检验 InferenceSession 是怎么调的
    sessions_created = []

    def _fake_session(model_path, sess_options=None, providers=None):
        s = MagicMock()
        s.get_providers.return_value = [
            p[0] if isinstance(p, tuple) else p for p in (providers or [])
        ]
        # 模拟一个 float32 input
        inp = MagicMock()
        inp.type = "tensor(float)"
        s.get_inputs.return_value = [inp]
        sessions_created.append({
            "model_path": model_path,
            "providers": providers,
        })
        return s

    fake.InferenceSession = _fake_session
    fake.get_available_providers = MagicMock(return_value=["CPUExecutionProvider", "CoreMLExecutionProvider"])

    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    # vendor encoder module 在顶层 `import onnxruntime as ort` — reload 让它看到 mock
    import importlib
    import src.core.vendor.qwen_asr_gguf.inference.encoder as enc_mod
    importlib.reload(enc_mod)

    # 关闭真实的 mel_extractor + 预热(避免 dummy_wav encode 调用真前端)
    monkeypatch.setattr(enc_mod.QwenAudioEncoder, "encode", lambda self, *a, **kw: None)

    yield fake, sessions_created, enc_mod

    # 还原: 重新 reload 让其他测试看到真 onnxruntime
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    importlib.reload(enc_mod)


class TestCoremlAneFeOnMacos:
    def test_macos_uses_coreml_fe_and_cpu_be(self, fake_ort_module, monkeypatch):
        """macOS + CoreMLExecutionProvider 可用时, sess_fe 走 CoreML MLProgram + CPUAndNeuralEngine, sess_be 强 CPU"""
        fake_ort, sessions, enc_mod = fake_ort_module
        monkeypatch.setattr(sys, "platform", "darwin")

        enc_mod.QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path="/fake/be.onnx",
            onnx_provider="COREML_ANE_FE",
            verbose=False,
        )

        assert len(sessions) == 2, "应当创建 sess_fe + sess_be"
        fe_providers = sessions[0]["providers"]
        be_providers = sessions[1]["providers"]

        # sess_fe 第一个 provider 必须是 CoreML + 正确配置
        assert isinstance(fe_providers[0], tuple)
        assert fe_providers[0][0] == "CoreMLExecutionProvider"
        opts = fe_providers[0][1]
        assert opts["ModelFormat"] == "MLProgram"
        assert opts["MLComputeUnits"] == "CPUAndNeuralEngine"
        assert opts["EnableOnSubgraphs"] == "0"
        # CPU fallback 必须保留兜底
        assert "CPUExecutionProvider" in fe_providers

        # sess_be 必须强制 CPU only (backend axis 4 op 卡, 走 CoreML 会报错)
        assert be_providers == ["CPUExecutionProvider"]


class TestCoremlAneFeFallback:
    def test_linux_fallback_cpu_only(self, fake_ort_module, monkeypatch):
        """非 macOS 平台, COREML_ANE_FE 自动 fallback 全 CPU"""
        fake_ort, sessions, enc_mod = fake_ort_module
        monkeypatch.setattr(sys, "platform", "linux")

        enc_mod.QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path="/fake/be.onnx",
            onnx_provider="COREML_ANE_FE",
            verbose=False,
        )

        assert sessions[0]["providers"] == ["CPUExecutionProvider"]
        assert sessions[1]["providers"] == ["CPUExecutionProvider"]

    def test_macos_no_coreml_ep_fallback_cpu(self, fake_ort_module, monkeypatch):
        """macOS 但 onnxruntime 没编 CoreMLExecutionProvider, fallback CPU"""
        fake_ort, sessions, enc_mod = fake_ort_module
        monkeypatch.setattr(sys, "platform", "darwin")
        fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        enc_mod.QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path="/fake/be.onnx",
            onnx_provider="COREML_ANE_FE",
            verbose=False,
        )

        assert sessions[0]["providers"] == ["CPUExecutionProvider"]
        assert sessions[1]["providers"] == ["CPUExecutionProvider"]


class TestExistingBranchesUnchanged:
    def test_default_cpu_still_works(self, fake_ort_module, monkeypatch):
        """原有 CPU 默认行为不破坏: sess_fe / sess_be 都用 CPU only"""
        fake_ort, sessions, enc_mod = fake_ort_module
        monkeypatch.setattr(sys, "platform", "darwin")

        enc_mod.QwenAudioEncoder(
            frontend_path="/fake/fe.onnx",
            backend_path="/fake/be.onnx",
            onnx_provider="CPU",
            verbose=False,
        )

        assert sessions[0]["providers"] == ["CPUExecutionProvider"]
        assert sessions[1]["providers"] == ["CPUExecutionProvider"]
