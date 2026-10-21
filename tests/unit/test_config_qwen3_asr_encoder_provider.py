"""
Phase 2 Step 2.3 — 加 FUNASR_QWEN3_ASR_ENCODER_PROVIDER env knob (生产 escape hatch)

设计:
- Qwen3Config.asr_encoder_provider: str = "auto" (auto / cpu / coreml_ane_fe)
- env FUNASR_QWEN3_ASR_ENCODER_PROVIDER 覆盖
- "auto" 时 build_engine_config 走平台感知; 其他值直接覆盖 onnx_provider

用途: 生产环境如果 ANE 出问题, 改 env "cpu" 一键回退, 不用改代码 / 重打包
"""
import pytest

from src.core.config import Qwen3Config, Config


@pytest.fixture
def mock_vendor_llama_native_module(monkeypatch):
    """Linux 配置单测使用真实 schema，但不加载 macOS 专用 llama 原生库。"""
    import sys
    import types

    if sys.platform != "darwin":
        llama_module = "src.core.vendor.qwen_asr_gguf.inference.llama"
        monkeypatch.setitem(sys.modules, llama_module, types.ModuleType(llama_module))
    import src.core.vendor.qwen_asr_gguf.inference.schema


class TestAsrEncoderProviderField:
    def test_default_asr_encoder_provider_is_auto(self):
        """默认 auto, 由 build_engine_config 按平台感知"""
        cfg = Qwen3Config()
        assert cfg.asr_encoder_provider == "auto"

    def test_init_override_cpu(self):
        cfg = Qwen3Config(asr_encoder_provider="cpu")
        assert cfg.asr_encoder_provider == "cpu"

    def test_init_override_coreml_ane_fe(self):
        cfg = Qwen3Config(asr_encoder_provider="coreml_ane_fe")
        assert cfg.asr_encoder_provider == "coreml_ane_fe"


class TestAsrEncoderProviderEnvOverride:
    def test_env_cpu_override(self, monkeypatch, tmp_path):
        """FUNASR_QWEN3_ASR_ENCODER_PROVIDER=cpu 覆盖默认 auto"""
        monkeypatch.setenv("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cpu")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.asr_encoder_provider == "cpu"

    def test_env_coreml_ane_fe_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "coreml_ane_fe")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.asr_encoder_provider == "coreml_ane_fe"

    def test_no_env_keeps_auto(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", raising=False)
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.asr_encoder_provider == "auto"


class TestBuildEngineConfigReadsConfigField:
    """build_engine_config 在 onnx_provider=None 时, 先读 config.qwen3.asr_encoder_provider

    优先级: 显式参数 > config 字段 (含 env) > 平台感知
    """

    def test_auto_on_macos_resolves_to_coreml_ane_fe(self, monkeypatch, mock_vendor_llama_native_module):
        import sys
        from src.core.qwen3.asr import build_engine_config

        # 默认 config 是 auto, 在 macOS 上 build_engine_config 应推 COREML_ANE_FE
        monkeypatch.setattr(sys, "platform", "darwin")
        cfg = build_engine_config(model_dir="/tmp/fake")
        assert cfg.onnx_provider == "COREML_ANE_FE"

    def test_config_cpu_overrides_macos_default(self, monkeypatch, tmp_path, mock_vendor_llama_native_module):
        """env FUNASR_QWEN3_ASR_ENCODER_PROVIDER=cpu 应能在 macOS 上一键关掉 ANE"""
        import sys
        from src.core import config as _config_module
        from src.core.qwen3.asr import build_engine_config

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cpu")
        # 重新加载 config 让 env 生效
        new_config = _config_module.Config.load_from_file(str(tmp_path / "nonexistent.json"))
        monkeypatch.setattr(_config_module, "config", new_config)

        cfg = build_engine_config(model_dir="/tmp/fake")
        assert cfg.onnx_provider == "CPU"

    def test_config_coreml_on_linux_still_uses_coreml(self, monkeypatch, tmp_path, mock_vendor_llama_native_module):
        """linux 用户显式 env=coreml_ane_fe 应能透传 (encoder 自身会 fallback CPU)"""
        import sys
        from src.core import config as _config_module
        from src.core.qwen3.asr import build_engine_config

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "coreml_ane_fe")
        new_config = _config_module.Config.load_from_file(str(tmp_path / "nonexistent.json"))
        monkeypatch.setattr(_config_module, "config", new_config)

        cfg = build_engine_config(model_dir="/tmp/fake")
        assert cfg.onnx_provider == "COREML_ANE_FE"

    def test_explicit_param_beats_config(self, monkeypatch, tmp_path, mock_vendor_llama_native_module):
        """显式传 onnx_provider 仍最高优先级, 不被 config 覆盖"""
        import sys
        from src.core import config as _config_module
        from src.core.qwen3.asr import build_engine_config

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cpu")
        new_config = _config_module.Config.load_from_file(str(tmp_path / "nonexistent.json"))
        monkeypatch.setattr(_config_module, "config", new_config)

        cfg = build_engine_config(model_dir="/tmp/fake", onnx_provider="COREML_ANE_FE")
        assert cfg.onnx_provider == "COREML_ANE_FE"
