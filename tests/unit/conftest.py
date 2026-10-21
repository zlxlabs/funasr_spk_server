"""
Unit 测试专属 fixtures。

仅作用于 tests/unit/（integration 真机路径不受影响——它可能需要 dev .env 里的
硬件 provider 值，如 FUNASR_QWEN3_ASR_ENCODER_PROVIDER=coreml_ane_full）。
"""
import os

# 模块级 config 在 pytest 收集时就会读取 .env 并校验；单测默认关闭通知，避免
# 开发机只配置了开关却没有 webhook 时收集失败。显式环境变量仍优先；python-dotenv
# 的 load_dotenv 默认 override=False，不会用 .env 覆盖这里或用户显式设置的值。
os.environ.setdefault("FUNASR_NOTIFICATION_ENABLED", "false")

import pytest


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Unit 测试 hermetic: 隔离 dev .env 对「默认值 / 无 env」类断言的污染。

    根因（dev 机 6 个 config 测试误挂，CI/纯净环境本不挂）：
      1. config.py 模块级 `config = Config.load_from_file()` 在 import 时调
         load_dotenv()，把 dev .env 的 FUNASR_QWEN3_ASR_ENCODER_PROVIDER=coreml_ane_full
         等塞进 os.environ，并建好一个「带 .env」的全局单例。
      2. 即便测试 monkeypatch.delenv，Config.load_from_file 内部又 load_dotenv() 读回。

    两步净化（走 monkeypatch，测试结束自动复原，零泄漏到 integration）：
      1. load_dotenv → no-op：杜绝 .env 被读回，修「自建 Config + delenv」类测试
         （test_no_env_*）。
      2. 把 import 时已被 .env 污染、单元测试关心的单例字段重置回默认态：修「读全局
         单例」类测试（build_engine_config → config.qwen3.asr_encoder_provider）。

    注：不重建整个 Config（其校验依赖 .env 提供的 webhook 等密钥，重建会误触
    notification 校验 sys.exit），只精准重置受测字段。需要别的 env 的测试仍可 setenv。
    """
    import src.core.config as config_mod

    monkeypatch.setattr(config_mod, "load_dotenv", lambda *args, **kwargs: None)
    # asr_encoder_provider 的 Pydantic 默认是 "auto"（runtime 感知 sentinel）；dev .env
    # 把它污染成 coreml_ane_full，令 build_engine_config 的平台默认断言误挂。重置回默认。
    monkeypatch.setattr(config_mod.config.qwen3, "asr_encoder_provider", "auto")


@pytest.fixture
def mock_vendor_llama_native_module(monkeypatch):
    """模拟 encoder provider 的单测隔离 llama 原生初始化，保留真实 encoder 代码。"""
    import sys
    import types

    if sys.platform != "darwin":
        llama_module = "src.core.vendor.qwen_asr_gguf.inference.llama"
        monkeypatch.setitem(sys.modules, llama_module, types.ModuleType(llama_module))


def test_unit_notification_default_respects_explicit_environment():
    """独立进程验证单测默认值，以及用户显式值不被覆盖。"""
    import subprocess
    import sys

    script = (
        "import os, runpy, sys; runpy.run_path(sys.argv[1]); "
        "print(os.environ.get('FUNASR_NOTIFICATION_ENABLED'))"
    )
    env = os.environ.copy()
    env.pop("FUNASR_NOTIFICATION_ENABLED", None)
    unset_result = subprocess.run(
        [sys.executable, "-c", script, __file__],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert unset_result.stdout.strip() == "false"

    env["FUNASR_NOTIFICATION_ENABLED"] = "true"
    explicit_result = subprocess.run(
        [sys.executable, "-c", script, __file__],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert explicit_result.stdout.strip() == "true"
