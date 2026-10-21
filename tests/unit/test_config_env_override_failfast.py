"""非法环境变量转换不得静默回退到默认配置。"""
from __future__ import annotations

import os

import pytest

_original_notification_enabled = os.environ.get("FUNASR_NOTIFICATION_ENABLED")
os.environ["FUNASR_NOTIFICATION_ENABLED"] = "false"
from src.core.config import Config
if _original_notification_enabled is None:
    os.environ.pop("FUNASR_NOTIFICATION_ENABLED", None)
else:
    os.environ["FUNASR_NOTIFICATION_ENABLED"] = _original_notification_enabled


def test_invalid_integer_environment_variable_fails_config_loading(monkeypatch, tmp_path):
    """Config 的公开加载入口应抛错，且异常必须指出出错变量名。"""
    env_name = "FUNASR_SERVER_MAX_CONNECTIONS"
    monkeypatch.setenv("FUNASR_NOTIFICATION_ENABLED", "false")
    monkeypatch.setenv(env_name, "not-an-integer")
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=env_name):
        Config.load_from_file(str(config_file))


def test_invalid_qwen_speaker_integer_environment_variable_fails_loading(monkeypatch, tmp_path):
    """特殊解析的整数环境变量也不得被单独吞掉。"""
    env_name = "FUNASR_QWEN3_NUM_SPEAKERS"
    monkeypatch.setenv("FUNASR_NOTIFICATION_ENABLED", "false")
    monkeypatch.setenv(env_name, "not-an-integer")
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=env_name):
        Config.load_from_file(str(config_file))
