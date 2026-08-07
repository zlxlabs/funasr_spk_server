"""Shared FUNASR_PROFILE values for application and read-only doctor consumers."""
from __future__ import annotations

from typing import Any


# Profile values are data only; consumers own their validation and logging.
PROFILES: dict[str, dict[str, Any]] = {
    "mac_prod": {
        "server": {"port": 8767},
        "transcription": {"default_engine": "funasr", "qwen3_pool_size": 1},
        "qwen3": {"asr_encoder_provider": "coreml_ane_full"},
    },
    "mac_dev": {
        "server": {"port": 8867},
        "transcription": {"default_engine": "funasr", "qwen3_pool_size": 1},
        "qwen3": {"asr_encoder_provider": "coreml_ane_full"},
        "logging": {"level": "DEBUG"},
    },
    "cuda_prod": {
        "transcription": {"default_engine": "qwen3", "qwen3_pool_size": 1},
        "qwen3": {"asr_encoder_provider": "cuda"},
    },
    "cuda_dev": {
        "server": {"port": 8867},
        "transcription": {"default_engine": "qwen3", "qwen3_pool_size": 1},
        "qwen3": {"asr_encoder_provider": "cuda"},
        "logging": {"level": "DEBUG"},
    },
}
