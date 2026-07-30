"""TranscriptionConfig.segment_merge_* 默认值 + env override"""
from __future__ import annotations

import os

import pytest

from src.core.config import TranscriptionConfig, Config


class TestSegmentMergeConfigDefaults:
    def test_defaults(self):
        cfg = TranscriptionConfig()
        assert cfg.segment_merge_gap_sec == 3.0
        assert cfg.segment_merge_max_span_sec == 120.0

    def test_init_override(self):
        cfg = TranscriptionConfig(segment_merge_gap_sec=1.5, segment_merge_max_span_sec=0.0)
        assert cfg.segment_merge_gap_sec == 1.5
        assert cfg.segment_merge_max_span_sec == 0.0


class TestSegmentMergeEnvOverride:
    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_SEGMENT_MERGE_GAP_SEC", "1.0")
        monkeypatch.setenv("FUNASR_SEGMENT_MERGE_MAX_SPAN_SEC", "60")
        # 清其它可能污染的 FUNASR_* (保留本测试需要的)
        for key in list(os.environ.keys()):
            if key.startswith("FUNASR_") and key not in (
                "FUNASR_SEGMENT_MERGE_GAP_SEC",
                "FUNASR_SEGMENT_MERGE_MAX_SPAN_SEC",
            ):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("FUNASR_NOTIFICATION_ENABLED", "false")
        monkeypatch.setenv("FUNASR_AUTH_ENABLED", "false")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.transcription.segment_merge_gap_sec == 1.0
        assert cfg.transcription.segment_merge_max_span_sec == 60.0
