"""毫秒输入渲染 SRT 时先四舍五入总毫秒，再拆分时间字段。"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def funasr_transcriber(monkeypatch):
    for name, attrs in (
        ("funasr", {"AutoModel": MagicMock()}),
        ("torch", {}),
        ("torch.cuda", {}),
        ("ffmpeg", {}),
    ):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    for name in (
        "src.core.funasr_transcriber",
        "src.utils.file_utils",
        "src.utils.torch_utils",
        "src.core.device_manager",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    from src.core.funasr_transcriber import FunASRTranscriber

    yield object.__new__(FunASRTranscriber)
    monkeypatch.delitem(sys.modules, "src.core.funasr_transcriber", raising=False)


@pytest.fixture
def database():
    from src.core.database import DatabaseManager

    return DatabaseManager(db_path=":memory:")


@pytest.mark.parametrize("formatter", ["funasr", "database"])
def test_srt_millisecond_rounding_and_second_carry(formatter, funasr_transcriber, database):
    """两个 raw FunASR SRT 入口都保留整数毫秒并正确处理进位。"""
    sentences = [
        {"start": 1234.9, "end": 999.6, "text": "fractional", "spk": 0},
        {"start": 0, "end": 1234, "text": "integer", "spk": 0},
    ]
    raw_result = [{"sentence_info": sentences}]
    renderer = (
        funasr_transcriber._generate_srt_from_raw_result
        if formatter == "funasr"
        else database._generate_srt_from_raw_result
    )

    srt = renderer(raw_result)

    assert "00:00:01,235 --> 00:00:01,000" in srt
    assert "00:00:00,000 --> 00:00:01,234" in srt
