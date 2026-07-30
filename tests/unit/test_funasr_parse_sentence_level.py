"""D2: FunASR _parse_and_merge_segments 返回句级（不再引擎内合并）

Linux 本机常无 funasr/torch 完整栈; 注入假模块后 import 类, 只测解析方法.
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _ensure_stub(name: str, **attrs) -> bool:
    """若模块未装则注入 stub, 返回是否新注入."""
    if name in sys.modules:
        return False
    fake = ModuleType(name)
    for k, v in attrs.items():
        setattr(fake, k, v)
    sys.modules[name] = fake
    return True


@pytest.fixture
def transcriber():
    """mock 重依赖后 import FunASRTranscriber, object.__new__ 跳过 __init__."""
    injected: list[str] = []
    for name, attrs in (
        ("funasr", {"AutoModel": MagicMock()}),
        ("torch", {}),
        ("torch.cuda", {}),
        ("ffmpeg", {}),
    ):
        if _ensure_stub(name, **attrs):
            injected.append(name)

    # 子模块路径
    if "torch.cuda" not in sys.modules:
        sys.modules["torch.cuda"] = ModuleType("torch.cuda")
        injected.append("torch.cuda")

    # 清半成品, 让 funasr_transcriber / file_utils / torch_utils 重新 import
    for mod in (
        "src.core.funasr_transcriber",
        "src.utils.file_utils",
        "src.utils.torch_utils",
        "src.core.device_manager",
    ):
        sys.modules.pop(mod, None)

    try:
        from src.core.funasr_transcriber import FunASRTranscriber

        t = object.__new__(FunASRTranscriber)
        yield t
    finally:
        for name in injected:
            sys.modules.pop(name, None)
        sys.modules.pop("src.core.funasr_transcriber", None)


class TestParseSentenceLevel:
    def test_returns_sentence_level_no_merge(self, transcriber):
        """同 speaker + 小 gap 的多句 → 仍返回多条 (canonical 句级)."""
        mock_result = [{
            "sentence_info": [
                {"start": 0, "end": 1000, "text": "你好。", "spk": 0},
                {"start": 1200, "end": 2000, "text": "世界。", "spk": 0},
                {"start": 2100, "end": 3000, "text": "继续。", "spk": 0},
            ]
        }]
        segs = transcriber._parse_and_merge_segments(mock_result)
        assert len(segs) == 3
        assert segs[0].text == "你好。"
        assert segs[1].text == "世界。"
        assert segs[2].text == "继续。"
        assert segs[0].start_time == 0.0
        assert segs[0].end_time == 1.0
        assert segs[1].start_time == 1.2
        assert all(s.speaker == "Speaker1" for s in segs)

    def test_speaker_mapping(self, transcriber):
        mock_result = [{
            "sentence_info": [
                {"start": 0, "end": 500, "text": "A", "spk": 0},
                {"start": 600, "end": 1000, "text": "B", "spk": 1},
            ]
        }]
        segs = transcriber._parse_and_merge_segments(mock_result)
        assert segs[0].speaker == "Speaker1"
        assert segs[1].speaker == "Speaker2"

    def test_empty_sentence_info(self, transcriber):
        assert transcriber._parse_and_merge_segments([{"sentence_info": []}]) == []

    def test_skips_empty_text(self, transcriber):
        mock_result = [{
            "sentence_info": [
                {"start": 0, "end": 500, "text": "  ", "spk": 0},
                {"start": 600, "end": 1000, "text": "有字", "spk": 0},
            ]
        }]
        segs = transcriber._parse_and_merge_segments(mock_result)
        assert len(segs) == 1
        assert segs[0].text == "有字"

    def test_merge_helpers_removed(self, transcriber):
        """引擎层合并方法已删除."""
        assert not hasattr(transcriber, "_merge_consecutive_segments")
        assert not hasattr(transcriber, "_should_merge_segments")
