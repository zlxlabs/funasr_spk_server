"""D2: FunASR _parse_and_merge_segments 返回句级（不再引擎内合并）

Linux 本机常无 funasr/torch 完整栈; 注入假模块后 import 类, 只测解析方法.
"""
from __future__ import annotations

import sys
import threading
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

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

    @pytest.mark.parametrize("result", [
        {"text": "recognition text"},
        {"text": "recognition text", "sentence_info": []},
    ])
    def test_nonempty_text_without_sentences_fails_json_and_srt(self, transcriber, result):
        with pytest.raises(ValueError, match="FunASR result invalid-shape"):
            transcriber._parse_and_merge_segments([result])
        with pytest.raises(ValueError, match="FunASR result invalid-shape"):
            transcriber._generate_srt_from_raw_result([result])

    @pytest.mark.parametrize("result", [
        {"text": ""},
        {"text": "", "sentence_info": []},
    ])
    def test_explicit_empty_text_without_sentences_is_empty_success(self, transcriber, result):
        assert transcriber._parse_and_merge_segments([result]) == []
        assert transcriber._generate_srt_from_raw_result([result]) == ""

    @pytest.mark.parametrize("result", [[], [None], None, {}])
    def test_unknown_or_missing_text_shape_fails(self, transcriber, result):
        with pytest.raises(ValueError, match="FunASR result invalid-shape"):
            transcriber._parse_and_merge_segments(result)
        with pytest.raises(ValueError, match="FunASR result invalid-shape"):
            transcriber._generate_srt_from_raw_result(result)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("output_format", ["json", "srt"])
    async def test_transcribe_rejects_invalid_result_without_sensitive_logs(
        self, transcriber, monkeypatch, output_format
    ):
        import src.core.funasr_transcriber as funasr_module
        import src.utils.file_utils as file_utils

        test_logger = MagicMock()
        monkeypatch.setattr(funasr_module, "logger", test_logger)
        transcriber.is_initialized = True
        transcriber.concurrency_mode = "lock"
        transcriber.config = {"funasr": {"batch_size_s": 20}, "transcription": {}}
        transcriber._model_lock = threading.Lock()
        transcriber.model = MagicMock()
        transcriber.model.generate.return_value = [{"text": "private recognition payload"}]
        monkeypatch.setattr(funasr_module, "get_audio_duration", lambda _path: 1.0)
        monkeypatch.setattr(funasr_module, "release_accelerator_memory", lambda **_kwargs: None)
        monkeypatch.setattr(file_utils, "calculate_file_hash", AsyncMock(return_value="hash"))

        with pytest.raises(Exception, match="FunASR result invalid-shape"):
            await transcriber.transcribe(
                audio_path="/tmp/private-audio.wav",
                task_id="task-42",
                output_format=output_format,
            )

        log_arguments = [
            str(value)
            for method in (test_logger.debug, test_logger.info, test_logger.warning, test_logger.error)
            for call in method.call_args_list
            for value in call.args
        ]
        logs = "\n".join(log_arguments)
        assert any(
            call.args[:2] == (
                "FunASR result invalid_shape: task_id={} state=invalid_shape",
                "task-42",
            )
            for call in test_logger.error.call_args_list
        )
        assert "invalid_shape" in logs
        assert "private recognition payload" not in logs
        assert "private-audio.wav" not in logs

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
