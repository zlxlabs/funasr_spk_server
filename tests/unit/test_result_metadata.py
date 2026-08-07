"""diarize 开关 (落地步骤 6, E2) — effective options 回显 metadata + 可观测性

- metadata 块: engine / diarize / word_align / language / projected
- serve 层组装 (fresh 出口 + 缓存命中出口), 不随缓存内容存取
- 合并优先级: request > 分片 session 回填 > config > 引擎默认
- 可观测性: projected serve 计数进 cache stats
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.database import DatabaseManager
from src.core.result_projection import build_result_metadata, project_result_nospk
from src.models.schemas import (
    TranscribeOptions,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
    TaskStatus,
)


def make_diarized_result(file_hash="h"):
    return TranscriptionResult(
        task_id="t", file_name="x.wav", file_hash=file_hash, duration=10.0,
        segments=[TranscriptionSegment(start_time=0, end_time=5, text="你好", speaker="Speaker1")],
        speakers=["Speaker1"], processing_time=0.5,
    )


@pytest.fixture
def wa_on():
    from src.core.config import config
    original = config.qwen3.word_align_enabled
    config.qwen3.word_align_enabled = True
    yield
    config.qwen3.word_align_enabled = original


@pytest.fixture
def wa_off():
    from src.core.config import config
    original = config.qwen3.word_align_enabled
    config.qwen3.word_align_enabled = False
    yield
    config.qwen3.word_align_enabled = original


# ==================== build_result_metadata 合并优先级 ====================


class TestBuildResultMetadata:
    def test_full_block_fields(self):
        md = build_result_metadata(
            engine="qwen3",
            options=TranscribeOptions(language="eng", diarize=False, word_align=True),
            projected=True,
        )
        assert md == {
            "engine": "qwen3", "diarize": False, "word_align": True,
            "language": "eng", "projected": True,
        }

    def test_word_align_reads_options_not_config(self, wa_off):
        """决策 1A: metadata.word_align 读 options.word_align (effective), 不读全局 config.
        config 关 (wa_off) 但请求显式开 → metadata True."""
        md = build_result_metadata(engine="qwen3", options=TranscribeOptions(word_align=True))
        assert md["word_align"] is True

    def test_word_align_config_on_but_options_off(self, wa_on):
        """config 开但 options 关 (显式) → metadata False (options 胜)."""
        md = build_result_metadata(engine="qwen3", options=TranscribeOptions(word_align=False))
        assert md["word_align"] is False

    def test_srt_word_align_false_delivered(self):
        """决策 2A: SRT 不挂词, 即使请求 word_align=true, delivered=false."""
        md = build_result_metadata(
            engine="qwen3", options=TranscribeOptions(word_align=True), output_format="srt",
        )
        assert md["word_align"] is False

    def test_failed_alignment_word_align_false_with_error(self):
        """codex #12: 请求 word_align 但实际无词 (has_words=False) → delivered=false + word_align_error."""
        md = build_result_metadata(
            engine="qwen3", options=TranscribeOptions(word_align=True),
            has_words=False, word_align_error="cuda_oom+cpu_fail: boom",
        )
        assert md["word_align"] is False
        assert md["word_align_error"] == "cuda_oom+cpu_fail: boom"

    def test_partial_words_still_delivered_true(self):
        """has_words=True (哪怕部分窗口失败) → delivered True."""
        md = build_result_metadata(
            engine="qwen3", options=TranscribeOptions(word_align=True), has_words=True,
        )
        assert md["word_align"] is True
        assert "word_align_error" not in md

    def test_request_language_wins_over_config(self):
        md = build_result_metadata(
            engine="qwen3", options=TranscribeOptions(language=" jpn ", word_align=True),
        )
        assert md["language"] == "jpn"  # request 优先 + strip 规范化

    def test_config_fallback_language_when_word_align_on(self):
        from src.core.config import config
        md = build_result_metadata(
            engine="qwen3", options=TranscribeOptions(word_align=True),
        )
        assert md["language"] == config.qwen3.word_align_language

    def test_funasr_word_align_always_false(self):
        md = build_result_metadata(
            engine="funasr", options=TranscribeOptions(word_align=True),
        )
        assert md["word_align"] is False

    def test_funasr_json_echoes_segment_merge_max_span(self):
        """D5: funasr + JSON 出口 metadata 带 segment_merge_max_span_sec."""
        from src.core.config import config
        md = build_result_metadata(
            engine="funasr", options=TranscribeOptions(), output_format="json",
        )
        assert md["segment_merge_max_span_sec"] == config.transcription.segment_merge_max_span_sec

    def test_qwen3_and_srt_no_segment_merge_key(self):
        """未应用 merge 视图的出口不带该键."""
        md_q = build_result_metadata(engine="qwen3", options=TranscribeOptions())
        assert "segment_merge_max_span_sec" not in md_q
        md_srt = build_result_metadata(
            engine="funasr", options=TranscribeOptions(), output_format="srt",
        )
        assert "segment_merge_max_span_sec" not in md_srt

    def test_defaults(self, wa_off):
        md = build_result_metadata(engine="qwen3", options=TranscribeOptions())
        assert md["diarize"] is True
        assert md["word_align"] is False
        assert md["projected"] is False
        assert "word_align_error" not in md


# ==================== fresh 出口组装 ====================


class TestFreshExitMetadata:
    @pytest.mark.asyncio
    async def test_fresh_funasr_nospk_metadata(self, wa_off, tmp_path):
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="t-md", file_name="x.wav", file_path=str(tmp_path / "x.wav"),
            file_size=100, file_hash="h", engine="funasr",
            options=TranscribeOptions(diarize=False),
        )
        mgr.tasks["t-md"] = task
        (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=(make_diarized_result(), {}))

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("t-md")

        md = task.result.metadata
        assert md["engine"] == "funasr"
        assert md["diarize"] is False
        assert md["projected"] is True, "funasr 照算出口投影 → projected=true"
        assert "context_applied" not in md
        # 入库的结果不带 metadata 污染 (save 在投影/组装之前)
        saved = mock_db.save_result.call_args.args[0]
        assert saved.metadata is None

    @pytest.mark.asyncio
    async def test_fresh_diarize_on_metadata(self, wa_off, tmp_path):
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="t-md2", file_name="x.wav", file_path=str(tmp_path / "x.wav"),
            file_size=100, file_hash="h", engine="qwen3",
        )
        mgr.tasks["t-md2"] = task
        (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=(make_diarized_result(), {}))

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("t-md2")

        md = task.result.metadata
        assert md == {
            "engine": "qwen3", "diarize": True, "word_align": False,
            "language": None, "projected": False,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("output_format", ["json", "srt"])
    async def test_funasr_terms_metadata_after_success(self, wa_off, tmp_path, output_format):
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        path = tmp_path / "x.wav"
        path.write_bytes(b"\0" * 100)
        task = TranscriptionTask(
            task_id=f"t-terms-{output_format}", file_name="x.wav", file_path=str(path),
            file_size=100, file_hash="h", engine="funasr",
            output_format=output_format, options=TranscribeOptions(terms=["Alpha", "Beta"]),
        )
        mgr.tasks[task.task_id] = task
        source = make_diarized_result("h")
        returned = (source, []) if output_format == "json" else {
            "format": "srt", "content": "1\ntext", "file_name": "x.wav", "file_hash": "h",
            "duration": 10.0, "processing_time": 0.5, "raw_result": [], "segments": source.segments,
        }
        fake = MagicMock(transcribe=AsyncMock(return_value=returned))
        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake), \
             patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
             patch.object(mgr, "_notify_task_complete", new=AsyncMock()), \
             patch.object(mgr, "_maybe_delete_task_file", new=AsyncMock()):
            await mgr._process_task(task.task_id)

        assert task.status is TaskStatus.COMPLETED
        assert task.result.metadata["context_applied"] is True
        assert task.result.metadata["terms_count"] == 2

    @pytest.mark.asyncio
    async def test_funasr_terms_failure_has_no_metadata(self, wa_off, tmp_path):
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        path = tmp_path / "x.wav"
        path.write_bytes(b"\0" * 100)
        task = TranscriptionTask(
            task_id="t-terms-fail", file_name="x.wav", file_path=str(path), file_size=100,
            file_hash="h", engine="funasr", retry_count=99,
            options=TranscribeOptions(terms=["Alpha"]),
        )
        mgr.tasks[task.task_id] = task
        fake = MagicMock(transcribe=AsyncMock(side_effect=RuntimeError("model failed")))
        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake), \
             patch.object(mgr, "_notify_task_failed", new=AsyncMock()), \
             patch.object(mgr, "_maybe_delete_task_file", new=AsyncMock()), \
             patch.object(mgr, "_send_wework_notification", new=AsyncMock()):
            await mgr._process_task(task.task_id)

        assert task.status is TaskStatus.FAILED
        assert task.result is None


def test_terms_metadata_is_supported_engine_only():
    funasr = build_result_metadata(engine="funasr", options=TranscribeOptions(terms=["Alpha"]))
    qwen = build_result_metadata(engine="qwen3", options=TranscribeOptions(terms=["Alpha"]))
    assert funasr["context_applied"] is True
    assert funasr["terms_count"] == 1
    assert qwen["context_applied"] is True
    assert qwen["terms_count"] == 1


@pytest.mark.asyncio
async def test_qwen_terms_failure_has_no_metadata(wa_off, tmp_path):
    from src.core.task_manager import TaskManager

    mgr = TaskManager()
    path = tmp_path / "qwen.wav"
    path.write_bytes(b"\0")
    task = TranscriptionTask(
        task_id="qwen-terms-fail", file_name=path.name, file_path=str(path),
        file_size=1, file_hash="qwen-fail", engine="qwen3",
        options=TranscribeOptions(terms=["Alpha"]), retry_count=99,
    )
    mgr.tasks[task.task_id] = task
    fake = MagicMock(transcribe=AsyncMock(side_effect=RuntimeError("model failed")))
    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake), \
         patch.object(mgr, "_notify_task_failed", new=AsyncMock()), \
         patch.object(mgr, "_maybe_delete_task_file", new=AsyncMock()), \
         patch.object(mgr, "_send_wework_notification", new=AsyncMock()):
        await mgr._process_task(task.task_id)

    assert task.status is TaskStatus.FAILED
    assert task.result is None


# ==================== 缓存命中出口组装 ====================


class TestCacheHitMetadata:
    @pytest.mark.asyncio
    async def test_submit_task_json_hit_composes_metadata(self, wa_off, tmp_path):
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="t-ch", file_name="x.wav", file_path="", file_size=1,
            file_hash="h-ch", engine="qwen3", options=TranscribeOptions(diarize=False),
        )
        mgr.tasks["t-ch"] = task
        f = tmp_path / "x.wav"
        f.write_bytes(b"\x00" * 10)

        projected = project_result_nospk(make_diarized_result("h-ch"))
        projected.metadata = {"projected": True}  # db 出口标记
        with patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.get_cached_result = AsyncMock(return_value=projected)
            with patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr.submit_task("t-ch", str(f))

        md = task.result.metadata
        assert md["engine"] == "qwen3"
        assert md["diarize"] is False
        assert md["projected"] is True

    @pytest.mark.asyncio
    async def test_submit_task_srt_hit_composes_metadata(self, wa_off, tmp_path):
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="t-cs", file_name="x.wav", file_path="", file_size=1,
            file_hash="h-cs", engine="qwen3", output_format="srt",
            options=TranscribeOptions(diarize=False),
        )
        mgr.tasks["t-cs"] = task
        f = tmp_path / "x.wav"
        f.write_bytes(b"\x00" * 10)

        srt_hit = {
            "format": "srt", "content": "1\n00:00:00,000 --> 00:00:05,000\n你好\n",
            "file_name": "x.wav", "file_hash": "h-cs", "duration": 10.0,
            "projected": True,
        }
        with patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.get_cached_result = AsyncMock(return_value=srt_hit)
            with patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr.submit_task("t-cs", str(f))

        md = task.result.metadata
        assert md["projected"] is True
        assert md["diarize"] is False


# ==================== projected serve 计数 (可观测性) ====================


class TestProjectedServeCounter:
    @pytest.mark.asyncio
    async def test_counter_increments_on_projection(self, tmp_path):
        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.init_db()
        await db.save_result(make_diarized_result("h-cnt"), raw_result=None, engine="qwen3")
        stats0 = await db.get_cache_stats()
        assert stats0.get("projected_serves") == 0
        await db.get_cached_result(
            "h-cnt", engine="qwen3+nospk", allow_cross_engine=False,
            options=TranscribeOptions(diarize=False),
        )
        stats1 = await db.get_cache_stats()
        assert stats1.get("projected_serves") == 1
