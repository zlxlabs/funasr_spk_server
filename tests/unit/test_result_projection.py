"""diarize 开关 (落地步骤 5, D3+T-A / E1+T2) — result_projection 纯函数 + 双出口投影

- 纯函数 project_result_nospk: speaker 抹 null + speakers=[] (words/时间保留, 幂等)
- 出口 1: db_manager.get_cached_result — nospk 请求 exact tag miss → 同引擎同
  wa-tag diarized 行现场投影返回 (标 projected:true), **不回写**; funasr 免折维
  行 (本身 diarized) 出口投影; SRT 从投影 segments 重渲染无前缀 (raw 路径旁路)
- 出口 2: task_manager fresh 结果 — funasr 照算后出口投影, 缓存仍存 diarized 原结果
- 容错: 投影回退读到坏行 → 具名异常当 miss + warn, 不静默吞
- metadata 不入库: save_result exclude, 缓存读出不继承上次请求的 projected
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from src.core.database import DatabaseManager
from src.core.result_projection import (
    cache_hit_metadata,
    merge_segments_view,
    project_result_nospk,
    segments_to_srt_text,
)
from src.models.schemas import (
    TranscribeOptions,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


def make_diarized_result(file_hash: str = "h", with_words: bool = False) -> TranscriptionResult:
    words = [WordTimestamp(text="你", start=0.1, end=0.3)] if with_words else None
    return TranscriptionResult(
        task_id="t", file_name="x.wav", file_hash=file_hash, duration=10.0,
        segments=[
            TranscriptionSegment(start_time=0.0, end_time=5.0, text="你好", speaker="Speaker1", words=words),
            TranscriptionSegment(start_time=5.0, end_time=10.0, text="再见", speaker="Speaker2"),
        ],
        speakers=["Speaker1", "Speaker2"],
        processing_time=0.5,
    )


NOSPK = TranscribeOptions(diarize=False)
SPK = TranscribeOptions(diarize=True)


# ==================== 纯函数 ====================


class TestProjectResultNospk:
    def test_projects_speaker_to_none_and_empty_speakers(self):
        r = make_diarized_result(with_words=True)
        p = project_result_nospk(r)
        assert all(s.speaker is None for s in p.segments)
        assert p.speakers == []
        # 时间 / 文本 / words 保留
        assert p.segments[0].text == "你好"
        assert p.segments[0].words[0].text == "你"
        assert p.duration == 10.0

    def test_original_not_mutated(self):
        r = make_diarized_result()
        project_result_nospk(r)
        assert r.segments[0].speaker == "Speaker1"
        assert r.speakers == ["Speaker1", "Speaker2"]

    def test_idempotent(self):
        r = make_diarized_result()
        p1 = project_result_nospk(r)
        p2 = project_result_nospk(p1)
        assert p2.speakers == []
        assert all(s.speaker is None for s in p2.segments)

    def test_segments_to_srt_text_no_speaker(self):
        p = project_result_nospk(make_diarized_result())
        srt = segments_to_srt_text(p.segments)
        assert "Speaker" not in srt
        assert "你好" in srt

    def test_segments_to_srt_text_with_speaker(self):
        srt = segments_to_srt_text(make_diarized_result().segments)
        assert "Speaker1:你好" in srt


# ==================== 共享纯函数: cache_hit_metadata (3 处出口去重) ====================


class TestCacheHitMetadata:
    """缓存命中出口共享纯逻辑: projected 提取 + metadata 构建 + SRT-dict 校验.
    codex #7/#8: 统一用 get 不 pop (不改 cached_result); SRT 无副作用; 控制流各出口自理.
    """

    def test_json_extracts_projected_and_builds_metadata(self):
        r = make_diarized_result("h")
        r.metadata = {"projected": True}
        md, projected, srt_ok = cache_hit_metadata(
            r, engine="qwen3", options=SPK, output_format="json"
        )
        assert srt_ok is True
        assert projected is True
        assert md["projected"] is True
        assert md["engine"] == "qwen3"
        assert md["diarize"] is True

    def test_json_projected_absent_defaults_false(self):
        r = make_diarized_result("h")  # metadata=None
        md, projected, srt_ok = cache_hit_metadata(
            r, engine="qwen3", options=SPK, output_format="json"
        )
        assert projected is False
        assert md["projected"] is False

    def test_srt_dict_valid_extracts_projected_without_mutating(self):
        cached = {"format": "srt", "content": "1\n...\n你好\n",
                  "file_hash": "h", "projected": True, "duration": 10.0}
        md, projected, srt_ok = cache_hit_metadata(
            cached, engine="funasr", options=NOSPK, output_format="srt"
        )
        assert srt_ok is True
        assert projected is True
        assert md["projected"] is True and md["diarize"] is False
        # codex #7/#8: 不 pop, 原 dict 的 projected 仍在 (调用方负责出口时排除)
        assert cached.get("projected") is True

    def test_srt_non_dict_is_invalid(self):
        """SRT 请求但缓存是 TranscriptionResult (非 srt-dict) → srt_ok=False, 调用方跳过缓存"""
        r = make_diarized_result("h")
        md, projected, srt_ok = cache_hit_metadata(
            r, engine="qwen3", options=NOSPK, output_format="srt"
        )
        assert srt_ok is False
        assert md is None

    def test_srt_dict_wrong_format_is_invalid(self):
        cached = {"format": "json", "content": "x"}
        md, projected, srt_ok = cache_hit_metadata(
            cached, engine="qwen3", options=NOSPK, output_format="srt"
        )
        assert srt_ok is False
        assert md is None


# ==================== 出口 1: get_cached_result ====================


@pytest.fixture
async def db(tmp_path):
    d = DatabaseManager(db_path=str(tmp_path / "t.db"))
    await d.init_db()
    return d


class TestCachedProjection:
    @pytest.mark.asyncio
    async def test_exact_nospk_row_hit_not_projected(self, db):
        """exact qwen3+nospk 行命中 → 原样返回 (真算的 nospk), projected=False"""
        nospk_result = project_result_nospk(make_diarized_result("h1"))
        await db.save_result(nospk_result, raw_result=None, engine="qwen3+nospk")
        cached = await db.get_cached_result("h1", engine="qwen3+nospk",
                                            allow_cross_engine=False, options=NOSPK)
        assert cached is not None
        assert cached.speakers == []
        assert (cached.metadata or {}).get("projected") is False

    @pytest.mark.asyncio
    async def test_nospk_miss_falls_back_to_diarized_row_projected(self, db):
        """E1: exact +nospk miss → 同引擎 diarized 行现场投影, 标 projected:true"""
        await db.save_result(make_diarized_result("h2"), raw_result=None, engine="qwen3")
        cached = await db.get_cached_result("h2", engine="qwen3+nospk",
                                            allow_cross_engine=False, options=NOSPK)
        assert cached is not None
        assert all(s.speaker is None for s in cached.segments)
        assert cached.speakers == []
        assert cached.metadata.get("projected") is True

    @pytest.mark.asyncio
    async def test_projection_does_not_write_back(self, db):
        """T2: 不回写 — 投影 serve 后缓存里仍只有 diarized 行"""
        await db.save_result(make_diarized_result("h3"), raw_result=None, engine="qwen3")
        await db.get_cached_result("h3", engine="qwen3+nospk",
                                   allow_cross_engine=False, options=NOSPK)
        async with aiosqlite.connect(db.db_path) as conn:
            cursor = await conn.execute(
                "SELECT engine FROM transcription_cache WHERE file_hash = ?", ("h3",)
            )
            engines = [r[0] for r in await cursor.fetchall()]
        assert engines == ["qwen3"], f"不准回写 nospk 行: {engines}"

    @pytest.mark.asyncio
    async def test_nospk_fallback_respects_wa_tag(self, db):
        """带 wa tag 的 nospk: 回退显式只查同引擎同 wa-tag 行 (T-D #7)"""
        await db.save_result(make_diarized_result("h4"), raw_result=None, engine="qwen3+wa:eng")
        # 同 wa-tag → 命中投影
        hit = await db.get_cached_result("h4", engine="qwen3+wa:eng+nospk",
                                         allow_cross_engine=False, options=NOSPK)
        assert hit is not None and hit.metadata.get("projected") is True
        # 不同 wa-tag (plain qwen3 行不存在) → miss
        miss = await db.get_cached_result("h4", engine="qwen3+nospk",
                                          allow_cross_engine=False, options=NOSPK)
        assert miss is None

    @pytest.mark.asyncio
    async def test_nospk_fallback_never_crosses_engine(self, db):
        """qwen3+nospk 回退禁 cross-engine: 只有 funasr 行时必须 miss"""
        await db.save_result(make_diarized_result("h5"), raw_result=None, engine="funasr")
        miss = await db.get_cached_result("h5", engine="qwen3+nospk",
                                          allow_cross_engine=False, options=NOSPK)
        assert miss is None

    @pytest.mark.asyncio
    async def test_funasr_nospk_projects_diarized_row_at_exit(self, db):
        """funasr 免折维 (D4): exact 'funasr' 行本身 diarized, 出口投影"""
        await db.save_result(make_diarized_result("h6"), raw_result=None, engine="funasr")
        cached = await db.get_cached_result("h6", engine="funasr", options=NOSPK)
        assert cached is not None
        assert all(s.speaker is None for s in cached.segments)
        assert cached.speakers == []
        assert cached.metadata.get("projected") is True

    @pytest.mark.asyncio
    async def test_diarize_on_request_unaffected(self, db):
        """diarize=True 请求行为不变 (不投影)"""
        await db.save_result(make_diarized_result("h7"), raw_result=None, engine="qwen3")
        cached = await db.get_cached_result("h7", engine="qwen3", options=SPK)
        assert cached.speakers == ["Speaker1", "Speaker2"]

    @pytest.mark.asyncio
    async def test_srt_nospk_funasr_bypasses_raw_and_has_no_prefix(self, db):
        """SRT + nospk: 旁路 funasr raw sentence_info 路径, 从投影 segments 重渲染无前缀"""
        funasr_raw = [{"sentence_info": [
            {"start": 0, "end": 5000, "text": "你好", "spk": 0},
        ]}]
        await db.save_result(make_diarized_result("h8"), raw_result=funasr_raw, engine="funasr")
        cached = await db.get_cached_result("h8", output_format="srt", engine="funasr", options=NOSPK)
        assert cached is not None
        assert cached["format"] == "srt"
        assert "Speaker" not in cached["content"]
        assert "你好" in cached["content"]
        assert cached.get("projected") is True

    @pytest.mark.asyncio
    async def test_srt_nospk_fallback_projection(self, db):
        """SRT + qwen3+nospk miss → diarized 行投影重渲染无前缀"""
        await db.save_result(make_diarized_result("h9"), raw_result=None, engine="qwen3")
        cached = await db.get_cached_result("h9", output_format="srt", engine="qwen3+nospk",
                                            allow_cross_engine=False, options=NOSPK)
        assert cached is not None
        assert "Speaker" not in cached["content"]
        assert cached.get("projected") is True

    @pytest.mark.asyncio
    async def test_corrupt_diarized_row_treated_as_miss_with_warn(self, db):
        """D6 容错: 投影回退读到坏行 (非法 JSON) → 当 miss 重算, 不抛不吞"""
        async with aiosqlite.connect(db.db_path) as conn:
            await conn.execute(
                "INSERT INTO transcription_cache (file_hash, file_name, result, engine) "
                "VALUES (?, ?, ?, ?)",
                ("h10", "x.wav", "{broken", "qwen3"),
            )
            await conn.commit()
        miss = await db.get_cached_result("h10", engine="qwen3+nospk",
                                          allow_cross_engine=False, options=NOSPK)
        assert miss is None

    @pytest.mark.asyncio
    async def test_funasr_json_applies_merge_view(self, db):
        """缓存命中 funasr 行 JSON: 应用 merge 视图 (同 spk 相邻句合并)."""
        sentence_level = TranscriptionResult(
            task_id="t", file_name="x.wav", file_hash="h-merge-fa", duration=10.0,
            segments=[
                TranscriptionSegment(start_time=0.0, end_time=1.0, text="你好。", speaker="Speaker1"),
                TranscriptionSegment(start_time=1.2, end_time=2.0, text="世界。", speaker="Speaker1"),
                TranscriptionSegment(start_time=5.0, end_time=6.0, text="换人", speaker="Speaker2"),
            ],
            speakers=["Speaker1", "Speaker2"], processing_time=0.5,
        )
        await db.save_result(sentence_level, raw_result=None, engine="funasr")
        cached = await db.get_cached_result("h-merge-fa", engine="funasr", options=SPK)
        assert cached is not None
        # Speaker1 两句合并, Speaker2 独立
        assert len(cached.segments) == 2
        assert cached.segments[0].text == "你好。世界。"
        assert cached.segments[0].speaker == "Speaker1"
        assert cached.segments[1].text == "换人"

    @pytest.mark.asyncio
    async def test_qwen3_json_does_not_apply_merge_view(self, db):
        """qwen3 行不过 merge 视图 (行为与改前一致)."""
        sentence_level = TranscriptionResult(
            task_id="t", file_name="x.wav", file_hash="h-merge-q", duration=10.0,
            segments=[
                TranscriptionSegment(start_time=0.0, end_time=1.0, text="A", speaker="Speaker1"),
                TranscriptionSegment(start_time=1.1, end_time=2.0, text="B", speaker="Speaker1"),
            ],
            speakers=["Speaker1"], processing_time=0.5,
        )
        await db.save_result(sentence_level, raw_result=None, engine="qwen3")
        cached = await db.get_cached_result("h-merge-q", engine="qwen3", options=SPK)
        assert len(cached.segments) == 2
        assert cached.segments[0].text == "A"
        assert cached.segments[1].text == "B"

    @pytest.mark.asyncio
    async def test_funasr_nospk_merge_then_project_order(self, db):
        """顺序铁律: 先 merge 后 nospk — merge 用 speaker, 再抹 speaker."""
        sentence_level = TranscriptionResult(
            task_id="t", file_name="x.wav", file_hash="h-merge-n", duration=10.0,
            segments=[
                TranscriptionSegment(start_time=0.0, end_time=1.0, text="A", speaker="Speaker1"),
                TranscriptionSegment(start_time=1.1, end_time=2.0, text="B", speaker="Speaker1"),
                TranscriptionSegment(start_time=2.1, end_time=3.0, text="C", speaker="Speaker2"),
            ],
            speakers=["Speaker1", "Speaker2"], processing_time=0.5,
        )
        await db.save_result(sentence_level, raw_result=None, engine="funasr")
        cached = await db.get_cached_result("h-merge-n", engine="funasr", options=NOSPK)
        # 若反序 (先 nospk) 会把 null speaker 全并成 1 条; 正确序: 2 条 (AB 合并 + C)
        assert len(cached.segments) == 2
        assert cached.segments[0].text == "AB"
        assert cached.segments[1].text == "C"
        assert all(s.speaker is None for s in cached.segments)
        assert cached.speakers == []

    @pytest.mark.asyncio
    async def test_funasr_nospk_srt_skips_merge_sentence_level(self, db):
        """nospk SRT 旁路: 不过 merge 视图, 句级直接渲染."""
        sentence_level = TranscriptionResult(
            task_id="t", file_name="x.wav", file_hash="h-srt-m", duration=10.0,
            segments=[
                TranscriptionSegment(start_time=0.0, end_time=1.0, text="句一", speaker="Speaker1"),
                TranscriptionSegment(start_time=1.1, end_time=2.0, text="句二", speaker="Speaker1"),
            ],
            speakers=["Speaker1"], processing_time=0.5,
        )
        await db.save_result(sentence_level, raw_result=None, engine="funasr")
        cached = await db.get_cached_result(
            "h-srt-m", output_format="srt", engine="funasr", options=NOSPK,
        )
        assert cached is not None
        # 句级两条都出现在 SRT (未合并)
        assert "句一" in cached["content"]
        assert "句二" in cached["content"]
        # 两条独立 cue (编号 1 和 2)
        assert "\n2\n" in cached["content"] or cached["content"].startswith("1\n")
        lines = [ln for ln in cached["content"].splitlines() if ln.strip().isdigit()]
        assert lines == ["1", "2"]


# ==================== metadata 不入库 ====================


class TestMetadataNotPersisted:
    @pytest.mark.asyncio
    async def test_save_result_excludes_metadata(self, db):
        r = make_diarized_result("h11")
        r.metadata = {"projected": True, "engine": "qwen3"}
        await db.save_result(r, raw_result=None, engine="qwen3")
        async with aiosqlite.connect(db.db_path) as conn:
            cursor = await conn.execute(
                "SELECT result FROM transcription_cache WHERE file_hash = ?", ("h11",)
            )
            row_json = (await cursor.fetchone())[0]
        assert "projected" not in row_json, "metadata 是请求级属性, 禁止入库 (T-D #9)"

    @pytest.mark.asyncio
    async def test_cache_read_does_not_inherit_projected(self, db):
        """缓存读出不继承上次请求的 projected"""
        r = make_diarized_result("h12")
        r.metadata = {"projected": True}
        await db.save_result(r, raw_result=None, engine="qwen3")
        cached = await db.get_cached_result("h12", engine="qwen3", options=SPK)
        assert not (cached.metadata or {}).get("projected")


# ==================== 出口 2: task_manager fresh 结果投影 ====================


class TestFreshExitProjection:
    def _funasr_task(self, tmp_path, output_format="json"):
        from src.models.schemas import TranscriptionTask
        task = TranscriptionTask(
            task_id="t-fr", file_name="x.wav", file_path=str(tmp_path / "x.wav"),
            file_size=100, file_hash="h-fr", engine="funasr",
            output_format=output_format,
            options=TranscribeOptions(diarize=False),
        )
        (tmp_path / "x.wav").write_bytes(b"\x00" * 100)
        return task

    @pytest.mark.asyncio
    async def test_funasr_fresh_json_projected_but_cache_keeps_diarized(self, tmp_path):
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        task = self._funasr_task(tmp_path)
        mgr.tasks["t-fr"] = task

        diarized = make_diarized_result("h-fr")
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=(diarized, {"sentence_info": []}))

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("t-fr")

        # 客户端看到投影后结果
        assert all(s.speaker is None for s in task.result.segments)
        assert task.result.speakers == []
        # 缓存存 diarized 原结果 (D4: 一行通吃)
        saved = mock_db.save_result.call_args.args[0]
        assert saved.speakers == ["Speaker1", "Speaker2"]
        assert mock_db.save_result.call_args.kwargs.get("engine") == "funasr"

    @pytest.mark.asyncio
    async def test_funasr_fresh_json_merge_view_cache_sentence_level(self, tmp_path):
        """fresh funasr JSON: 客户端见 merge 视图; 缓存存句级; 顺序 merge→nospk."""
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        task = self._funasr_task(tmp_path)
        mgr.tasks["t-fr"] = task

        sentence_level = TranscriptionResult(
            task_id="t-fr", file_name="x.wav", file_hash="h-fr", duration=10.0,
            segments=[
                TranscriptionSegment(start_time=0.0, end_time=1.0, text="A", speaker="Speaker1"),
                TranscriptionSegment(start_time=1.1, end_time=2.0, text="B", speaker="Speaker1"),
                TranscriptionSegment(start_time=2.1, end_time=3.0, text="C", speaker="Speaker2"),
            ],
            speakers=["Speaker1", "Speaker2"], processing_time=0.5,
        )
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(
            return_value=(sentence_level, {"sentence_info": []})
        )

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("t-fr")

        # 客户端: merge 后 nospk → 2 段, speaker null
        assert len(task.result.segments) == 2
        assert task.result.segments[0].text == "AB"
        assert task.result.segments[1].text == "C"
        assert all(s.speaker is None for s in task.result.segments)
        # 缓存: 句级 3 段 + speakers 保留 (先 save 再投影)
        saved = mock_db.save_result.call_args.args[0]
        assert len(saved.segments) == 3
        assert saved.segments[0].text == "A"
        assert saved.speakers == ["Speaker1", "Speaker2"]
        # metadata 回显 cap
        assert task.result.metadata.get("segment_merge_max_span_sec") == 120.0

    @pytest.mark.asyncio
    async def test_funasr_fresh_srt_does_not_apply_merge(self, tmp_path):
        """SRT 分支不应用 merge 视图 (content 来自引擎 raw 路径 / 投影 segments)."""
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        task = self._funasr_task(tmp_path, output_format="srt")
        # diarize=True 避免 nospk 重渲染, 直接用引擎 content
        task.options = TranscribeOptions(diarize=True)
        mgr.tasks["t-fr"] = task

        segs = [
            TranscriptionSegment(start_time=0.0, end_time=1.0, text="句一", speaker="Speaker1"),
            TranscriptionSegment(start_time=1.1, end_time=2.0, text="句二", speaker="Speaker1"),
        ]
        srt_dict = {
            "format": "srt",
            "content": (
                "1\n00:00:00,000 --> 00:00:01,000\nSpeaker1:句一\n\n"
                "2\n00:00:01,100 --> 00:00:02,000\nSpeaker1:句二\n"
            ),
            "file_name": "x.wav", "file_hash": "h-fr", "duration": 10.0,
            "processing_time": 0.5,
            "raw_result": [{"sentence_info": []}],
            "segments": segs,
        }
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=srt_dict)

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("t-fr")

        # SRT content 原样 (两句未合并)
        assert "句一" in task.srt_content
        assert "句二" in task.srt_content
        # 缓存存句级
        saved = mock_db.save_result.call_args.args[0]
        assert len(saved.segments) == 2
        # SRT metadata 不带 segment_merge_max_span_sec
        assert "segment_merge_max_span_sec" not in (task.result.metadata or {})

    @pytest.mark.asyncio
    async def test_funasr_fresh_srt_content_no_prefix(self, tmp_path):
        from src.core.task_manager import TaskManager

        mgr = TaskManager()
        task = self._funasr_task(tmp_path, output_format="srt")
        mgr.tasks["t-fr"] = task

        diarized_segments = make_diarized_result("h-fr").segments
        srt_dict = {
            "format": "srt",
            "content": "1\n00:00:00,000 --> 00:00:05,000\nSpeaker1:你好\n",
            "file_name": "x.wav", "file_hash": "h-fr", "duration": 10.0,
            "processing_time": 0.5,
            "raw_result": [{"sentence_info": []}],
            "segments": diarized_segments,
        }
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=srt_dict)

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("t-fr")

        assert "Speaker" not in task.srt_content, "fresh SRT 出口必须投影重渲染无前缀"
        assert "你好" in task.srt_content
        # 缓存仍存 diarized segments
        saved = mock_db.save_result.call_args.args[0]
        assert saved.speakers == ["Speaker1", "Speaker2"]

    @pytest.mark.asyncio
    async def test_qwen3_fresh_nospk_passthrough(self, tmp_path):
        """qwen3 fresh nospk (引擎原生 nospk 输出) → 投影幂等不变"""
        from src.core.task_manager import TaskManager
        from src.models.schemas import TranscriptionTask

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="t-q", file_name="x.wav", file_path=str(tmp_path / "x.wav"),
            file_size=100, file_hash="h-q", engine="qwen3",
            options=TranscribeOptions(diarize=False),
        )
        mgr.tasks["t-q"] = task
        (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

        native_nospk = project_result_nospk(make_diarized_result("h-q"))
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=(native_nospk, {"diarize": False}))

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("t-q")

        assert task.result.speakers == []
        assert mock_db.save_result.call_args.kwargs.get("engine") == "qwen3+nospk"


# ==================== 早返回出口传 options (handler 两处) ====================


class TestEarlyReturnPassesOptions:
    @pytest.mark.asyncio
    async def test_submit_task_passes_options_to_cache_lookup(self, tmp_path):
        from src.core.task_manager import TaskManager
        from src.models.schemas import TranscriptionTask

        tm = TaskManager()
        task = TranscriptionTask(
            task_id="t-el", file_name="a.wav", file_path="", file_size=1,
            file_hash="h-el", engine="qwen3", options=TranscribeOptions(diarize=False),
        )
        tm.tasks["t-el"] = task
        f = tmp_path / "x.wav"
        f.write_bytes(b"\x00" * 10)
        with patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.get_cached_result = AsyncMock(return_value=None)
            await tm.submit_task("t-el", str(f))
            opts = mock_db.get_cached_result.call_args.kwargs.get("options")
            assert opts is not None and opts.diarize is False

    @pytest.mark.asyncio
    async def test_force_refresh_skips_cache_and_projection(self, tmp_path):
        """force_refresh=true: 跳过缓存查询 (含投影回退), 强制重算"""
        from src.core.task_manager import TaskManager
        from src.models.schemas import TranscriptionTask

        tm = TaskManager()
        task = TranscriptionTask(
            task_id="t-ff", file_name="a.wav", file_path="", file_size=1,
            file_hash="h-ff", engine="qwen3", force_refresh=True,
            options=TranscribeOptions(diarize=False),
        )
        tm.tasks["t-ff"] = task
        f = tmp_path / "x.wav"
        f.write_bytes(b"\x00" * 10)
        with patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.get_cached_result = AsyncMock(return_value=None)
            await tm.submit_task("t-ff", str(f))
            assert not mock_db.get_cached_result.called
