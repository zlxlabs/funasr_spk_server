"""
Qwen3PoolTranscriber — 主进程 wrapper, 通过 FileBasedProcessPool 调度 worker subprocess

设计:
- wrapper 接口与 Qwen3DiarizeTranscriber.transcribe 鸭子兼容
  * JSON: (TranscriptionResult, raw_result)
  * SRT:  {format, content, file_name, file_hash, duration, processing_time, raw_result}
- 持有 FileBasedProcessPool, worker_entry_script="src/core/qwen3_worker_process.py"
- transcribe(audio_path, task_id, progress_callback, output_format) →
    pool.generate_with_pool(extra_task_fields={"output_format": output_format})

测试覆盖:
1. constructor 默认创建 pool, pool_size 从 config 读, entry 正确
2. constructor 可注入自定义 pool (依赖注入便于测试)
3. JSON 模式 transcribe → 返回 (TranscriptionResult, raw)
4. SRT 模式 transcribe → 返回 SRT dict
5. progress_callback (sync + async) 在 0% / 100% 被调
6. pool 抛错 → wrapper 透传
7. transcribe 把 output_format 通过 extra_task_fields 传给 pool
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.schemas import TranscriptionResult, TranscriptionSegment


# ==================== 辅助 ====================


def _fake_json_result(task_id: str = "t", text: str = "你好世界"):
    """构造 JSON 模式的 worker 返回 (TranscriptionResult, raw)"""
    tres = TranscriptionResult(
        task_id=task_id,
        file_name="a.wav",
        file_hash="h",
        duration=1.0,
        segments=[
            TranscriptionSegment(start_time=0.0, end_time=1.0, text=text, speaker="Speaker1")
        ],
        speakers=["Speaker1"],
        processing_time=0.1,
    )
    return (tres, {"asr_text": text, "engine": "qwen3"})


def _fake_srt_result(text: str = "字幕内容"):
    return {
        "format": "srt",
        "content": f"1\n00:00:00,000 --> 00:00:01,000\nSpeaker1:{text}\n",
        "file_name": "a.wav",
        "file_hash": "h",
        "duration": 1.0,
        "processing_time": 0.1,
        "raw_result": {"asr_text": text, "engine": "qwen3"},
    }


# ==================== Constructor ====================


class TestConstructor:
    """wrapper 持有 pool, entry 正确"""

    def test_default_pool_has_qwen3_entry(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        wrapper = Qwen3PoolTranscriber(pool_size=3)
        assert wrapper._pool.worker_entry_script == "src/core/qwen3_worker_process.py"
        assert wrapper._pool.pool_size == 3

    def test_default_pool_uses_isolated_task_dir(self):
        """task_dir 必须与 FunASR 池物理隔离, 否则同机器 FunASR daemon 抢任务文件"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        wrapper = Qwen3PoolTranscriber(pool_size=2)
        # 必须不是 FunASR 默认 ./temp/tasks
        assert str(wrapper._pool.task_dir) != "temp/tasks"
        assert str(wrapper._pool.task_dir) != "./temp/tasks"
        # 推荐路径: temp/tasks_qwen3
        assert "qwen3" in str(wrapper._pool.task_dir).lower()

    def test_inject_custom_pool(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock(name="custom_pool")
        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)
        assert wrapper._pool is custom_pool


# ==================== transcribe 模式透传 ====================


class TestTranscribeJsonMode:
    """JSON 模式直接返回 (TranscriptionResult, raw)"""

    @pytest.mark.asyncio
    async def test_returns_tuple_for_json_mode(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        expected = _fake_json_result(task_id="t-1")
        custom_pool.generate_with_pool = AsyncMock(return_value=expected)

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        result = await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t-1", output_format="json"
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        tres, raw = result
        assert tres.task_id == "t-1"
        assert raw["engine"] == "qwen3"

    @pytest.mark.asyncio
    async def test_extra_task_fields_includes_output_format_json(self):
        """wrapper 必须把 output_format 透传给 pool 的 extra_task_fields"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t", output_format="json")

        # 检查 pool 收到了 output_format=json
        call_kwargs = custom_pool.generate_with_pool.call_args.kwargs
        assert call_kwargs.get("extra_task_fields") == {
            "output_format": "json",
            "options": {"language": None, "diarize": True, "word_align": False},
        }


class TestTranscribeSrtMode:
    """SRT 模式返回 dict"""

    @pytest.mark.asyncio
    async def test_returns_dict_for_srt_mode(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_srt_result(text="对白"))

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)
        result = await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t-srt", output_format="srt"
        )

        assert isinstance(result, dict)
        assert result["format"] == "srt"
        assert "对白" in result["content"]

    @pytest.mark.asyncio
    async def test_extra_task_fields_includes_output_format_srt(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_srt_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)
        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t", output_format="srt")

        call_kwargs = custom_pool.generate_with_pool.call_args.kwargs
        assert call_kwargs.get("extra_task_fields") == {
            "output_format": "srt",
            "options": {"language": None, "diarize": True, "word_align": False},
        }


# ==================== progress_callback ====================


class TestProgressCallback:
    """跨进程进度难传, wrapper 模拟 0% / 100% 两次回调"""

    @pytest.mark.asyncio
    async def test_async_callback_called(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        calls = []
        async def progress(pct):
            calls.append(pct)

        await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t", progress_callback=progress
        )

        assert 0 in calls
        assert 100 in calls

    @pytest.mark.asyncio
    async def test_sync_callback_called(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        calls = []
        def progress(pct):
            calls.append(pct)

        await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t", progress_callback=progress
        )

        assert 0 in calls
        assert 100 in calls

    @pytest.mark.asyncio
    async def test_none_callback_does_not_crash(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)
        # 不传 progress_callback (None) 不应崩
        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t")


# ==================== heartbeat progress (避免 client recv timeout 30 min) ====================


class TestHeartbeatProgress:
    """长任务期间周期性发 progress_callback, 避免 client recv timeout.

    背景: 149min mp3 转录耗时 34 min, 但 client 默认 30 min recv timeout.
    wrapper 必须在 pool.generate_with_pool 等待期间周期性触发 progress_callback,
    使 task_manager → ws_handler 发 task_progress 给 client 重置 timer.
    """

    @pytest.mark.asyncio
    async def test_heartbeat_calls_callback_multiple_times_during_long_task(self):
        """模拟 1 秒任务, 设 heartbeat 0.05s → callback 应至少被调 5+ 次 (除 0 / 100)"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()

        async def slow_pool_op(*args, **kwargs):
            await asyncio.sleep(1.0)
            return _fake_json_result()

        custom_pool.generate_with_pool = AsyncMock(side_effect=slow_pool_op)

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool, heartbeat_interval_seconds=0.05)

        calls = []
        async def progress(pct):
            calls.append(pct)

        await wrapper.transcribe(
            audio_path="/fake/a.wav", task_id="t-hb", progress_callback=progress
        )

        # 至少 0%, 多次心跳, 100% (>=5 次)
        assert len(calls) >= 5, f"心跳次数应 >=5, 实际 {len(calls)}: {calls}"
        assert 0 in calls, "应有 0% 起始"
        assert 100 in calls, "应有 100% 完成"
        # 心跳值在 (0, 100) 之间至少 3 次
        heartbeat_values = [c for c in calls if 0 < c < 100]
        assert len(heartbeat_values) >= 3, f"心跳中间值应 >=3 次, 实际 {heartbeat_values}"

    @pytest.mark.asyncio
    async def test_heartbeat_progress_monotonically_increases_then_caps_at_95(self):
        """心跳值应单调递增, 封顶 95 (避免比 100% 早)"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()

        async def slow(*args, **kwargs):
            await asyncio.sleep(0.6)
            return _fake_json_result()

        custom_pool.generate_with_pool = AsyncMock(side_effect=slow)

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool, heartbeat_interval_seconds=0.05)

        calls = []
        async def progress(pct):
            calls.append(pct)

        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t", progress_callback=progress)

        # 0, [heartbeats monotonically increase capped at 95], 100
        heartbeat_values = [c for c in calls if 0 < c < 100]
        assert all(v <= 95 for v in heartbeat_values), f"心跳值应封顶 95, got: {heartbeat_values}"
        assert heartbeat_values == sorted(heartbeat_values), f"心跳值应单调递增: {heartbeat_values}"

    @pytest.mark.asyncio
    async def test_heartbeat_stops_after_pool_completes(self):
        """pool 完成后 heartbeat 必须立即 cancel, 不再产生额外 callback"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool, heartbeat_interval_seconds=0.05)

        calls = []
        async def progress(pct):
            calls.append(pct)

        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t", progress_callback=progress)
        snapshot = len(calls)
        # transcribe 完成后等一会儿, callback 不应再被调
        await asyncio.sleep(0.3)
        assert len(calls) == snapshot, f"transcribe 完成后 heartbeat 仍在跑: 前 {snapshot}, 后 {len(calls)}"

    @pytest.mark.asyncio
    async def test_heartbeat_stops_on_pool_error(self):
        """pool 抛错时 heartbeat 也必须 cancel"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()

        async def fail_slow(*args, **kwargs):
            await asyncio.sleep(0.2)
            raise RuntimeError("pool 炸了")

        custom_pool.generate_with_pool = AsyncMock(side_effect=fail_slow)

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool, heartbeat_interval_seconds=0.05)

        calls = []
        async def progress(pct):
            calls.append(pct)

        with pytest.raises(RuntimeError, match="pool 炸了"):
            await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t", progress_callback=progress)

        snapshot = len(calls)
        await asyncio.sleep(0.3)
        # 错误也应停 heartbeat
        assert len(calls) == snapshot, "pool 抛错后 heartbeat 仍在跑"

    @pytest.mark.asyncio
    async def test_heartbeat_default_interval_is_30_seconds(self):
        """默认心跳 30s (覆盖 client 30 min recv timeout, 客户端最长 30 min 无消息)"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber
        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=MagicMock())
        assert wrapper.heartbeat_interval_seconds == 30.0

    @pytest.mark.asyncio
    async def test_no_callback_means_no_heartbeat_overhead(self):
        """progress_callback=None 时不应启 heartbeat task (节约 overhead)"""
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(return_value=_fake_json_result())

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool, heartbeat_interval_seconds=0.01)
        # 不传 progress_callback 不应崩 + 不应启 heartbeat
        await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t")


# ==================== error propagation ====================


class TestErrorPropagation:
    """pool 抛错 → wrapper 透传"""

    @pytest.mark.asyncio
    async def test_pool_error_propagates(self):
        from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

        custom_pool = MagicMock()
        custom_pool.generate_with_pool = AsyncMock(side_effect=RuntimeError("pool 炸了"))

        wrapper = Qwen3PoolTranscriber(pool_size=2, pool=custom_pool)

        with pytest.raises(RuntimeError, match="pool 炸了"):
            await wrapper.transcribe(audio_path="/fake/a.wav", task_id="t")


# ==================== pool extra_task_fields 参数支持 ====================


class TestPoolExtraTaskFields:
    """FileBasedProcessPool.generate_with_pool 必须接受 extra_task_fields 关键字参数"""

    @pytest.mark.asyncio
    async def test_extra_task_fields_merged_into_task_json(self, tmp_path, monkeypatch):
        """extra_task_fields 的字段应该合并到任务 JSON 文件中"""
        import json
        from src.core.file_based_process_pool import FileBasedProcessPool

        monkeypatch.chdir(tmp_path)

        pool = FileBasedProcessPool(pool_size=1)
        pool.is_initialized = True  # 跳过初始化

        # mock _ensure_workers_alive / _spawn_worker / _calculate_timeout
        pool._ensure_workers_alive = AsyncMock()
        pool._spawn_worker = AsyncMock()
        pool._calculate_timeout = MagicMock(return_value=600)

        # mock worker 不存在(_launch 不会真跑)
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        pool.worker_processes = [fake_proc]
        fake_proc.pid = 4321

        # 准备 fake audio
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"\x00")

        # 在 pool 写完 task_file 之后立刻塞一个 fake 结果文件触发返回
        import asyncio as _asyncio
        import pickle as _pickle

        async def watcher():
            # 等待 task 文件出现, 然后回写 pickle 结果
            for _ in range(50):
                task_files = list(pool.task_dir.glob("worker_0_*.task"))
                if task_files:
                    task_file = task_files[0]
                    task_data = json.loads(task_file.read_text())
                    # 暴露 task_data 给主测试线
                    captured["task_data"] = task_data

                    # 写 pickle 结果
                    result_file = task_file.with_suffix(".pkl")
                    with open(result_file, "wb") as f:
                        _pickle.dump(
                            {"task_id": task_data["task_id"], "success": True, "result": "OK", "worker_pid": fake_proc.pid},
                            f,
                        )
                    return
                await _asyncio.sleep(0.05)

        captured = {}
        watcher_task = _asyncio.create_task(watcher())

        try:
            result = await pool.generate_with_pool(
                audio_path=str(audio),
                extra_task_fields={"output_format": "json", "custom_field": 42},
            )
        finally:
            await watcher_task

        assert result == "OK"
        # 检查 task_data 包含 extra 字段
        td = captured["task_data"]
        assert td["output_format"] == "json"
        assert td["custom_field"] == 42
        # 同时也保留 audio_path / task_id (基础字段)
        assert "task_id" in td
        assert "audio_path" in td
