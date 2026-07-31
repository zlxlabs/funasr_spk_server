"""
PR1 — FunASR semantic parity test

意图：验证 PR1 改动后，FunASR 路径的语义输出和 PR1 前一致。

设计：
- 默认 skip（FunASR 模型 ~2GB，加载 + 转录耗时数分钟，不适合 CI 默认跑）
- 设 FUNASR_RUN_INTEGRATION=1 后才执行
- 首次运行：当 golden 文件不存在 → 跑转录 + 写入 golden（建立 baseline）
- 后续运行：跑转录 + 与 golden 做 semantic 对比

Semantic 算法（修 codex review T4：byte-equal 不现实）：
- 文本：每段 text 严格相等
- 时间窗：start_time / end_time 容差 ±50ms
- speakers：数量相等（标签允许重映射）
- 忽略：created_at / processing_time / task_id 等运行时字段
"""
import asyncio
import json
import os
from pathlib import Path

import pytest

from src.models.schemas import TranscriptionResult


RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="设置 FUNASR_RUN_INTEGRATION=1 环境变量启用（默认 skip，避免加载 ~2GB FunASR 模型）",
)


# 容差：50ms
TIMESTAMP_TOLERANCE_SECONDS = 0.05


@pytest.fixture(scope="session", autouse=True)
def force_lock_mode():
    """
    强制 FunASR 走 lock 模式（单进程 + threading.Lock）
    原因：pool 模式在 pytest 进程里 fork worker 会很复杂；
    PR1 没改 FunASRTranscriber 类，lock 和 pool 调用 model.generate() 参数完全一致，
    所以 lock 模式生成的 golden 等价于 pool 模式。
    """
    from src.core.config import config
    original = config.transcription.concurrency_mode
    config.transcription.concurrency_mode = "lock"
    yield
    config.transcription.concurrency_mode = original


@pytest.fixture(scope="session", autouse=True)
def force_funasr_engine():
    """强制 default_engine=funasr。

    resolve_transcriber("funasr") 会拒绝 engine != config 默认引擎的请求，而
    dev 环境 .env 可能写死 qwen3。本文件断言的是 FunASR 引擎行为，不该依赖
    环境的引擎配置。
    """
    from src.core.config import config
    original = config.transcription.default_engine
    config.transcription.default_engine = "funasr"
    yield
    config.transcription.default_engine = original


@pytest.fixture(autouse=True)
def reset_transcriber_singleton():
    """每个测试前重置 transcriber 单例，避免状态串扰"""
    import src.core.funasr_transcriber as ft
    # 跑完后保留实例（同进程下重复加载模型代价太大）
    yield
    # 不主动重置 —— 单例复用避免反复加载 2GB 模型


def assert_semantic_equal(actual: TranscriptionResult, golden: dict):
    """语义级别对比 — 容忍非语义差异（时间戳轻微波动、运行时字段）"""
    actual_segs = actual.segments
    golden_segs = golden["segments"]
    assert len(actual_segs) == len(golden_segs), \
        f"segment 数量不一致: actual={len(actual_segs)} golden={len(golden_segs)}"

    for i, (a, g) in enumerate(zip(actual_segs, golden_segs)):
        assert a.text == g["text"], f"segment[{i}] text 不一致: {a.text!r} vs {g['text']!r}"
        assert abs(a.start_time - g["start_time"]) <= TIMESTAMP_TOLERANCE_SECONDS, \
            f"segment[{i}] start_time 差超过 {TIMESTAMP_TOLERANCE_SECONDS}s: {a.start_time} vs {g['start_time']}"
        assert abs(a.end_time - g["end_time"]) <= TIMESTAMP_TOLERANCE_SECONDS, \
            f"segment[{i}] end_time 差超过 {TIMESTAMP_TOLERANCE_SECONDS}s: {a.end_time} vs {g['end_time']}"

    assert len(actual.speakers) == len(golden["speakers"]), \
        f"speaker 数量不一致: actual={len(actual.speakers)} golden={len(golden['speakers'])}"


async def _run_funasr_once(audio_path: Path) -> TranscriptionResult:
    """走 PR1 真实路径（resolve_transcriber → FunASREngine → transcribe）跑一次"""
    from src.core.transcriber_dispatch import resolve_transcriber
    transcriber = resolve_transcriber("funasr")
    if not transcriber.is_initialized:
        await transcriber.initialize()
    result = await transcriber.transcribe(
        audio_path=str(audio_path),
        task_id="parity-test",
        progress_callback=None,
        output_format="json",
    )
    # JSON 模式返回 (TranscriptionResult, raw_result)
    return result[0]


def _golden_path(golden_dir: Path, audio_name: str) -> Path:
    return golden_dir / f"{audio_name}.golden.json"


@pytest.mark.integration
class TestParityFunasrSemantic:
    @pytest.mark.asyncio
    async def test_tts_1speaker_5s(self, tts_audio: Path, golden_dir: Path):
        await self._parity_check(tts_audio, golden_dir)

    @pytest.mark.asyncio
    async def test_silence_5s(self, silence_audio: Path, golden_dir: Path):
        await self._parity_check(silence_audio, golden_dir)

    @pytest.mark.asyncio
    async def test_podcast_2speakers_60s(self, podcast_audio: Path, golden_dir: Path):
        await self._parity_check(podcast_audio, golden_dir)

    async def _parity_check(self, audio_path: Path, golden_dir: Path):
        golden_file = _golden_path(golden_dir, audio_path.stem)
        actual = await _run_funasr_once(audio_path)

        if not golden_file.exists():
            # 首次运行：把当前输出当作 baseline 保存
            golden_dir.mkdir(parents=True, exist_ok=True)
            golden_data = {
                "segments": [
                    {"start_time": s.start_time, "end_time": s.end_time, "text": s.text, "speaker": s.speaker}
                    for s in actual.segments
                ],
                "speakers": actual.speakers,
            }
            golden_file.write_text(json.dumps(golden_data, ensure_ascii=False, indent=2))
            pytest.skip(f"首次运行：已写入 golden baseline {golden_file}，复跑即生效")
        else:
            golden = json.loads(golden_file.read_text())
            assert_semantic_equal(actual, golden)
