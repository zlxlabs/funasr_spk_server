# Qwen3 speaker over-detect 修复 — 新 session prompt

> 任务类型: **严格 TDD 修复** — 不再做调研, 调研结论已锁定
> 预估 session: 1-2 次, 6-8 个 commit
> 工作模式: **中途非必要不要停下来询问, 跑通直到 DoD 全满足**
> 工作目录: `~/Dev/projects/250729_funasr_spk_server/funasr_spk_server`
> 起点分支: `spike/qwen3-diarize-poc` HEAD `92e8442` (3 commit ahead origin 已 push)
> 修复 PR 切新分支: `fix/qwen3-spk-overdetect`

---

## 1. 上下文 (必读, 否则不知道在改什么)

### 1.1 调研结论 (锁死, 不再争议)

**根因 commit**: `cd578a8 fix(qwen3): worker 加 ffmpeg audio format 转换` (2026-05-15 22:42).

**机制**: 该 commit 在 `src/core/qwen3_worker_process.process_task` 加了 `convert_to_wav` (ffmpeg → 16kHz mono wav) 预处理. 修复了真实的 m4a 不能被 sherpa-onnx libsndfile 直接读取的 production bug, 但**代价是改变了喂给 sherpa diarize 的音频字节**:

- mp3 直读 (`librosa.load`) vs ffmpeg→wav 后读 (`sf.read`), 样本数一致 (57.6M / 3600s) 但 rms_diff 0.4%, 仅 4.8% 样本近相等
- 这点差异通过 pyannote + NeMo TitaNet embedding 放大, 把 60min-2spk audio 的 FastClustering@0.9 输出从 7 cluster (主 2 + 5 个 <3s 噪声) 改为 11 cluster (主 2 + 2 个 43.2s/61.6s 中长噪声 + 7 个 <2s 噪声)
- 那两个 43.2s / 61.6s cluster **恰好突破 `filter_spurious_speakers` 的 `max(2s, 1% × 3600s) = 36s` 阈值**, 漏网 → final 4 speakers

**完整调研报告**: `docs/开发/archive/spk-over-detect-归因调研结果.md`

### 1.2 现有 Red test (锁定 bug 形态)

`tests/unit/test_qwen3_spk_overdetect_repro.py`:
- `test_filter_spurious_60min_2spk_ffmpeg_wav_over_detect` — xfail (复现 over-detect 后处理路径)
- `test_filter_spurious_60min_2spk_clean_path_passes` — passing (验证 mp3 直读路径 filter 正常)

修复完成后, 这两个 test 都应该改成 regression guard (取消 xfail). 不要删, 留作活文档.

### 1.3 选定的修复方案 (其它方向不要追)

**方向 2 + 方向 4 组合** (报告 §4 推荐方案):

- **方向 2** — `qwen3_worker_process` 对 sherpa-supported 格式跳过 ffmpeg 转换, 让 sherpa 直接读 mp3/flac/ogg (走 librosa fallback, 跟 PR3 PoC 路径一致)
- **方向 4** — 扩展 `apply_cluster_centroid_merge` 的 dominant 模式: 当 dominant cluster 占比 ≥ `dominant_share` (默认 0.6) 时, **所有 minor cluster (share < `cluster_merge_min_main_share`)** 也跟 dominant 做 cosine 比较, 接近就吃掉

**为什么两个都做**:
- 方向 2 是治本 — 让 sherpa 拿到的 audio 跟 PR3 PoC 干净路径一致, 期望直接收敛到 2 spk
- 方向 4 是兜底 — 即使将来又有 audio 触发类似 over-detect (不同 m4a 解码器 / 不同 audio profile), cluster_merge 多人模式能扩到 minor cluster 拦截

---

## 2. DoD (Definition of Done)

按顺序全部满足才算完成. 任一条不满足, 不能宣布修复完成.

1. ✅ 现有 red test `test_filter_spurious_60min_2spk_ffmpeg_wav_over_detect` 取消 `@pytest.mark.xfail` 标记, 仍然能通过 (因为生产路径下 ffmpeg 不再被触发, 这种 turns 形态不会再出现; 留作历史 regression guard)
2. ✅ 新 integration test (放 `tests/integration/test_qwen3_spk_overdetect_fix.py`): 60min-2spk audio 走完整 worker pipeline (mp3 输入) N=1 单跑, assert `speakers_count == 2`. 修复前 fail, 修复后 pass.
3. ✅ Eval set 4 个 audio 全过回归:
   - `audio_1spk_real.m4a` → 1 speaker
   - `audio_2spk_60min.mp3` → **2** speakers
   - `audio_4spk.m4a` → 5 speakers (含 Speaker66 英文歌, 跟 PR3 PoC 一致, 不要去管)
   - `audio_6spk_60min.m4a` → 6 speakers
4. ✅ 全套 unit test 绿: `venv/bin/python -m pytest tests/unit/ -v`
5. ✅ 全套 integration test 绿: `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/`
6. ✅ 至少 6 个 commit, 红绿分明, 每个 commit message 写清楚改了什么 + 解决什么问题
7. ✅ `docs/开发/archive/spk-over-detect-归因调研结果.md` 补 "修复" 章节 (写哪个 commit 修了, eval set 修复后数据)
8. ✅ 把 `docs/开发/Qwen3-spk-over-detect-修复-新session-prompt.md` (本文件) 归档到 `docs/开发/archive/`

---

## 3. TDD 步骤化 (严格按这个顺序, 每一步 commit 一次)

### 步骤 0: 切分支 + 跑现有 baseline (不改代码, 0 commit)

```bash
cd ~/Dev/projects/250729_funasr_spk_server/funasr_spk_server
git checkout -b fix/qwen3-spk-overdetect spike/qwen3-diarize-poc
venv/bin/python -m pytest tests/unit/test_qwen3_spk_overdetect_repro.py -v
# 期望: 1 xfailed (test_filter...over_detect) + 1 passed (test_filter...clean_path_passes)

venv/bin/python -m pytest tests/unit/ -v 2>&1 | tail -5
# 期望: 全绿 (或仅 spk_overdetect_repro 那 1 个 xfail)
```

### 步骤 1: 加 integration red test (commit 1) 🔴

新建 `tests/integration/test_qwen3_spk_overdetect_fix.py`:

```python
"""Integration test: 60min-2spk audio 走完整 worker pipeline 不应 over-detect."""
import os
import pytest
import asyncio
import json
from pathlib import Path

if not os.environ.get("FUNASR_RUN_INTEGRATION"):
    pytest.skip("integration test, set FUNASR_RUN_INTEGRATION=1 to run", allow_module_level=True)


AUDIO = "tmp_long_audio/eval_set/audio_2spk_60min.mp3"


@pytest.mark.asyncio
async def test_2spk_60min_no_over_detect():
    """60min mp3 (2 真实 speaker) 走 worker pipeline, 期望最终 segments speakers == 2."""
    if not Path(AUDIO).exists():
        pytest.skip(f"audio not found: {AUDIO}")

    from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

    pool = Qwen3PoolTranscriber(pool_size=1)
    await pool.initialize()
    try:
        result, raw = await pool.transcribe(
            audio_path=AUDIO,
            task_id="overdetect-fix-test",
            progress_callback=None,
            output_format="json",
        )
    finally:
        # pool 清理由 GC + atexit
        pass

    assert len(result.speakers) == 2, (
        f"60min-2spk audio 应有 2 speaker, 实测 {len(result.speakers)}: "
        f"{result.speakers}"
    )
```

跑一次确认它 fail (修复前期望 fail):
```bash
FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/test_qwen3_spk_overdetect_fix.py -v -s
# 期望: 1 failed (speakers_count=4)
```

**Commit 1**: `test(qwen3): 加 60min-2spk over-detect integration red test`

### 步骤 2: 方向 2 绿 — worker 对 sherpa-supported 格式跳过 ffmpeg (commit 2-3) 🟢

#### Commit 2: 加 unit test 锁定 worker 路径选择契约

新建 `tests/unit/test_qwen3_worker_skip_ffmpeg_for_sherpa.py`:
- mock `convert_to_wav`, 验证调用次数
- 测试 5 个 case (sherpa-supported 跳过, m4a 仍转换):
  - `.wav` → 不调 convert_to_wav (现有契约保留)
  - `.mp3` → 不调 convert_to_wav (新契约, 当前 fail)
  - `.flac` → 不调 convert_to_wav (新契约)
  - `.ogg` → 不调 convert_to_wav (新契约)
  - `.m4a` → 调 convert_to_wav (现有契约保留)
  - `.aac` → 调 convert_to_wav (现有契约保留)

参考 `tests/unit/test_qwen3_worker_process.py` 现有结构 (commit cd578a8 已经写过 3 个 case, 扩展即可).

跑一次确认它 fail:
```bash
venv/bin/python -m pytest tests/unit/test_qwen3_worker_skip_ffmpeg_for_sherpa.py -v
# 期望: 至少 mp3/flac/ogg 3 个 case fail
```

**Commit 2**: `test(qwen3): worker 对 sherpa-supported 格式 (mp3/flac/ogg) 跳过 ffmpeg red test`

#### Commit 3: 改 worker_process 跳过 sherpa-supported 格式

改 `src/core/qwen3_worker_process.py:81-93`:

```python
# 当前 (cd578a8 引入):
if not audio_path.lower().endswith(".wav"):
    # ffmpeg convert
    ...

# 改为:
SHERPA_SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".opus"}  # libsndfile + sherpa 直读
audio_ext = os.path.splitext(audio_path)[1].lower()
if audio_ext not in SHERPA_SUPPORTED_EXTENSIONS:
    # 仅对 m4a/aac/mp4/mov/webm 等非 libsndfile 格式才 ffmpeg 转换
    ...
```

注意:
- `_load_audio_mono_16k` (`src/core/qwen3/diarize.py`) 已经有 `sf.read → librosa.load` fallback, 所以 mp3 直传 sherpa pipeline 不会 crash; sherpa 内部 OfflineSpeakerDiarization 接受 ndarray 输入
- 但 ASR vendor (`src/core/qwen3/asr.py` → `run_asr`) 也要确认能读 mp3. 如果不能, 这一步可能要分 ASR/diarize 两条路 (diarize 走 ndarray, ASR 单独 ffmpeg 转 wav 给 vendor)
- 如果 ASR vendor 不接受 mp3, 退化方案: 只 diarize 跳 ffmpeg, ASR 仍 ffmpeg 转 wav (用同一 audio_path 不能两条路, 需要修改 transcribe 接口接受 audio ndarray)
- **如果遇到 ASR vendor 障碍, 不要硬刚 vendor 代码** — 改成最小侵入: worker 在调 transcribe 前**双重转**(转 wav 给 ASR vendor 用, 原 mp3 path 同时也透传, 让 diarize 用原 mp3). 或者: 实验确认 PoC 时期 mp3 给 ASR vendor 也能跑 (历史已成功跑过 audio_149min.mp3 经 PoC 脚本, 说明 vendor 能读 mp3)

跑:
```bash
venv/bin/python -m pytest tests/unit/test_qwen3_worker_skip_ffmpeg_for_sherpa.py -v
# 期望: 全绿

FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/test_qwen3_spk_overdetect_fix.py -v -s
# 期望: 1 passed (speakers_count==2 修复了!)
```

**Commit 3**: `fix(qwen3): worker 对 sherpa-supported 格式 (mp3/flac/ogg) 跳过 ffmpeg, 修 60min-2spk over-detect`

> ⚠️ **如果 commit 3 已经让 integration test 绿了**, 方向 4 仍然要做, 因为它是兜底防御层. 不要因为"已修好"就跳过.

### 步骤 3: 方向 4 — cluster_merge dominant 扩展到 minor (commit 4-5) 🔴🟢

#### Commit 4: red test for minor-fold-to-dominant

新建 `tests/unit/test_cluster_merge_dominant_minor_fold.py`:
- mock `extractor_fn` 返回 deterministic embedding (5 个 fixed unit vectors)
- 构造场景:
  - speaker "A": 600s (60% main, 即 dominant)
  - speaker "B": 300s (30% main)
  - speaker "C": 50s (5% main, 短但 ≥ min_main_share=0.03)
  - speaker "D": 20s (2% minor, **<** min_main_share, embedding 跟 A 接近, cos ≥ 0.6)
  - speaker "E": 10s (1% minor, embedding 跟 A 远, cos < 0.5)
- audio_duration = 1000s
- 当前算法行为 (期望 fail): D 仍以独立 speaker 输出
- 修复后行为: D 被合到 A (因为 dominant=A ≥ 60% 且 minor D 跟 dominant cos ≥ 阈值), E 保留

跑:
```bash
venv/bin/python -m pytest tests/unit/test_cluster_merge_dominant_minor_fold.py -v
# 期望: fail (当前算法不会把 minor 合到 dominant)
```

**Commit 4**: `test(qwen3): cluster_merge dominant 模式吃 minor cluster red test`

#### Commit 5: 实现 minor-fold-to-dominant + 加 config 字段

改 `src/core/qwen3/cluster_merge.py` 的 `apply_cluster_centroid_merge`:
- 在 dominant 模式判断 (`if dominant_share >= ...`) 之后, 加一段:
  - 收集所有 minor speaker (share < min_main_share)
  - 对每个 minor, 算它的 centroid 跟 dominant centroid 的 cosine
  - 如果 cos ≥ 新阈值 `dominant_minor_threshold` (默认 0.5, 比 main_threshold=0.78 宽松), 合并到 dominant
  - log 加 `action=minor_folded_into_dominant`

改 `src/core/config.py` `Qwen3Config`:
- 加字段 `cluster_merge_dominant_minor_threshold: float = 0.5`
- 加 env override `FUNASR_QWEN3_CLUSTER_MERGE_DOMINANT_MINOR_THRESHOLD`

改 `src/core/qwen3_transcriber.py`:
- `__init__` 加参数 `cluster_merge_dominant_minor_threshold: float = 0.5`
- `build_engine_config` / `get_qwen3_transcriber` singleton 透传新字段

改测试 (`tests/unit/test_config_qwen3_cluster_merge.py` + `tests/unit/test_qwen3_cluster_merge.py`):
- 加默认值断言 + env override 断言

跑:
```bash
venv/bin/python -m pytest tests/unit/test_cluster_merge_dominant_minor_fold.py -v
# 期望: 全绿

venv/bin/python -m pytest tests/unit/ -v 2>&1 | tail -10
# 期望: 全绿
```

**Commit 5**: `feat(qwen3): cluster_merge dominant 模式吃相似 minor cluster (新 dominant_minor_threshold=0.5)`

### 步骤 4: 取消 red test xfail + eval set 回归 (commit 6) ✅

#### 取消 xfail

改 `tests/unit/test_qwen3_spk_overdetect_repro.py`:
- 删 `@pytest.mark.xfail(...)` 装饰器 (留 docstring 说明 historical regression)
- 该 test 原本 assert 后处理路径 filter_spurious 在 11-cluster 输入下应剩 2 spk; 修复后产生该 11 cluster 的路径 (ffmpeg) 不再被触发, 但**该 test 是直接给硬编码 turns 喂 filter_spurious, 跟 audio path 无关**
- 所以 xfail 取消后该 test 仍会 fail, 因为 filter_spurious 阈值没改
- **正确做法**: 改 test 描述为 "如果未来又出现这种 turns 形态, cluster_merge 多人模式 (方向 4) 兜底应该能合掉" — 跑完整 pipeline (filter_spurious + apply_cluster_centroid_merge_to_turns), 注入 mock extractor_fn 让 minor cluster 跟 main cosine 高, assert 最终 2 spk

如果该 test 改造太复杂, **直接删掉 xfail test 也可以**, 因为方向 2 已经从根上解决, 没必要保留 "如果回归到 ffmpeg + filter 阈值都没修, 该怎么样" 的反事实测试. 留 `test_filter_spurious_60min_2spk_clean_path_passes` 作 sanity.

#### Eval set 回归

跑全 eval set (N=1 单跑, 不用 N=2 复杂化):

```bash
# 用 bench_n1_concurrency 或仿造 bench_n2 写个 N=1 单跑脚本
cat > /tmp/eval_set_n1_verify.py <<'PY'
"""N=1 单跑 eval_set 4 个 audio, 验证 spk 数."""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, '~/Dev/projects/250729_funasr_spk_server/funasr_spk_server')
import os; os.chdir('~/Dev/projects/250729_funasr_spk_server/funasr_spk_server')

from src.core.qwen3_pool_transcriber import Qwen3PoolTranscriber

EVAL_SET = {
    "1spk_real": ("tmp_long_audio/eval_set/audio_1spk_real.m4a", 1),
    "2spk_60min": ("tmp_long_audio/eval_set/audio_2spk_60min.mp3", 2),
    "4spk_44min": ("tmp_long_audio/eval_set/audio_4spk.m4a", 5),   # 含 Speaker66 英文歌
    "6spk_60min": ("tmp_long_audio/eval_set/audio_6spk_60min.m4a", 6),
}

async def main():
    pool = Qwen3PoolTranscriber(pool_size=1)
    await pool.initialize()
    for name, (audio, exp) in EVAL_SET.items():
        if not Path(audio).exists():
            print(f"{name}: SKIP (audio missing)"); continue
        r, _ = await pool.transcribe(audio_path=audio, task_id=f"verify-{name}",
                                       progress_callback=None, output_format="json")
        ok = "✓" if len(r.speakers) == exp else "✗"
        print(f"{name:12s} expected={exp} actual={len(r.speakers)} {r.speakers} {ok}")

asyncio.run(main())
PY
venv/bin/python /tmp/eval_set_n1_verify.py
# 期望:
#   1spk_real    expected=1 actual=1 ✓
#   2spk_60min   expected=2 actual=2 ✓
#   4spk_44min   expected=5 actual=5 ✓
#   6spk_60min   expected=6 actual=6 ✓
```

如果某个不通过, **不能 commit**. 调参或重新 design, 直到全过.

**Commit 6**: `test(qwen3): 取消 over-detect xfail + eval_set N=1 全量回归通过`

### 步骤 5: 文档 + 归档 (commit 7) ✅

改 `docs/开发/archive/spk-over-detect-归因调研结果.md`:
- 加 "修复" 章节, 写 commit 列表 + eval_set 修复后数据
- 把 §7 "还没做的" 列表对应项打钩

移动 `docs/开发/Qwen3-spk-over-detect-修复-新session-prompt.md` (本文件) → `docs/开发/archive/`.

更新 `CLAUDE.md` 的 "ASR 引擎架构" → "Qwen3 后处理 pipeline" 章节, 加 cluster_merge 多人 minor-fold 段说明.

**Commit 7**: `docs(qwen3): spk over-detect 修复落档 + 归档 prompt`

---

## 4. 关键文件清单

### 调研产物 (起点)
- `docs/开发/archive/spk-over-detect-归因调研结果.md` — 调研报告 (必读)
- `tests/unit/test_qwen3_spk_overdetect_repro.py` — Red test
- `tmp_long_audio/eval_set/README.md` — Eval set metadata

### 要改的代码
- `src/core/qwen3_worker_process.py` — 方向 2 主要修改点 (78-93 行 ffmpeg 判断)
- `src/core/qwen3/cluster_merge.py` — 方向 4 主要修改点 (`apply_cluster_centroid_merge` 函数, dominant 模式段)
- `src/core/config.py` — 加新 config 字段 (`Qwen3Config` 区块 + `_apply_env_overrides`)
- `src/core/qwen3_transcriber.py` — `__init__` 接受新参数 + singleton 透传

### 要加 / 改的测试
- `tests/integration/test_qwen3_spk_overdetect_fix.py` (新) — 60min-2spk e2e
- `tests/unit/test_qwen3_worker_skip_ffmpeg_for_sherpa.py` (新) — worker 路径选择
- `tests/unit/test_cluster_merge_dominant_minor_fold.py` (新) — minor fold red+绿
- `tests/unit/test_qwen3_spk_overdetect_repro.py` (改) — 取消 xfail / 删除
- `tests/unit/test_config_qwen3_cluster_merge.py` (改) — 加新字段断言
- `tests/unit/test_qwen3_cluster_merge.py` (改) — 加 minor fold unit case

### 不要碰
- `src/core/qwen3/diarize.py` — `_load_audio_mono_16k` 已有 librosa fallback, 不动
- `src/core/qwen3/merge.py` — `filter_spurious_speakers` 阈值不动 (方向 1 不采纳)
- `src/core/qwen3/asr.py` — 除非 vendor 必须 ffmpeg, 否则不动
- `src/core/vendor/` — vendor 代码绝对不改, 调研已确认是上层问题

---

## 5. 环境 / 命令

```bash
# venv
venv/bin/python --version  # Python 3.12.13

# 跑 unit
venv/bin/python -m pytest tests/unit/ -v

# 跑 integration (需要模型 + audio 在位)
FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/

# 跑特定 test
venv/bin/python -m pytest tests/unit/test_qwen3_spk_overdetect_repro.py -v

# ffmpeg / sherpa / Qwen3 vendor library 已就绪 (调研阶段验证过)
```

环境变量 (按需 export):
```bash
export FUNASR_RUN_INTEGRATION=1
unset TMPDIR; export TMPDIR=/tmp   # 防止 ctx-mode 污染 TMPDIR (memory: feedback_pm2_clean_env)
export DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin"
```

---

## 6. 工作约束 / 不变量

### 必须遵守
- ✅ **严格 TDD**: 每个改动**先写红 test, 跑确认它 fail, 再写代码让它绿, 再 commit** (memory `feedback-tdd-strict`). 不要先写代码后补测试.
- ✅ **最小 commit**: 红 → 绿 → commit 是最小单位. 不要积累多个改动一次性提交.
- ✅ **commit message 中英混合可以, 但必须写清楚改了什么 + 解决什么问题**.
- ✅ **不动 src/ vendor**, vendor 代码绝对不改.
- ✅ **不动 ASR 路径** (`src/core/qwen3/asr.py`, `src/core/funasr_transcriber.py`), 只动 diarize / worker / config.
- ✅ **每个 commit 后立刻跑相关 test 确认绿**, 失败就 fix-up 同一 commit, 不要再 commit 一次.

### 风险点 / 决策点 (自己判断, 不要问)
- 如果 commit 3 让 integration 直接绿, 仍然继续 commit 4-5 做方向 4 (兜底).
- 如果 commit 3 让 integration 没绿 (仍 over-detect), 走方向 4 继续, **同时**回头 debug 为什么方向 2 没起效 (可能 ASR vendor 也 ffmpeg 内部转码改变了 audio, 或某个隐藏路径). 不要因为方向 2 不通就放弃整个方案.
- 如果 ASR vendor 不接受 mp3 → 方案 A: 透传 audio bytes 给 vendor 强转; 方案 B: ASR 用 ffmpeg, diarize 用 librosa, 两条独立路径. 优先 B (改动小).
- eval set 全过是硬指标, 任何 audio 退化 (1spk → 0/2, 4spk → 4/6, 6spk → 5/7) 都不算修好. 调参或加 minor_threshold env override 给特定 audio 用.

### 不变量 (任何 commit 后都必须满足)
- 跨引擎缓存 (`cache_cross_engine`) 行为不变
- FunASR 引擎完全不受影响 (其 worker 路径不动)
- Qwen3 现有 RTF (0.20-0.22) 不显著退化 (±5% 内可接受)

---

## 7. 完成清单 (修复人自查)

修复结束前对照打勾:

- [ ] 切了新分支 `fix/qwen3-spk-overdetect`
- [ ] 至少 6 commit, 红绿分明
- [ ] `venv/bin/python -m pytest tests/unit/ -v` 全绿
- [ ] `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/` 全绿
- [ ] `venv/bin/python /tmp/eval_set_n1_verify.py` 4/4 ✓
- [ ] 调研报告补 "修复" 章节
- [ ] 本 prompt 归档到 `archive/`
- [ ] CLAUDE.md 更新 cluster_merge 段说明
- [ ] PR 标题: `fix(qwen3): spk over-detect — worker 对 sherpa-supported 格式跳 ffmpeg + cluster_merge 多人模式吃 minor`

---

## 8. 跑出来后通知人

只在以下情况停下来问用户:
1. ASR vendor 完全无法读 mp3 / flac, 必须 ffmpeg 强制转 wav, 而且 diarize/ASR 两条路改造超出 1 个文件 5 行修改 → 询问是否允许重构 transcribe 接口
2. Eval set 某个 audio 反复调参怎么都过不了 (多人 4spk / 6spk 退化超过 ±1) → 询问是否接受退化或要新调参方向
3. 修复方案在性能 (RTF) 上引入显著退化 (> 10%) → 询问是否接受

**除此之外的所有决策, 自己拍板, 跑通 DoD 为止**.

完成后给一份简要报告:
- 每个 commit 标题 + 一行说明
- Eval set 修复后数据表
- RTF 对比 (修复前 0.20-0.22 vs 修复后)
- 还有什么残留 / 后续 PR 应该做的
