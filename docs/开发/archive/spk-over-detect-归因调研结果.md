# Qwen3 speaker 过检测 (over-detect) 归因调研 — 结果

> 日期: 2026-05-17
> Session 起点: `docs/开发/archive/Qwen3-spk-over-detect-归因调研-新session-prompt.md`
> 输出代码: `tests/unit/test_qwen3_spk_overdetect_repro.py` (xfail red test)
> 工作目录 HEAD: `92e8442` 分支 `spike/qwen3-diarize-poc`

---

## TL;DR

**罪魁 commit**: `cd578a8 fix(qwen3): worker 加 ffmpeg audio format 转换 + 长音频并发冒烟脚本` (2026-05-15 22:42)

**机制**: 该 commit 在 `qwen3_worker_process.process_task` 加了 `convert_to_wav` (ffmpeg → 16kHz mono wav) 预处理. 修复了 m4a/aac 无法被 sherpa-onnx libsndfile 直接读取的 production bug, 但**改变了喂给 sherpa diarize 的音频字节** (mean_abs diff 0.001137 ~ rms_diff 0.4%). 这点差异通过 pyannote + NeMo embedding 放大, 把 60min-2spk audio 的 FastClustering@0.9 输出从 7 cluster (主 2 + 5 个 <3s 噪声) 改为 11 cluster (主 2 + 2 个 43.2s/61.6s 中长噪声 + 7 个 <2s 噪声). 那两个 43.2s / 61.6s cluster **恰好突破 `filter_spurious_speakers` 的 `max(2s, 1% × 3600s) = 36s` 阈值**, 滤不掉 → final 4 speakers.

**性质**: regression. 触发 audio 的边界特性 (long-form + 2 distinct speakers + 噪声分布)，**非 audio-specific 历史遗留**.

**44min-4spk 输出 5 speakers (含 Speaker66 英文歌)**: 跟 PR3 PoC 时期完全一致, **非 regression**, 历史就是这样.

---

## 1. 现象 (复盘)

`92e8442` HEAD, N=2 真并发实测 (`bench_n2_concurrency.py` → `_bench_n2_result.json`):

| audio | 真实 spk | 当前 N=2 实测 | PR3 PoC 时期 (2026-05-16) | 是否 regression |
|---|:---:|:---:|:---:|:---:|
| `audio_2spk_60min.mp3` (60min) | **2** | **4** (Speaker1, 3, 4, 6) | **2** (Speaker1 87.5% + Speaker3 12.5%) | **✗ regression** |
| `audio_4spk.m4a` (44min) | 4 | 5 (Speaker2, 3, 7, 15, 66) | **5** (相同, 含 Speaker66 英文歌) | 无 regression (历史如此) |
| `audio_1spk_real.m4a` (16min) | 1 | 1 ✓ | 1 ✓ | 无 regression |
| `audio_6spk_60min.m4a` (60min) | 6 | (未测) | 6 ✓ | 无 regression |

PR3 PoC 时期数据来源: `tmp_long_audio/eval_set/README.md` + `tmp_long_audio/eval_set/2spk_pipeline/baseline/audio_2spk_60min.qwen3_long_poc.json` 等三阶段 JSON.

---

## 2. 调研路径

按 prompt 5 步走, 但中途两次根据数据修正假设:

### Step 0: 复用 PR3 PoC baseline JSON 比对 (替代昂贵的 ASR + diarize 完整重跑)

直接读 `tmp_long_audio/eval_set/2spk_pipeline/baseline/*.json` 的 `diarization_turns` 字段:

- **2spk PR3 PoC baseline**: 2 cluster (Speaker1, Speaker3) — 原始 sherpa 输出就是 2 spk
- **2spk PR3 PoC merged**: 2 cluster (cluster_merge log 0 entries, nothing to merge)
- **4spk PR3 PoC baseline**: 9 cluster → merged 5 cluster (Speaker2, 3, 7, 15, 66 — 跟当前实测一字不差)

**关键发现**: 2spk regression 在 **sherpa diarize 上游层**, 而不在 cluster_merge / filter_spurious 算法层 — 因为 PR3 PoC 时 sherpa 直接吐 2 cluster, 没东西可以"过分聚类".

### Step 1: 排除 num_threads 假设

最早怀疑: commit `9692929 chore(qwen3): num_threads 默认 8→4` (2026-05-16 21:48) 影响了 sherpa ONNX 多线程浮点 reduce 顺序 → embedding 微差 → 边界聚类突变.

**实验** (`/tmp/diarize_60min_sweep_v2.py` on `audio_2spk_60min.mp3` 直读):

| num_threads | turns | clusters | cluster ids | wall (60min audio) |
|---|---:|---:|---|---:|
| 8 | 115 | 7 | `[0,1,2,3,8,9,18]` | 267.2s |
| 4 | 115 | 7 | `[0,1,2,3,8,9,18]` | 297.5s |

cluster durations 一字不差: `{0:3128.6, 2:451.6, 8:2.8, 9:1.6, 1:0.4, 3:0.4, 18:0.4}`.

→ **`num_threads` 完全不影响 sherpa diarize 输出**. 这两份结果经过 `filter_spurious_speakers(min_speaker_share=0.01, audio_duration=3600)` 阈值 36s 过滤后都是 2 cluster ✓ (复现 PR3 PoC 行为).

假设证伪.

### Step 2: ffmpeg → wav 转换路径假设

发现 PR3 PoC `qwen3_long_audio_poc.py` 直接传 mp3 给 `run_diarization()`, 而 production `qwen3_worker_process.process_task` 在 commit `cd578a8` 之后**统一先 ffmpeg → wav 转换再喂给 transcribe**.

**实验**: 先 `ffmpeg -ar 16000 -ac 1` 转 wav, 再调 `run_diarization(wav)`, num_threads=4:

```
audio: /tmp/audio_2spk_60min.converted.wav (115MB)
turns: 132 (vs mp3 直读 115)
clusters: 11 (vs mp3 直读 7)
cluster_ids: [0, 2, 3, 5, 6, 7, 12, 15, 16, 17, 18]
cluster_durations: {
  '0': 3089.7,  # 主 (86%)
  '5':  394.4,  # 主 (11%)
  '3':   61.6,  # ⚠️ 噪声但 > 36s 阈值
  '2':   43.2,  # ⚠️ 噪声但 > 36s 阈值
  '6': 0.4, '7': 0.4, '12': 0.4, '15': 0.8,
  '16': 0.4, '17': 0.3, '18': 1.7
}
```

经过 `filter_spurious_speakers` (阈值 36s):
- cluster 0 (3089.7s) > 36 ✓ keep
- cluster 5 (394.4s) > 36 ✓ keep
- cluster 3 (61.6s) > 36 ✓ keep ← **噪声但漏网**
- cluster 2 (43.2s) > 36 ✓ keep ← **噪声但漏网**
- 其余 7 个 < 36s ✗ 滤掉

→ **4 cluster 留下** (1-indexed: Speaker[1, 3, 4, 6]) — **完美对应 production bench 实测**.

### Step 3: 音频字节级证据

`_load_audio_mono_16k(mp3)` vs `_load_audio_mono_16k(ffmpeg-converted-wav)` (脚本 `/tmp/audio_compare.py`):

| 指标 | 值 |
|---|---|
| 样本数 | 一致 (57600016 = 3600s × 16kHz) ✓ |
| dtype / sr | 都是 float32 / 16000 ✓ |
| RMS | mp3 0.139156 vs wav 0.138812 (0.25% 差) |
| `max_abs(diff)` | **1.605740** |
| `mean_abs(diff)` | **0.001137** |
| `rms_diff` | **0.004259** (~0.4%) |
| 相关系数 | 0.999534 |
| 样本中近相等 (diff < 1e-5) | **仅 4.80%** |

→ ffmpeg 解码出的 PCM 跟 librosa 解码 mp3 的 PCM **不是 bit-identical, 也不接近 bit-identical** (只有 4.8% 样本近相等). 喂给 pyannote segmentation + NeMo TitaNet embedding 后, 在 cluster_threshold=0.9 边界 case 上聚类结果跳变.

---

## 3. 根因分析

### 3.1 罪魁 commit

```
cd578a886ad58e50af937238b2adcb7b0d46f925
Author: <user> <zj1123581321@gmail.com>
Date:   Fri May 15 22:42:34 2026 +0800

    fix(qwen3): worker 加 ffmpeg audio format 转换 + 长音频并发冒烟脚本
```

修复了真实 m4a/aac 长音频 production bug (sherpa libsndfile 不支持 m4a 报错 → 任务 FAILED), **代价**: 把 mp3 (sherpa 本来能跑通的) 也强制走 ffmpeg → wav 路径, 改变了喂给 sherpa 的 audio bytes.

### 3.2 为什么 2spk 触发 / 4spk 不触发

**4spk 不触发**:
- PR3 PoC: 9 raw clusters → cluster_merge 收敛到 5 (4 真人 + 1 英文歌 Speaker66)
- 当前: 5 clusters (一字不差)
- 解释: 4 个真人差异大, sherpa FastClustering 离 0.9 阈值边界很远, 即使 ffmpeg 引入 0.4% 音频 drift 也撼不动聚类结果
- Speaker66 英文歌是 sherpa 算法把非语音段独立聚类的合法行为, PR3 PoC 就接受了

**2spk 触发**:
- 2 人对话 (吴明辉 男声 + 程曼祺 女声), 60 分钟长, 段内有大量短停顿 / 呼吸声 / 笑声碎片
- sherpa FastClustering 对这种 audio 在阈值边界附近, 0.4% 音频 drift 足以让某些非语音 frame 跨阈值 → 多吐 4 个非语音 cluster (其中 2 个 43.2s / 61.6s, 2 个 <2s)
- `filter_spurious_speakers` 的 `min_speaker_share=0.01` 在 60min audio 上换算 36s — 那两个 43.2s / 61.6s 恰好超过, 漏网

**1spk 不触发**:
- 单人无聚类歧义

### 3.3 为什么 num_threads 不是罪魁 (虽然时间上巧合)

`9692929` 改 8→4 是 perf 优化 (-11.5% wall), 时间上巧合在 regression 引入前后. 但实验证明 num_threads 不影响 sherpa diarize 输出 (mp3 直读路径 t=8 / t=4 输出字节一致).

那次 PoC 调优只测 wall time, **没测 spk 数变化** — 巧合的是, 那次 PoC 用的 audio (`spikes/qwen3_mac_hw_accel/num_threads_tuning.md` 中的 4spk_44min / 1spk_16min) 都不是边界 case, 所以即使有微差也看不出来.

---

## 4. 修复方向 (供下一个 PR 选择, 本调研不修)

按 "改动小 → 大" 排:

1. **调高 `filter_spurious_speakers` 阈值** (最简, 但治标不治本)
   - `min_speaker_share: 0.01 → 0.02` 让阈值变成 72s, 滤掉 43.2s/61.6s
   - 风险: 在更短音频 (10min × 2% = 12s) 上可能误杀真实少数派 speaker
2. **跳过 ffmpeg, 走 librosa 统一加载** (改回 PR3 PoC 路径)
   - 在 `_load_audio_mono_16k` 让所有非 wav 格式都走 librosa, **同时**在 worker 里也跳过 `convert_to_wav`
   - 但要保证 sherpa-onnx 本身能消费 librosa 输出的 ndarray (它确实能, sherpa pipeline 接受 numpy)
   - 风险: librosa 对 m4a 也走 ffmpeg backend (audioread), 跟当前直接 ffmpeg 转 wav 殊途同归, 不一定 bit-identical 解决问题; 需要先实验
3. **提高 sherpa `cluster_threshold` 0.9 → 1.0+** 让聚类更激进合并
   - 副作用: 多人音频 (4spk/6spk) 可能掉 cluster, 需要全 eval_set 回测
4. **`cluster_centroid_merge` 多人模式 (dominant) 触发更激进合并**
   - 60min-2spk audio 主 cluster 占 86%, 已经 ≥ `dominant_share=0.6`, 应该会触发 dominant 模式
   - 但当前 dominant 模式只在 "remaining_mains 还有 main" 时才合并, 而 43.2s / 61.6s cluster 占比 1.2% / 1.7% < `cluster_merge_min_main_share=0.03` → 它们根本不算 "main", 不会进 dominant 流程
   - 修复: 让 dominant 模式也比较 minor cluster 跟 dominant 的 cosine, 接近就合并; 或者降 `cluster_merge_min_main_share`

**推荐组合**: 方向 2 (走 librosa fallback, 不走 ffmpeg 转 wav, 但保留 m4a 兜底) + 方向 4 (`cluster_merge` 多人模式扩展到 minor cluster). 先做实验验证方向 2 能否让 sherpa 输出收敛到 2 cluster.

---

## 5. 复现脚本路径

| 脚本 | 路径 | 作用 |
|---|---|---|
| 通用 sweep | `/tmp/diarize_60min_sweep_v2.py` | 接受 audio + num_threads, 输出 cluster 数 + 时长 |
| mp3 vs wav | `/tmp/diarize_wav_v_mp3.py` | ffmpeg → wav, 再调 diarize 对比 |
| audio bytes 对比 | `/tmp/audio_compare.py` | 实测 mp3 / wav 加载后 numpy 数组差异 |
| Red test | `tests/unit/test_qwen3_spk_overdetect_repro.py` | xfail 复现 + 对照, 入库 |

实验数据 JSON: `/tmp/diarize_60min_sweep_result.json`, `/tmp/diarize_wav_result.json` (临时文件, 未 commit).

---

## 6. DoD 自查

| 问题 | 状态 | 说明 |
|---|---|---|
| Q1: regression 还是历史遗留? | ✅ | **regression**: 60min-2spk 从 2 → 4 spk 由 cd578a8 引入 |
| Q2: 罪魁组件 + 参数 | ✅ | `qwen3_worker_process.convert_to_wav` (ffmpeg path) → sherpa diarize → filter_spurious 36s 阈值漏网 |
| Q3: 4spk/6spk 为什么不影响 | ✅ | 多人 audio 在阈值边界远, 微小音频差异撼不动聚类 |
| 调研报告产出 | ✅ | 本文件 |
| Red unit test | ✅ | `tests/unit/test_qwen3_spk_overdetect_repro.py` xfail+passing 对照 |
| 不动 src/ | ✅ | 调研阶段无 src 改动 |

---

## 7. 备忘 — 还没做的 / 后续 PR 应该做的

- [ ] 在 ffmpeg 路径下也跑 num_threads=8, 排除 wav+多线程组合 case (我只跑了 wav+t4 + mp3+t8/t4)
- [x] 实验方向 2: 让 sherpa 直接消费 librosa decode 的 ndarray (skip ffmpeg+wav), 看 2spk 能否收敛到 2 cluster
  → 修复 PR `fix/qwen3-spk-overdetect` commit `bbdb173` 实现 (worker 跳 ffmpeg for sherpa-supported), 60min-2spk 实测 4 → 2 spk ✓ (§9)
- [ ] 加 `audio_6spk_60min.m4a` 到 eval_set 回测 (m4a 必走 ffmpeg, 看是否也 over-detect)
- [ ] PR3 cluster_merge 5 个 threshold 在新 eval set (含 2spk_60min) 上重新调参
- [x] 加 N=1 单跑 regression test: 验证 production pipeline 在 2spk_60min 上能稳定输出 2 spk
  → `tests/integration/test_qwen3_spk_overdetect_fix.py::test_2spk_60min_mp3_no_over_detect`

---

## 8. 时间线 (commit 倒序)

| commit | 时间 | 内容 | 跟本调研关系 |
|---|---|---|---|
| `92e8442` | 2026-05-17 19:50 | N=2/N=3 并发 bench | 本调研起点, 暴露 over-detect |
| `009e8bc` | 2026-05-17 | silence-align 携进 src/ | 不影响 spk 数, 实验已排除 |
| `9692929` | 2026-05-16 21:48 | num_threads 8→4 | 时间巧合, **实验证伪非罪魁** |
| `b5b9350` | 2026-05-16 17:16 | cluster_merge 集成 + librosa fallback | 加了 librosa fallback (但只在 sf.read 失败时触发) |
| `cd578a8` | 2026-05-15 22:42 | **worker 加 ffmpeg 转换** | **罪魁** — 强制走 ffmpeg → wav, 改变 audio bytes |
| `668e524` | (PR4 PoC) | 全套 PoC 脚本 | PR3 PoC baseline JSON 数据来源, 直接读 mp3 → 干净路径 |

---

## 9. 修复完成 (PR `fix/qwen3-spk-overdetect`)

调研结论在 PR `fix/qwen3-spk-overdetect` 上落地, 严格 TDD 7 commit (commit 0 baseline + commit 1-6 红绿循环), 修复后 60min-2spk 实测 4 → 2 spk ✓.

### 9.1 修复方案 (方向 2 + 方向 4 组合)

**方向 2 — 治本 (commit `bbdb173`)**:
- `src/core/qwen3_worker_process.py` 引入 `SHERPA_SUPPORTED_EXTENSIONS = {.wav, .flac, .ogg, .mp3, .opus}`
- 这些格式 sherpa diarize 通过 `_load_audio_mono_16k` 的 librosa fallback 可以直读, 跳过 ffmpeg `convert_to_wav`
- 只有 m4a/aac/mp4/mov/webm 等 libsndfile/librosa 都读不了的格式才走 ffmpeg 转码
- sherpa 拿到的 audio 跟 PR3 PoC 时期一致, 不再触发 over-detect

**方向 4 — 兜底 (commit `a843b27`)**:
- `src/core/qwen3/cluster_merge.py` 新增 `merge_minor_into_dominant` 函数
- `apply_cluster_centroid_merge` 加 `dominant_minor_threshold` 参数 (默认 0.5, 比 minor->main relabel 阈值 0.55 宽松)
- 当 dominant cluster share ≥ 0.6 触发 dominant 模式时, 把所有 minor cluster 跟 dominant centroid 比较, cos ≥ 0.5 合到 dominant
- log 加 `action=minor_folded_into_dominant` 标记
- 即使将来又有 audio 触发同形 over-detect (新解码器 / 新 audio profile), 兜底层能拦截

### 9.2 Commit 列表 (7 个, 严格 TDD)

| commit | 类型 | 内容 |
|---|---|---|
| `58e8482` | chore | 落档 spk over-detect 调研产物 baseline |
| `b7c9cf4` | test 🔴 | 加 60min-2spk over-detect integration red test |
| `cab4699` | test 🔴 | worker 对 sherpa-supported 格式 (mp3/flac/ogg/opus) 跳 ffmpeg red test |
| `bbdb173` | fix 🟢 | worker 跳 sherpa-supported 格式 ffmpeg (方向 2) |
| `3e0adc5` | test 🔴 | cluster_merge dominant 模式吃 minor cluster red test |
| `a843b27` | feat 🟢 | cluster_merge dominant minor-fold + config 字段 (方向 4) |
| `e235e22` | test | 取消 over-detect xfail, 改为 cluster_merge minor-fold regression guard |

### 9.3 Eval set 实测 (修复后)

N=1 单 worker pool, 全 4 audio (2spk_60min 由 integration test 覆盖, 其余 3 audio 由 `/tmp/eval_set_n1_verify_rest3.py` 覆盖).

修复前实测 (commit `b7c9cf4` 时期, baseline log `/tmp/red_test_60min_baseline.log`):
- 2spk_60min: **4 spk** (Speaker1, Speaker3, Speaker4, Speaker6) ✗ over-detect

修复后实测 (commit `e235e22` 时期):

| audio | 真实 | 修复前 | 修复后 | RTF | wall | speakers |
|---|---:|---:|---:|---:|---:|---|
| 1spk_real (16.2min m4a) | 1 | 1 | **1** ✓ | 0.149 | 144.9s | `Speaker1` |
| 2spk_60min (60min mp3) | 2 | **4** | **2** ✓ | 0.158 | 569.6s | `Speaker1, Speaker3` |
| 4spk (43.8min m4a) | 4 | 5 (含 Speaker66 英文歌) | **5** ✓ | 0.144 | 377.5s | `Speaker2, 3, 7, 15, 66 (英文歌)` |
| 6spk_60min (60min m4a) | 6 | 6 | **6** ✓ | 0.148 | 533.7s | `Speaker13, 20, 23, 41, 46, 74` |

数据源:
- 2spk_60min: integration test `tests/integration/test_qwen3_spk_overdetect_fix.py::test_2spk_60min_mp3_no_over_detect` PASSED log
- 其余 3 audio: 临时脚本 `/tmp/eval_set_n1_verify_rest3.py` (N=1 串行跑), 总耗时 1056s

所有 audio 全过, 跟 `tmp_long_audio/eval_set/README.md` 中"当前评测结果 (2026-05-16)" baseline 完全对齐, 即 over-detect 修复对其他 audio 无副作用.

### 9.4 测试覆盖

- 全套 unit 297 passed (相比修复前 281 passed + 1 xfailed, 新增 16 case)
- Integration `test_qwen3_spk_overdetect_fix.py::test_2spk_60min_mp3_no_over_detect` PASSED (569.6s)
- Unit 关键 case:
  - `test_qwen3_worker_skip_ffmpeg_for_sherpa.py`: 11 parametrize case (5 sherpa-supported skip + 5 non-sherpa convert + 1 大小写)
  - `test_cluster_merge_dominant_minor_fold.py`: 3 case (minor close fold / log / share<0.6 不触发)
  - `test_qwen3_spk_overdetect_repro.py`: 2 case (regression guard + clean path sanity, xfail 取消)
  - `test_config_qwen3_cluster_merge.py`: +2 case (默认值 + env override)

### 9.5 性能 (RTF)

修复对 RTF 几乎无影响:
- worker 跳 ffmpeg 反而省了一次 ffmpeg 转码 (60min mp3 节省 ~3-5s)
- cluster_merge minor-fold 仅在 dominant 模式触发时多算 minor 数量次 cos (< 10 次, 毫秒级)
- 整体 RTF 修复前后 0.16-0.17 区间, 无显著退化

### 9.6 配置 / env override

新加配置字段:
- `Qwen3Config.cluster_merge_dominant_minor_threshold: float = 0.5`
- env `FUNASR_QWEN3_CLUSTER_MERGE_DOMINANT_MINOR_THRESHOLD`

调高 (e.g. 0.6) 更保守, 噪声 cluster 不易被吃; 调低 (e.g. 0.4) 更激进, 适合特定 audio 微调.
