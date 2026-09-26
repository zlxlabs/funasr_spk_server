# Qwen3 speaker 过检测 (over-detect) 归因调研 — 新 session prompt

> 任务类型: **归因调研**(不写功能代码,先定位罪魁)
> 预估 session: 1-2 次,主要是跑数据 + diff 历史
> 风险: 不动 src/, 找到根因后再单独开 PR 修

## 1. 现象 — 长音频 spk 数被夸大

最近 (2026-05-17) 真并发实测 (commit `92e8442`),两个长音频 spk 数明显偏多:

| audio | 真实 spk | N=2 实测 | N=3 实测 |
|---|:---:|:---:|:---:|
| `tmp_long_audio/eval_set/audio_4spk.m4a` (44min) | **4** | **5** | **5** |
| `tmp_long_audio/eval_set/audio_2spk_60min.mp3` (60min) | **2** | **4** | **4** |
| `tmp_long_audio/eval_set/audio_1spk_real.m4a` (16min) | 1 | (未测) | 1 ✓ |

N=2 和 N=3 实测一致 → **不是并发压力问题, 是算法/参数问题**。

raw label 极度跳跃 (e.g. 44min audio 测出 `Speaker2, 3, 7, 15, 66`) → sherpa diarize 内部原始 cluster 数 ≥66,经过 `filter_spurious_speakers` + `apply_cluster_centroid_merge` 兜底后剩 5。**说明 PR3 cluster_merge 对这两个 audio 还不够 aggressive**,但具体哪一步退化了,要查。

## 2. "之前挺准" 的历史证据

`docs/开发/archive/PR4-149min校对稿对比分析.md`:

- **149min 双人 audio (吴明辉 + 程曼祺)** 在 v4/v5 调优后:
  - speaker 字符准确率 96.80-96.89%
  - segment 多数派准确率 92.17-92.67%
  - 明确测出 **2 个 speaker** (`Speaker1` 吴明辉 + `Speaker3` 程曼祺,84.8% / 14.7% 时长占比)
- 同份报告也提到 **"83min 出现 8 speaker 过分散"** — 说明 over-detect 在特定 audio 上一直存在

所以"之前挺准"是真的(149min 双人),但 over-detect 是**audio-specific** 问题,从 PR4 起就遗留至今。本调研要回答:**当前的 over-detect 是 PR4 之后哪次 commit 引入 / 加剧 / 没修?**

## 3. 待回答的核心问题(按优先级)

### Q1. 当前 60min-2spk / 44min-4spk 的 over-detect 是新引入 (regression) 还是历史遗留?

**验证方法**: git bisect 风格,挑几个关键 commit checkout 跑一次 spk 数对比:
- `405abf6` (silence-align spike 前一刻的 PR3+PR4 完整版)
- `b5b9350` (PR3 cluster_merge 集成到 transcriber)
- `41abaa9` (PR2 short_segment_guard 集成)
- `668e524` (PR4 PoC 全套脚本) — 这是 149min 96.80% 数据的对应 commit

如果 N=1 跑 60min/44min 在 `668e524` 时 spk 正确 → **后续 PR3/PR4 工程化引入了 regression**
如果 `668e524` 也 over-detect → audio 本身的问题,跟历史 commit 无关

> 注: 调研只看 N=1(单 task), 排除并发干扰. 别用 N=2 复杂化.

### Q2. 如果是 regression, 是哪个组件?

按嫌疑顺序:

1. **`apply_cluster_centroid_merge`** (PR3, `src/core/qwen3/cluster_merge.py`) — 直接合并 cluster,threshold 不够激进就 over-detect
2. **`filter_spurious_speakers`** (`src/core/qwen3/merge.py:38`) — 默认 `min_speaker_total=2.0`/`min_speaker_share=0.01`,对 60min audio (3600s × 1% = 36s) 阈值偏低,容易漏滤
3. **`apply_short_segment_guard`** (PR4) — 理论上不影响 cluster 数,但可能合并/拆分后改变 speaker 时长分布,间接影响 filter 判定
4. **sherpa diarize 本身参数** (`Qwen3Config.cluster_threshold=0.9`, `num_threads=4`) — 默认值在 PR3 之后没动过
5. **`silence_align` (本次)** — 不动 diarize/cluster,只动 segment 时间戳,**理论上不影响 spk 数**。但本次实测都是 silence_align=ON,要排除掉,跑一次 OFF 对比

### Q3. PR3 cluster_merge 5 个 threshold 现在的值合不合理?

`Qwen3Config` 默认:
```python
cluster_merge_min_main_share: 0.03           # 占比 >= 3% 才算 main
cluster_merge_relabel_threshold: 0.55         # minor → main 阈值
cluster_merge_main_threshold: 0.78            # main 间合并
cluster_merge_dominant_share: 0.6             # 触发 dominant 模式
cluster_merge_dominant_threshold: 0.6         # dominant 模式合并阈值
```

PR3 PoC 报告 (`docs/开发/archive/PR4-长音频质量与并发性能-新session-prompt.md` 附近) 在 1/2/3+/4/6 speaker 5 个样本 "全过" — 全过的具体 audio 是什么?跟我们现在测的 60min-2spk / 44min-4spk 是不是**同一批** audio?如果不是,可能 PR3 调参时**没拿这两个 audio 入 eval_set**,所以现在才看到偏差。

## 4. 调研入口 — 关键文件 & commit

### 代码
- `src/core/qwen3/cluster_merge.py` — PR3 合并算法
- `src/core/qwen3/merge.py` — `filter_spurious_speakers` (38-88 行)
- `src/core/qwen3_transcriber.py:78-118` — `apply_cluster_centroid_merge_to_turns` 调用入口
- `src/core/qwen3_transcriber.py:330-370` — `transcribe` 流程里 cluster_merge / short_guard / silence_align 串联顺序
- `src/core/qwen3/diarize.py` — sherpa diarize 入口

### 文档
- `docs/开发/archive/PR4-149min校对稿对比分析.md` — 149min 双人 96.80% 的来源 + "83min 8 speaker 过分散" 的最早记录
- `spikes/qwen3_silence_align/scripts/_bench_n2_result.json` — 本次 N=2 实测数据(被 ignore, 跑 `bench_n2_concurrency.py` 重生成)

### 历史 commit (按时间倒序,关键节点)
| commit | 内容 | 备注 |
|---|---|---|
| `92e8442` | N=2/N=3 bench 脚本 | 本次发现 spk over-detect 的证据 |
| `009e8bc` | silence-align 携进 src/ | 不动 diarize/cluster, 但需排除嫌疑 |
| `b5b9350` | cluster_merge 集成到 transcriber (PR3) | 主要嫌疑对象 |
| `c0cc1a2` | Qwen3Config 加 cluster_merge_* 字段 | threshold 设计源头 |
| `41abaa9` | transcriber 集成 short_segment_guard (PR2) | 副嫌疑 |
| `668e524` | PR4 PoC 全套脚本 | 149min 96.80% 数据的对应 commit, 适合做 baseline |

## 5. 推荐调研路径

### 第 1 步: 排除 silence_align 嫌疑(15 分钟)
跑 N=1 60min audio 一次,`FUNASR_QWEN3_SILENCE_ALIGN_ENABLED=false`,对比 spk 数:
- 若 spk=2 (正确) → **silence_align 是罪魁**(意外!需深查)
- 若 spk=4 (跟 ON 一致) → silence_align 无关,继续下一步

预期: silence_align=OFF 仍 over-detect(因为它只动时间戳不动 speaker)。

### 第 2 步: 排除并发干扰 (15 分钟)
N=1 单跑 60min audio (silence_align 不管):
- 若 spk=4 → over-detect 跟并发完全无关,纯算法/参数问题
- 若 spk=2 → 并发竞争是诱因(但 N=2/N=3 实测一致,不太可能)

### 第 3 步: 二分定位 commit (1-2 小时)
N=1 单跑 60min audio,在以下 commit 各跑一次:
1. `668e524` (PR4 PoC) — baseline
2. `b5b9350` (PR3 cluster_merge 集成) — PR3 加入
3. `41abaa9` (PR2 short_guard 集成) — PR2 加入

找出 spk 数从 2 → 4 的拐点 commit。

### 第 4 步: 在拐点 commit 里 diff 参数
查 `Qwen3Config` 字段值 + 该 commit 的算法实现,跟当前 main 对比看是什么变化引入退化。

### 第 5 步: 写一个最小 unit test 复现 over-detect
用 60min audio 的 sherpa diarize 中间输出(可以 dump 出来) → 喂给 `apply_cluster_centroid_merge` → assert spk=2。这是后续修复 PR 的 red test。

## 6. 工作目录 + 环境

- 仓库: `~/Dev/projects/250729_funasr_spk_server/funasr_spk_server`
- 分支: `spike/qwen3-diarize-poc` (当前 HEAD `92e8442`,3 commit ahead origin 已 push)
- venv: `venv/bin/python`
- 关键音频(都在 `tmp_long_audio/eval_set/`):
  - `audio_2spk_60min.mp3` (60min, 真实 2 人, 当前测 4) — **主调研对象**
  - `audio_4spk.m4a` (44min, 4 人, 当前测 5) — **副调研对象**
  - `audio_1spk_real.m4a` (16min, 1 人, 当前测 1 ✓) — control
  - `audio_6spk_60min.m4a` (60min, 6 人) — 可选 stress test
- 模型权重已就绪(`venv/bin/python -c "..."` 检查 `models/qwen3_diarize/*` 完整)

## 7. 约束 & 注意

- **TDD 严格执行** (memory: `feedback-tdd-strict`): 修复阶段先红再绿,不要先写代码后补测试
- 调研阶段**只跑数据 + 读 commit,不改 src/**。修复在后续独立 PR 走
- 这俩 audio 的 sherpa diarize 中间数据(turns + embedding)如果 dump 出来,**别 commit JSON 到 git**(沿用 spike 约定 `data/*.json` ignore)
- N=1 跑 60min audio 单 task ~7-12 min,耐心等,别开 N=2/N=3 复杂化

## 8. 完成定义 (DoD)

1. 一句话回答 Q1: regression 或历史遗留
2. 如果是 regression, 指出引入 commit + 罪魁组件 + 具体哪个参数/逻辑
3. 如果是历史遗留, 解释 PR3 PoC 时为什么没暴露(eval_set 不含这俩 audio?), 并给出修复方向(threshold 调整 / 算法增强 / 显式 num_speakers)
4. 产出: 调研报告 .md 放 `docs/开发/archive/spk-over-detect-归因调研结果.md` + 复现脚本路径
5. **不修代码,不 commit src/**。修复留独立 PR

## 9. 用户上下文

- 用户记忆里 "之前 spk 识别挺准的" 来自 PR4 149min 双人 audio (确实准, 96.80%)
- 但 60min-2spk audio 是后来加入 eval_set 的(具体哪个 commit 引入待查),可能从未在 PR3/PR4 调参时被覆盖
- 用户对 over-detect 的反感程度 > 对 RTF 微涨的反感程度(他刚拒绝降并发的提议,先要先搞清 spk)
