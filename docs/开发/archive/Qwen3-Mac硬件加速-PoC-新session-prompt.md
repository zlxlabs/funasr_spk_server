# 新 session 启动 prompt — Qwen3 Mac 硬件加速 PoC

> 日期: 2026-05-16
> 上一阶段: PR4 工程化 + follow-up 已落地 (extractor cache, server e2e, no-dedup contract test)
> 触发动机: 长音频 N=2 真并发实测 RTF 从单跑 0.10 翻到 0.30-0.43, 用户观察到前期 CPU 满 + GPU 0% 空等
> 阶段定位: **PoC** (先验证可行性 + 测真实收益, 不直接工程化), PoC 通过再考虑下个工程化 PR

## Prompt 正文（复制以下内容到新 session）

你接手 funasr_spk_server 的 Qwen3 引擎 Mac 硬件加速 PoC 阶段。**目标是解决双任务并发时前期 CPU 打满, GPU 长时间 0% 空等的问题**。验证多种 Mac 硬件加速路线 (CoreML/ANE/Metal), 测真实收益, 不做工程化集成。如 PoC 验证有效, 后续单独开 PR 工程化。

### 项目路径
`~/Dev/projects/250729_funasr_spk_server/funasr_spk_server`

### 必读 (按顺序, 用于建立 context)

1. `docs/开发/PR4-工程化-新session-prompt.md` — 上阶段 PR4 工程化 prompt + 已拍板决策
2. `docs/开发/重构计划-ASR引擎抽象.md` — 全部 PR1-PR4 + follow-up 总结 (含本次 PoC 触发的实测数据)
3. `docs/部署.md` 第 160-165 行 — pool 章节, "长音频并发 RTF 翻倍预警"
4. `src/core/qwen3/asr.py` — ASR engine 入口, **`build_engine_config` 默认 onnx_provider="CPU"** (line 73 注释 "Mac 上 ANE/CoreML 实测无加速" — 这个结论可能过时, 重新验证)
5. `src/core/qwen3/diarize.py` — sherpa OfflineSpeakerDiarization, provider 来自 `Qwen3Config.provider` (default "cpu")
6. `src/core/qwen3_transcriber.py:39-77` — `build_embedding_extractor_fn`, sherpa SpeakerEmbeddingExtractorConfig provider 同上
7. `src/core/vendor/qwen_asr_gguf/inference/asr.py` — vendor ASR engine, 看 encoder/decoder 是 batch 还是 streaming
8. `tests/integration/test_qwen3_server_websocket_e2e.py` — 真 server+client+model+cache 套件, PoC 复用此 fixture
9. `tests/manual/server/smoke_qwen3_concurrent.py` — 手工并发脚本, 改造时参考
10. (上 session 写的 ad-hoc) `/tmp/qwen3_long_audio_real_concurrent.py` — 长音频并发 perf 测试脚本

### 问题定义

**生产场景**: N=2 worker 并发跑 2 个长音频 (16min + 44min), wall 13.5min, 期间观察:
- 前期 CPU 100% (htop 看 user threads ~16 抢 10 cores)
- 同时 GPU 利用率 ~0% (`asitop` / `mactop` 看 Metal usage)
- 中期 GPU 才起来 (LLM decode 阶段)
- 后期 CPU 又满 (cluster_merge per-segment embedding)

**根因 (上 session 已分析)**:
```
worker 拿任务
   │
   ├─► [CPU] ffmpeg 转码 m4a/aac (~1-3s)
   ├─► [CPU] audio loading (sf/librosa, ~0.5s)
   │
   ├─► 并行:
   │     ├─ [CPU] ASR encoder ONNX (provider=cpu)  ←━━ 长音频 30-60s ⚠️ GPU 空等
   │     └─ [CPU] sherpa segmentation + nemo-titanet (8 threads)
   │
   ├─► [GPU] ASR LLM decoder (llama.cpp Metal)  ←━━ GPU 唯一工作阶段
   │
   ├─► [CPU] PR3 cluster_merge: nemo-titanet 各段 embedding (CPU)
   │
   └─► [CPU] PR2 short_guard (<0.1s)
```

**N=2 时**: 2 worker × 8 threads = 16 threads 抢 10 cores, CPU 严重过订阅, GPU 空转。

### PoC 候选方向 (按 ROI 升序探索)

#### 方向 0: 现状精确 profiling (必做, 2-4h)

不动 production 代码, 加临时 timing logger, 跑长音频 N=1 / N=2 各一遍, 输出 per-stage 时间分布:

| 阶段 | 单 task 耗时 | N=2 时单 task 耗时 | CPU/GPU |
|---|---|---|---|
| ffmpeg 转码 | ? | ? | CPU |
| audio loading | ? | ? | CPU |
| ASR encoder | ? | ? | CPU |
| sherpa segmentation | ? | ? | CPU |
| sherpa embedding | ? | ? | CPU |
| ASR LLM decoder | ? | ? | GPU |
| cluster_merge embedding | ? | ? | CPU |
| 后处理 | ? | ? | CPU |

**度量工具**:
- 临时改 `src/core/qwen3/asr.py` / `diarize.py` / `qwen3_transcriber.py` 加 `logger.info("[stage X] elapsed=...")`
- 用 `asitop` / `mactop` (`brew install asitop`) 监控 GPU/ANE/CPU 真实 utilization
- 写到 `spikes/qwen3_mac_hw_accel/profiling_baseline.md`

**输出**: 一份 `profiling_baseline.md`, 表格列各 stage 时间, 截图 asitop 在关键 phase 的 utilization。这是后续所有方案的 baseline。

#### 方向 1: ASR encoder 切 CoreML/ANE provider (低成本, 1-2h)

asr.py:73 注释 "Mac 上 ANE/CoreML 实测无加速" 是历史结论, 重测:

```python
# build_engine_config 改 onnx_provider="CoreMLExecutionProvider"
# 看 encoder elapsed 是否降, 是否走 ANE
```

**验证**:
- 单 task: encoder elapsed CPU vs CoreML 对比
- N=2 并发: 是否能让 CPU 释放给 sherpa diarize, GPU 能否早期介入
- 实测 RTF / wall time 对比

**注意**:
- onnxruntime 必须 ship CoreMLExecutionProvider (`pip show onnxruntime` 看 + 测 `ort.get_available_providers()`)
- ANE 对 op 支持有限, 可能 fallback 到 CPU (要看 ORT log)
- CoreML graph compile 第 1 次跑慢 (warmup), 测稳态需重复 3-5 次

**输出**: `spikes/qwen3_mac_hw_accel/coreml_asr_encoder.md`

#### 方向 2: sherpa-onnx 切 CoreML/Metal provider (中成本, 2-3h)

sherpa 的 `OfflineSpeakerDiarizationConfig` / `SpeakerEmbeddingExtractorConfig` 都有 `provider` 字段, 当前是 "cpu"。

**调研**:
- sherpa-onnx 文档/源码看支持哪些 provider (`coreml`? `cuda`? `metal`?)
- 测 `cfg.validate()` 在 provider="coreml" 时是否通过

**验证**:
- segmentation provider=coreml 单跑 elapsed 对比
- nemo-titanet embedding provider=coreml 单跑对比
- 长音频 cluster_merge per-segment embedding 总时间对比

**输出**: `spikes/qwen3_mac_hw_accel/sherpa_coreml.md`

#### 方向 3: num_threads 调优 + 错峰 (低成本, 30min, 但收益受限)

PoC 上 session 已建议但没测: 把 `FUNASR_QWEN3_NUM_THREADS=4`, 让 N=2 总 threads 8 ≈ cores, 减少争抢。

**验证**:
- 长音频 N=2 并发, num_threads=4 vs 8 vs 12 对比 wall + 单 task RTF
- 单 task num_threads=8 vs 4 单跑对比 (确认调小不让单 task 太慢)

**输出**: 对比表格写到 `spikes/qwen3_mac_hw_accel/num_threads_tuning.md`, 决定生产 default

#### 方向 4: ASR encoder/decoder 流水线 (高成本, 留 last)

vendor `src/core/vendor/qwen_asr_gguf/inference/asr.py` 看 encoder 是不是 batch 一次性跑完再 decode。如果是, 可以拆 chunk 让 encoder/decoder 流水化, GPU 早期介入。

**只在方向 1-3 都不够时再做**, 因为改 vendor code 风险高。

#### 方向 5: 多个 ANE 实例并发 (探索性)

ANE 是物理硬件, 看是否支持多 worker 同时 inference (CoreML 通常允许多个 model 实例并存)。

**输出**: `spikes/qwen3_mac_hw_accel/ane_concurrency.md`

### 度量基线 (PoC 必须复现)

固定 audio: `tmp_long_audio/eval_set/audio_1spk_real.m4a` (16min) + `tmp_long_audio/eval_set/audio_4spk.m4a` (44min)

固定脚本: 复制 `/tmp/qwen3_long_audio_real_concurrent.py` 到 `spikes/qwen3_mac_hw_accel/run_long_concurrent.py`

**Baseline (当前 main)**: N=2, num_threads=8, all CPU
- 1spk-16min: wall=421.3s, RTF=0.429
- 4spk-44min: wall=808.8s, RTF=0.302
- 总 wall: 808.9s
- 期间 GPU peak utilization (asitop): ?% (PoC 阶段补)
- 期间 CPU peak utilization (asitop): ?% (PoC 阶段补)

每个候选方案跑同样的 N=2 并发, 比较 wall + 单 task RTF + GPU peak + CPU peak。

### 工作方式约束 (与上阶段一致)

- 用中文回应
- 用 ctx_batch_execute / ctx_execute_file 处理大输出, Read 用于 Edit 前
- venv: `venv/bin/python`
- 设环境: `unset TMPDIR; export TMPDIR=/tmp; export DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin"`
- 长任务 (>1min): `Bash run_in_background: true`, 等通知
- 不要 ctx_execute 内 nohup (TMPDIR 污染坑)
- 测试: `venv/bin/python -m pytest`
- e2e: `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest`
- 不提交 audio/模型/tmp_long_audio 到 git
- 监控 Mac 硬件: `brew install asitop` 然后 `sudo asitop` (需 sudo)

### 已知 TODO (用户已拒收, 不要做)

- 不做工程化集成 (PoC 验证有效再单独开 PR)
- 不动 cluster_merge / short_guard 算法 (PoC v12 + PR4 已 ship)
- 不重写 vendor ASR engine 架构 (除非方向 4 必须, 且要单独评审)
- 不动主算法 / 默认配置, 只在 spikes/ 下出 PoC 报告

### PoC 输出物 (每方向一份)

每个方向一个 `spikes/qwen3_mac_hw_accel/<方向名>.md`, 包含:
- 改动了什么 (临时 patch 或 env override)
- 复现命令
- 实测对比表 (baseline vs 方案)
- 结论: 是否值得工程化 + 风险点

最终 1 份 `spikes/qwen3_mac_hw_accel/SUMMARY.md`:
- 各方向 ROI 排序
- 推荐工程化路线 (单选或组合)
- 下个工程化 PR 的范围建议

### 开始

1. 跑 `git status`, 确认在 `spike/qwen3-diarize-poc` 分支或新开 spike branch
2. `mkdir -p spikes/qwen3_mac_hw_accel` (如果不存在)
3. 复制 `/tmp/qwen3_long_audio_real_concurrent.py` 到 `spikes/qwen3_mac_hw_accel/run_long_concurrent.py` 作为复现基线
4. **先做方向 0 (profiling baseline)**, 没有这个 baseline 后续方案对比无意义
5. 装 asitop: `brew install --cask asitop` (如未装)
6. 跑一次 baseline, 截图 asitop, 写 `profiling_baseline.md`
7. 然后按 ROI 排序做方向 1 → 2 → 3, 每个出报告
8. 决定是否做方向 4-5 (按方向 1-3 的留白)

### PoC 完成的标准

- [ ] `profiling_baseline.md` 完成 (per-stage 时间表 + asitop 截图)
- [ ] 方向 1-3 各自完成 + 报告 (即使是 "无收益" 也要明确说明并附数据)
- [ ] `SUMMARY.md` 完成, 给出推荐工程化路线
- [ ] 所有 PoC 改动放 `spikes/qwen3_mac_hw_accel/`, 不动 `src/`
- [ ] 不破坏现有测试 (跑一遍 `venv/bin/python -m pytest tests/unit/` 确认 215/215)

### PoC 反模式 (不做)

- ❌ 直接改 `src/core/qwen3/asr.py` 的默认 provider (PoC 阶段只用 env override 或 monkey-patch)
- ❌ 不出报告就开始下一个方向
- ❌ 单点数据下结论 (每个方案至少 3 次跑取中位数)
- ❌ baseline 之外的对比 (一律对照 baseline 而不是其他方案)
- ❌ 改 vendor 代码不留 git diff 备份
