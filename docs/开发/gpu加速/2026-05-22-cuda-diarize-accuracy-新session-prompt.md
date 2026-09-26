# 新 session 上下文 — CUDA 环境 Qwen3 说话人区分准确度评测

> **本文档是开新 session 用的 priming prompt**. 新 session 启动时把这份内容一次性给 Claude, 让它快速进入上下文.

## 一句话目标

在 cuda dev box 上跑几个测试音频, **评测 Qwen3 + ort_cuda diarize backend 的说话人区分准确度**, 看 over-detect / under-detect / 后处理 pipeline 各层贡献, 落档成一份"CUDA 环境 diarize 准确度报告".

## 上下文 — 为什么要做

历史 sprint (`docs/开发/gpu加速/2026-05-22-ORT-CUDA-diarize-backend.md` + `2026-05-23-CUDA并发突破.md`) 把 ort_cuda backend 落地 + pool dispatch 接通, 但**评测重心一直是 wall RTF (吞吐), 没有系统评测 spk 准确度**:

- 已知历史问题: spk over-detect (`docs/开发/archive/spk-over-detect-归因调研结果.md`), Mac sherpa 路径已经用 `cluster_centroid_merge` + `short_segment_guard` + `silence_align` 治理过
- 但 cuda 路径用的是 `ort_cuda` backend (Python ORT wrap + scipy 复刻 FastClustering), **跟 Mac sherpa C++ 实现是两套代码**, 准确度需要单独验证
- `tests/integration/test_diarize_ort_parity.py` 是 parity test, 但只比较 segment 数量近似, 不评测真实场景准确度
- 现在缺一份"cuda 上 spk 准确度怎么样, 三层后处理各贡献多少"的报告

## 四个核心评测维度

| 维度 | 指标 | 怎么测 |
|---|---|---|
| 1. 说话人**数量** | 真实 N spk → 识别 M spk, 误差 |\|N-M\|| | 跟人工标注对比 (如果有 GT) 或定性看输出 |
| 2. 说话人**归属** | 每段归属正确率 | DER (Diarization Error Rate, 标准指标), 需要 GT |
| 3. **Over-detect** | 一人识别成多人 (历史痛点) | 看 cluster 数 + cluster size 分布 |
| 4. **Under-detect** | 多人识别成一人 | 听 minority speaker 段有没有归错 majority |

DER 需要 GT 才能算, 没有 GT 也能跑前 3 个维度的定性评测.

## 必读先验材料 (按顺序)

1. **`CLAUDE.md`** 整段 — 重点看 "ASR 引擎架构" / "Runtime + Diarize backend 抽象" / "Qwen3 后处理 pipeline". **必读**.
2. **`docs/开发/gpu加速/2026-05-22-ORT-CUDA-diarize-backend.md`** — ort_cuda backend 设计 + wall RTF 数据, 是本评测的上下文.
3. **`docs/开发/archive/spk-over-detect-归因调研结果.md`** — 历史 spk over-detect 问题归因 + 治理路径 (cluster_merge / short_guard / silence_align). 评测时要看这套治理在 cuda 上还有没有效果.
4. **`tests/integration/test_diarize_ort_parity.py`** — 现有 ort vs sherpa parity test 结构, 评测可以复用同款 fixture / mock 模式.
5. **`src/core/qwen3/diarize_ort.py`** — ort_cuda backend 实现 (pyannote-segmentation-3.0 + TitaNet + scipy 复刻 FastClustering). 不动它, 但要知道 cluster_threshold 等参数怎么影响输出.
6. **`src/core/qwen3_transcriber.py`** 中的后处理 pipeline:
   - `filter_spurious_speakers` (固定开)
   - `apply_cluster_centroid_merge` (config `cluster_merge_enabled`)
   - `apply_short_segment_guard` (config `short_segment_guard_enabled`)
   - `apply_silence_align_to_segments` (config `silence_align_enabled`)
7. **`scripts/_remote_diarize_parity_probe.py`** — 现成远端 diarize 评测脚本, 可参考改造.

## 三个变量的评测矩阵

```
变量 1: backend          ort_cuda / sherpa (远端 cuda dev box 都跑得起)
变量 2: 后处理 pipeline   default (全开) / 关 cluster_merge / 关 short_guard / 关 silence_align / 全关
变量 3: 音频              2-spk / 3+ spk / 长音频 / 短音频 / 含噪声
```

不需要全交叉跑 3×5×N 这么大, 重点跑 baseline + 单点 ablation.

## 推荐方案 (评测分 3 步)

### Step 1: 摸清可用音频资源 + 决定 GT 来源

```bash
# 本地 fixture
ls tests/fixtures/audio/

# 远端可能有更多 (历史 sprint 跑过的)
ssh <user>@<dev-host> 'ls ~/Dev/projects/funasr_spk_server/tests/fixtures/audio/ 2>/dev/null; \
                       ls ~/Dev/projects/funasr_spk_server/fixtures/ 2>/dev/null'
```

确认有几段 + 各自的 speaker count (人工听确认) + 是否有人工校对稿 (GT).

### Step 2: 设计评测脚本

写一个 `scripts/_remote_diarize_accuracy_eval.py`:
- 输入: audio_path, backend, pipeline_config (4 个开关)
- 跑 `run_diarization_dispatched` + 后处理 pipeline
- 输出: 检测到的 speaker count, 每个 speaker 总时长 + 占比, segment 列表, cluster size 分布
- 不写 DER (除非有 GT), 只输出结构化数据

```python
# 调用示例
result = eval_diarize(
    audio="podcast_2speakers_60s.wav",
    backend="ort_cuda",
    pipeline={"cluster_merge": True, "short_guard": True, "silence_align": True},
)
# 输出 {
#   "speaker_count": 2,
#   "speakers": [{"id": "Speaker1", "duration": 35.2, "share": 0.59}, ...],
#   "segments": [...],
#   "cluster_sizes": [320, 240, 12],  # 看是否有 over-detect 的小 cluster
# }
```

### Step 3: 跑评测 + 生成报告

每段音频跑 baseline + 单点 ablation, 整理表格:

```
podcast_2speakers_60s.wav (真实 2 人):
  ort_cuda + default        → speaker_count=2, durations=[35s,25s]  ✅
  ort_cuda + 关 cluster_merge  → speaker_count=3, [30s,20s,10s]      ❌ over-detect 复现
  ort_cuda + 关 short_guard   → speaker_count=2, [34s,26s], segs+15  ⚠️  jitter
  ort_cuda + 关 silence_align → speaker_count=2, [35s,25s], align↓  ⚠️
  sherpa + default          → speaker_count=2, [35.5s,24.5s]        ✅ parity
```

报告落档: `docs/开发/gpu加速/2026-05-22-cuda-diarize-accuracy.md`, 含:
- 评测设置 (硬件 + audio + GT)
- 每段音频每个配置组合的结果表
- 三层后处理各自的"贡献度" (开关前后的关键指标差)
- 跟 Mac sherpa 路径的跨平台对比
- 已知问题 / 不足 / 改进建议

## 开场拍板项 (新 session 开始时 AskUserQuestion 问用户)

1. **音频和 GT**: 现有的 60s 音频够不够? 要不要找更多 (尤其 3+ spk 多人音频)? 有没有人工校对稿能算 DER?
2. **范围**: 只评 cuda (重点) 还是 cuda + Mac sherpa 跨平台对比?
3. **后处理 ablation 深度**: 单层关闭 (4 组) 够了还是要两两组合 (6 组)?
4. **harness 化**: 一次性手工跑就行, 还是写自动化 eval harness 进 `tests/integration/`?

## 自主推进准则

跟前几次 session 一致.

### 什么时候 **不要** 中途询问
- ✅ 写评测脚本 / 跑 ssh 命令 / 整理报告表格, 直接做
- ✅ ssh 同步 rsync + LD_LIBRARY_PATH 这种基础设施, 直接做
- ✅ commit message 仿 git log
- ✅ 跑某个 ablation 看结果, 直接做不用问

### 什么时候 **必须** 停下来
- ❌ 评测脚本需要修改 `src/core/qwen3/diarize_ort.py` 这种 backend 代码 (本评测应该 read-only)
- ❌ 发现 over-detect 严重退化要做新的 治理 patch (这是下个 sprint 的事, 本评测只报告)
- ❌ GT 真实性存疑需要用户确认人工标注是否准确

## 不在范围内 (避免 scope creep)

- ❌ **改 ort_cuda backend 代码** — 评测路径 read-only, 发现问题写报告不动代码
- ❌ **加新 backend** — 不在评测范围
- ❌ **优化 RTF / wall** — 上次 sprint 已做, 本次只看准确度
- ❌ **改后处理算法** — cluster_merge / short_guard / silence_align 参数可调, 但算法本身不动
- ❌ **训练新模型** — 评测现有 segmentation/embedding 模型, 不训练
- ❌ **写 DER 计算** 如果没 GT — 跳过, 改用定性指标

## 报告输出

落档到 `docs/开发/gpu加速/2026-05-22-cuda-diarize-accuracy.md`. 结构:

```markdown
# CUDA 环境 Qwen3 Diarize 准确度评测报告

## 评测设置
- 硬件: cuda dev box (8 vCPU + RTX 3060)
- 音频: <list>
- Backend: ort_cuda (vs sherpa 对比)
- 后处理 pipeline: default + 4 组 ablation

## 主要发现
- <bullet points>

## 详细数据

### Audio 1: <name>
真实 N spk, GT 来源 <human/none>

| Backend | cluster_merge | short_guard | silence_align | spk count | durations | DER |
|---|---|---|---|---|---|---|
| ort_cuda | ✓ | ✓ | ✓ | 2 | [35,25] | N/A |
| ort_cuda | ✗ | ✓ | ✓ | 3 | [30,20,10] | N/A |
| ... |

### Audio 2: ...

## 跨平台对比 (cuda vs Mac sherpa)
...

## 后处理 pipeline 贡献度
...

## 已知问题 / 改进建议
...
```

## 落档 / commit 约定

- **独立维护, 不开 PR** (`feedback_no_pr_workflow` memory)
- **不主动 push**, git commit 后等用户指示
- 评测脚本 commit 进 `scripts/_remote_diarize_accuracy_eval.py`
- 报告 commit 到 `docs/开发/gpu加速/`
- commit message 风格仿现有 `git log --oneline -10`

## 起手第一步

```bash
# 1. 扫一遍可用音频
ls tests/fixtures/audio/
ssh <user>@<dev-host> 'ls ~/Dev/projects/funasr_spk_server/tests/fixtures/audio/ 2>/dev/null; \
                       ls ~/Dev/projects/funasr_spk_server/fixtures/ 2>/dev/null'

# 2. 看现有 diarize 评测脚本 (能不能复用)
cat scripts/_remote_diarize_parity_probe.py

# 3. 看 ort_cuda backend 接口 (评测脚本要调哪个函数)
grep -n "def run_diarization" src/core/qwen3/diarize_ort.py src/core/qwen3/diarize.py

# 4. AskUserQuestion 问用户 4 个拍板项 (音频/GT/范围/harness)

# 5. 写评测脚本 → ssh 同步 + LD_LIBRARY_PATH → 跑 baseline → 跑 ablation → 报告
```

## 远端 cuda 环境提醒

- ssh `<user>@<dev-host>`
- 工作目录: `~/Dev/projects/funasr_spk_server/`
- LD_LIBRARY_PATH 配置见 `scripts/_remote_*.sh` 头部, 或参考:
  ```bash
  NV=venv/lib/python3.12/site-packages/nvidia
  TRT=venv/lib/python3.12/site-packages/tensorrt_libs
  export LD_LIBRARY_PATH="${TRT}:${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/nvjitlink/lib:${NV}/cusparse/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cuda_cupti/lib"
  ```
- 一句 env 配置: `export FUNASR_PROFILE=cuda_dev` (见 CLAUDE.md "Config 体系 > 切换设备/切引擎操作手册")
- rsync 正确姿势 (path 已修正, 之前 plan doc 写错):
  ```bash
  rsync -av --exclude='__pycache__' src/ <user>@<dev-host>:~/Dev/projects/funasr_spk_server/src/
  rsync -av --exclude='__pycache__' scripts/ <user>@<dev-host>:~/Dev/projects/funasr_spk_server/scripts/
  ```

---

**Good luck. 半天到一天搞定评测 + 报告**.
