# 新 session 上下文 — Qwen3 CUDA 平台适配 + ORT 直 wrap diarize backend

> **本文档是开新 session 用的 priming prompt**. 新 session 启动时把这份内容一次性给 Claude, 让它快速进入上下文.

## 一句话目标

实现 **runtime platform detection + diarize backend 抽象**, 在 CUDA 平台上用 Python `onnxruntime` 直 wrap pyannote-seg + TitaNet 跑 GPU(替代当前 sherpa-onnx CPU), **把 5min+ 长音频 wall RTF 从 0.090 压到 ~0.04-0.05**. Mac 路径保持现状不变.

整个实现按 **严格 TDD 流程**: 红 → 绿 → commit, 每个最小单位都是一个独立 commit.

## 必读先验材料(顺序)

1. **`docs/开发/gpu加速/2026-05-21-3060-CUDA移植与优化.md`** — 上一个 sprint 的所有发现, 含 CUDA 移植、性能调优(num_threads=4 sweet spot)、TRT 不可行、sherpa-onnx GPU 跟 LLM 撞、以及 **ORT 直 wrap 验证通过** 的完整记录. **这是基础, 必读**.
2. **`CLAUDE.md`** — 项目工程约定, 特别注意:
   - 必须严格 TDD (`feedback_tdd_strict` 用户 memory 反复强调)
   - ASR 引擎架构现状(轻量 dispatch + 全局唯一引擎, **不要再引入 ABC 抽象**, 决策档案在 `docs/开发/archive/重构计划-ASR引擎抽象.md` — 已被 PR2-4 工程化验证过度设计)
   - Qwen3 后处理 pipeline 顺序(`filter_spurious_speakers → cluster_centroid_merge → merge_asr_chunks_and_diarize → short_segment_guard → silence_align`)
3. **`src/core/qwen3_transcriber.py`** — 关键看 `transcribe()` 主入口(L355-395 的 asyncio.gather), `_ensure_embedding_extractor_fn` (L42-60 sherpa 构造点)
4. **`src/core/qwen3/diarize.py`** — sherpa 当前包装层, 新 backend 要保持相同输出 schema(`List[DiarizeTurn]` 含 `start/end/speaker_id`)
5. **`scripts/_remote_ort_cuda_clash_check.py`** — 已验证 ORT CUDA + llama.cpp CUDA 共存的 PoC 脚本, 实现可直接参考其 ORT session 创建方式

## 环境约束

| 项 | 值 |
|---|---|
| dev 机器 | `ssh <user>@<dev-host>` (Tailscale), RTX 3060 12G, Ubuntu 24.04, Python 3.12, 4 vCPU(用户会改 8) |
| 工作目录 | `~/Dev/projects/funasr_spk_server` |
| venv | `venv/`(已装 onnxruntime-gpu 1.26 + sherpa-onnx 1.13.2+cuda12.cudnn9 + tensorrt-cu12 10.9 + 所有依赖) |
| 关键 env | `LD_LIBRARY_PATH` 必须包含 venv 内 nvidia/{cudnn,cublas,cufft,cuda_runtime,cuda_nvrtc,nvjitlink,cusparse,curand,cusolver,cuda_cupti}/lib (见 `scripts/_remote_run_provider.sh`) |
| 测试模型 | 60s / 300s / 1800s podcast fixture 已落地 `tests/fixtures/audio/podcast_2speakers_*.wav` |
| 模型权重 | `models/qwen3_diarize/sherpa/{pyannote-segmentation-3.0/model.onnx, nemo-titanet-small/embedding.onnx}` |

启动远端 venv + LD_LIBRARY_PATH 的 helper: `bash scripts/_remote_run_provider.sh cuda <label>`.

## 工程目标

### 目标 1: Runtime Platform Detection + Backend 抽象

新增 `src/core/runtime.py`, 集中表达"环境感知配置".

```python
# 设计草图(只是示意, 实际 API 由 TDD 测试驱动设计)
class RuntimeEnvironment(Protocol):
    name: str  # "mac_ane" / "cuda" / "cpu"
    
    def validate(self) -> None:
        """启动 sanity check, 缺关键 lib 直接 raise(替代当前 silent fallback)."""
    
    def recommend_diarize_backend(self) -> str:
        """返回 'sherpa' / 'ort_cuda' / ...; 由 config 字段 + env override 兜底."""
    
    def recommend_num_threads(self) -> int:
        """根据 CPU 核数算 sherpa num_threads (Mac default 4, Linux 4 vCPU → 2, 8+ vCPU → 4)."""

def detect_runtime() -> RuntimeEnvironment: ...
```

#### Acceptance criteria

- [ ] 单元测试覆盖 detection 三个分支(darwin → MacRuntime, linux + cuda libs → CudaRuntime, 其他 → CpuRuntime)
- [ ] validate() 检查关键 lib 缺失时 fail-fast, 跟 silent fallback 区分
- [ ] `recommend_num_threads()` 按 `os.cpu_count()` 给推荐值, 跟 4 vCPU/8+ vCPU 实测最优一致
- [ ] **零代码侵入现有 Mac 路径** — 现有 e2e test (FunASR + Qwen3) 全部通过, parity 不掉
- [ ] 启动日志增加一行: `runtime=mac_ane diarize_backend=sherpa num_threads=4` 给运维可观测性

### 目标 2: `OrtCudaDiarizeBackend` 工程化实现

替代 sherpa-onnx 的 `OfflineSpeakerDiarization`, 拆成 4 个组件用 Python `onnxruntime` 直跑:

1. **pyannote-seg sliding window**: 输入 (1, 1, 160000) 即 10s @ 16k, 输出 (1, 589, 7) speaker activity. 滑动窗步长 ~1s 覆盖完整音频, 加权融合
2. **TitaNet mel preprocessing**: numpy 实现 80-band log-mel (sample_rate=16000, win_length=400, hop_length=160, n_mels=80, normalize). NeMo 配置可从 `nemo-toolkit` 源码 cross-ref
3. **TitaNet ORT CUDA inference**: 吃 `(B, 80, T_mel)` 出 192-dim embedding
4. **FastClustering Python 实现**: 凝聚聚类 + cosine distance + threshold cut. sherpa 算法在 `https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/fast-clustering.cc` 可参考

#### Acceptance criteria

- [ ] 跟现有 sherpa diarize 行为 parity: 同一份 fixture 跑出来 **speaker 数完全一致**, **每个 segment 时间戳差异 ≤ 0.2s, IoU ≥ 95%**
- [ ] 60s podcast 端到端 wall < **3s**(vs 当前 sherpa CPU ~13s)
- [ ] 30min podcast 端到端 wall < **15s**(vs 当前 sherpa CPU ~100s)
- [ ] `FUNASR_QWEN3_DIARIZE_BACKEND=ort_cuda` env 切到新 backend; `FUNASR_QWEN3_DIARIZE_BACKEND=sherpa` 切回旧 backend(防回归)
- [ ] **保留所有现有后处理**(`filter_spurious_speakers / cluster_centroid_merge / short_segment_guard / silence_align`)不动 — 新 backend 只换 segment+embed+cluster 这一段, 后处理依赖不变
- [ ] 跟 LLM CUDA 共存稳定, 跑 e2e test 不 segfault(整个 sprint 的核心假设)

## 自主推进准则(非常重要 — 别中途停下来问)

**用户已经明确授权**: 整个 sprint 按 TDD 推, 测试通过就 commit, **持续推进直到全部任务完成**. 不要在每个小决策上停下来问用户.

### 什么时候 **不要** 中途询问
- ✅ 测试绿了 → 直接 commit, 不要问"要不要 commit"
- ✅ 完成一个 acceptance criterion → 直接进下一个, 不要问"接下来做什么"
- ✅ 已知陷阱里有的事(LD_LIBRARY_PATH / num_threads / mel preprocessing 等)→ 按 prompt 里写的执行, 不要问
- ✅ commit message 风格 → 仿照 `git log --oneline -20` 已有风格, 不要问要怎么写
- ✅ 测试设计 / fixture 命名 / 工程小决策 → 自己拍板, 不要问

### 什么时候 **必须** 停下来
- ❌ 卡在底层不可调和的问题(比如 ORT CUDA 又撞 LLM、模型权重坏掉 download 不下来), 多次尝试无解
- ❌ 准确度 parity 测试反复失败, 已经怀疑 sherpa baseline 本身不稳定
- ❌ 用户的工程意图明显有冲突(比如某 acceptance criterion 跟 CLAUDE.md 约定冲突)
- ❌ 发现需要做 >1 天工程量的预料外重构, 该说服用户调整 scope

简单原则: **能往下推就往下推, 不能往下推就讲清楚阻塞 + 选项**.

## TDD 流程要求(严格执行, 用户原话: "先红再绿再 commit")

### 红 → 绿 → commit 是最小单位, 测试一通过立刻 commit

**不要积累多个改动一次性提交**. 每个改动:
1. 先写测试, 跑 → 红(失败 + 写下你期望什么样的 API)
2. 写最小代码让测试绿 → 绿
3. **立刻 `git add` + `git commit`** — 包含测试 + 最小 impl, 不要继续往下写下一个 feature 再 commit
4. 继续下一个 红 → 绿 → commit 循环

一次 sprint 期望产生 10+ 个细粒度 commit, 不是 2-3 个大 commit. 单 commit 改动 < 200 行更健康.

参考项目里之前的 commit 风格(`git log --oneline -20`):
- `feat(qwen3): cluster_merge dominant 模式吃相似 minor cluster (新 dominant_minor_threshold=0.5)`
- `perf(qwen3): 3060 CUDA 移植 smoke test + 性能调优, wall RTF 0.225→0.090`

### 推荐的 commit 拆分(粗粒度)

**目标 1(runtime detect)**:
1. `test+feat(runtime): detect_runtime() 三平台分支 + 单测`
2. `test+feat(runtime): validate() fail-fast 检查 cudnn / num_threads`
3. `test+feat(runtime): recommend_*() 配置推荐 + cpu_count 适配`
4. `feat(qwen3): main entry point 集成 runtime + 日志增强`

**目标 2(ORT diarize backend)**:
5. `test+feat(diarize/ort): pyannote-seg sliding window 单测 + 实现`
6. `test+feat(diarize/ort): TitaNet mel preprocessing 跟 NeMo 比对 parity 单测`
7. `test+feat(diarize/ort): TitaNet ORT CUDA inference + cluster Python 实现`
8. `test+feat(diarize/ort): OrtCudaDiarizeBackend 包装 + dispatch 集成`
9. `test(integration): ort_cuda vs sherpa parity test (≤ 0.2s 时间戳容差)`
10. `perf(diarize/ort): 30min 长音频 wall RTF 验证(目标 < 15s)`

每个 commit 都要附 `via [HAPI](https://hapi.run)` + `Co-Authored-By: HAPI <noreply@hapi.run>` + Claude attribution(项目惯例, 见 user CLAUDE.md).

### 测试文件位置

- 单元测试: `tests/unit/test_runtime.py`, `tests/unit/test_diarize_ort_backend.py`, `tests/unit/test_titanet_mel.py`, `tests/unit/test_fast_clustering.py`
- 集成测试: `tests/integration/test_diarize_ort_parity.py`(需 `FUNASR_RUN_INTEGRATION=1` 才跑)
- mel preprocessing 的 ground truth: 用 NeMo 跑一次 dump numpy array 落到 `tests/fixtures/golden/titanet_mel_60s_podcast.npy`

## 已知陷阱(避免重复踩)

1. **silent fallback 是默认行为**: ORT 加载 CUDA EP 失败会 silent 回 CPU 不抛异常. `validate()` 里必须显式 `assert "CUDAExecutionProvider" in sess.get_providers()`, 不能信赖 ORT 报错.
2. **sherpa thread 必须 ≤ vCPU/2**: 4 vCPU 上 sherpa num_threads=8 会让 ASR 的 numpy.fft 第一次跑 40s mel 等 9.5s (实测). `runtime.recommend_num_threads()` 必须考虑这个.
3. **LD_LIBRARY_PATH 是隐性依赖**: venv 内 nvidia/cudnn/lib 等不在 LD path 默认搜索范围, 必须 export. PM2 ecosystem 启动也要带, 不能假定 shell env 继承.
4. **TitaNet ONNX 期望 mel features (B, 80, T) 不是 raw audio**: 这是 NeMo 跟 sherpa 的关键差异. sherpa 把 mel + embedding 黑盒打包, Python 直 wrap 必须自己做 mel.
5. **CUDA context 谁先谁后**: 已实测 llm_first / ort_first / interleaved 三种顺序都不撞(`scripts/_remote_ort_cuda_clash_check.py`). 但工程上推荐 **llm_first**(贴近现有启动顺序, 减少行为变化).
6. **不要改 default config**: Mac 上 num_threads=4 是 PoC 验证最优. 改 Pydantic default 会影响 Mac baseline. 让 `runtime.recommend_*()` 在启动时 inject 到 config 更安全(行为变化只在 Linux 体现).
7. **e2e parity 是硬指标**: 项目花了大量精力 tune sherpa 的 over-detect 修复(见 `docs/开发/archive/spk-over-detect-归因调研结果.md`). 新 backend **必须 IoU ≥ 95%**, 否则等于推倒重做调优. 任何准确度回归都要 hold commit, 调到 OK 再 commit.

## 部署与 commit 约定(再次强调)

- **独立维护, 不开 PR**(`feedback_no_pr_workflow`). 不要说"PR 标题"/"等待 review", 直接讲 commit/push.
- **不要主动 push 到 remote** 除非用户明确说. `git commit` OK, `git push` 等用户指示.
- Mac 是生产, Linux 是 dev. 改动要 Mac 路径零回归.
- 部署不走 docker(`project_funasr_no_docker`).

## 远端机器使用 tips

```bash
# 同步本地改动到远端(dev 用)
rsync -av --exclude=venv --exclude=models --exclude=temp --exclude='__pycache__' \
  src/ scripts/ tests/ docs/ <user>@<dev-host>:~/Dev/projects/funasr_spk_server/

# 远端跑测试
ssh <user>@<dev-host> 'bash -lc "cd ~/Dev/projects/funasr_spk_server && \
  source venv/bin/activate && \
  FUNASR_QWEN3_NUM_THREADS=4 FUNASR_RUN_INTEGRATION=1 \
  bash scripts/_remote_run_provider.sh ort_cuda new-backend-test"'

# 远端 background 长跑用 Bash run_in_background=true, 不要本地 poll
```

## 起手第一个 commit + 整体推进节奏

### 起手第一个 commit

**`test(runtime): 添加 detect_runtime() 红灯单测`**

```python
# tests/unit/test_runtime.py
import sys
import pytest
from unittest.mock import patch

def test_detect_runtime_returns_mac_on_darwin():
    with patch.object(sys, "platform", "darwin"):
        from src.core.runtime import detect_runtime
        runtime = detect_runtime()
        assert runtime.name == "mac_ane"

def test_detect_runtime_returns_cuda_when_libs_present():
    # ... 
```

跑 `pytest tests/unit/test_runtime.py` → 红(`src.core.runtime` 模块不存在). 然后写最小实现让它绿. 然后 commit. 然后下一个红灯.

### 整体推进节奏

按 "推荐的 commit 拆分" 那节的 10 个 commit 串行推, **一鼓作气推到全部 acceptance criteria 都过**, 不要中途等用户确认每个步骤.

每个 commit 完成后:
- ✅ 测试在远端 CUDA 机器上跑过(`bash scripts/_remote_run_provider.sh ...`)
- ✅ Mac 路径的 parity test 仍然过(可在本地 venv 跑, 或者远端模拟)
- ✅ commit message 清晰

**只在以下情况向用户汇报**:
- 全部 10 个 commit 都完成, 准备总结成果
- 卡在不可调和的问题, 多次尝试无解
- 发现 scope 严重偏离原 prompt(比如要做重构远超 1.5 天)

否则就是 — **写测试 → 跑测试 → 改代码 → 测试绿 → commit → 下一个**.

最后一个 commit 落档后, 给用户 summary:
- 实际 wall RTF 数据(60s / 5min / 30min)
- 跟 sherpa CPU baseline 的 parity 比对
- 总 commit 数和改动行数
- 已知遗留问题(if any)

---

**Good luck. 整个 sprint 完成后, 项目应该是 — Linux + CUDA 下端到端 wall RTF < 0.05, Mac 路径零变化, 一个清晰的 runtime + backend 抽象, 全套 TDD 单测覆盖. 一次性推到底, 不要在中间停.**
