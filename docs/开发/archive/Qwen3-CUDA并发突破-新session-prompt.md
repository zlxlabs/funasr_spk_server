# 新 session 上下文 — RTX 3060 单卡支持 ort_cuda + pool=2 并发

> **本文档是开新 session 用的 priming prompt**. 新 session 启动时把这份内容一次性给 Claude, 让它快速进入上下文.

## 一句话目标

在 dev box (8 vCPU + RTX 3060 12G + Ubuntu 24.04) 上让 Qwen3 引擎支持 **ort_cuda backend + pool_size=2 并发**, **2 个 1800s 任务同时跑且单任务 RTF < 0.1**. 完成时间: 明天早上 (用户原话). 探索完全自主, 只关心最终结果.

## 当前已知的拦路虎 (核心问题)

刚结束的 sprint (`docs/开发/gpu加速/2026-05-22-ORT-CUDA-diarize-backend.md`) 实测确认:
**2 进程并发 ort_cuda backend 会 CUDNN cross-process race 撞死 worker 2**.

错误栈位:
```
进程 A 跑到 ORT CUDA cudnnRNNForward (TitaNet LSTM, cudnn_rnn_base.cc:335)
  → CUDNN_STATUS_EXECUTION_FAILED 5000
  → CUDA failure 700: illegal memory access
  → 进程 B 的 llama.cpp ggml_backend_cuda_synchronize abort
(单 GPU + 多进程同时跑 cudnn RNN + LLM CUDA context, cudnn handle 跨进程 race)
```

**这不是 ORT 算法 bug**, 是 nvidia driver / cuDNN 跨进程内核共享的内存安全问题. 单进程内 `asyncio.gather (ASR + diarize)` 是 OK 的 (sprint 已验证). 多进程并发 ort_cuda 不行.

## 必读先验材料 (按顺序)

1. **`docs/开发/gpu加速/2026-05-22-ORT-CUDA-diarize-backend.md`** — 上一个 sprint 全成果, 含 perf 数据 / 资源占用 / 并发撞 CUDNN 实测. **这是基础, 必读**.
2. **`docs/开发/gpu加速/2026-05-21-3060-CUDA移植与优化.md`** — sherpa-onnx CUDA build 撞 llama.cpp 的历史, 跟现在的 CUDNN race 同类原因.
3. **`scripts/_remote_concurrent_probe.sh` + `scripts/_remote_concurrent_worker.py`** — 上次撞死的复现脚本, 拿来对照新方案是否解决.
4. **`scripts/_remote_resource_probe.sh`** — 单任务资源采样脚本, 拿来对照并发优化前后 GPU/CPU 占用.
5. **`CLAUDE.md`** — 项目 ASR 架构 + Runtime + Diarize backend 抽象现状.
6. **`src/core/qwen3_pool_transcriber.py` + `src/core/qwen3_worker_process.py`** — 现有 worker pool 派发实现 (file-based queue), 改并发架构时基础.

## 环境约束 (跟上一个 sprint 一致)

| 项 | 值 |
|---|---|
| dev 机器 | `ssh <user>@<dev-host>` (Tailscale), RTX 3060 12G, Ubuntu 24.04, **8 vCPU** |
| 工作目录 | `~/Dev/projects/funasr_spk_server` (远端) |
| venv | `venv/` (已装 onnxruntime-gpu 1.26 + sherpa-onnx 1.13.2+cuda12.cudnn9 + tensorrt-cu12 10.9) |
| LD_LIBRARY_PATH | 见 `scripts/_remote_run_provider.sh` 头部 |
| 测试 fixture | `tests/fixtures/audio/podcast_2speakers_{60s,300s,1800s}.wav` (远端有, mac dev 也有 60s) |
| 模型 | `models/qwen3_diarize/sherpa/{pyannote-segmentation-3.0/model.onnx, nemo-titanet-small/embedding.onnx}` 远端齐 |
| 当前 baseline (单任务) | 1800s ort_cuda wall 83.5s, **RTF 0.046** |
| 当前 baseline (单任务资源) | GPU mem peak 2.77 GB, GPU util peak 100%, Python CPU peak 393%, RSS peak 1.76 GB |

## 探索方向 (顺序仅建议, 自主调整)

### 候选 1: CUDA MPS (Multi-Process Service) — 首推
nvidia 官方 daemon, 专为单 GPU 多进程共享设计, **应用层 transparent** (启动 `nvidia-cuda-mps-control -d` 即可). 解决 cudnn handle 跨进程 race 的标准方案. 远端 Ubuntu 24.04 系统级 daemon, 不需要改 application code, 改完 `pgrep mps` 看 daemon 起没起即可验证.

验证步骤:
1. `nvidia-cuda-mps-control -d` 起 daemon
2. 跑 `scripts/_remote_concurrent_probe.sh` 看 2 worker 是否还撞
3. 若 OK, 测 2 并发 e2e wall + RTF, 确认 < 0.1

风险: MPS 在某些 driver 版本下对 cudnn 加速效果有限或不支持 (需查 driver 570 + cuda 12.8 是否支持).

### 候选 2: NVIDIA Triton Inference Server
把 pyannote-seg + TitaNet ORT session 跑成一个 daemon service, 多 worker 通过 grpc/http 调. 单 daemon 单 GPU context, 多 client 走 IPC. 工程量大但生产级.

GitHub 参考:
- https://github.com/triton-inference-server/server (官方)
- https://github.com/k2-fsa/sherpa-onnx/blob/master/triton (sherpa-onnx Triton 部署示例 — 可参考其 onnx 模型转 Triton model_repository 配方)

风险: 部署 + 模型转换 + grpc 客户端改造工程量大, 一天内做完 ASR + diarize 全套 service 风险高. **如果 MPS 行就别走这条**.

### 候选 3: 单进程 task queue + batched inference (架构改造)
把 `qwen3_pool_transcriber` 从"多进程 pool"改成"单进程 task queue + ORT batched". 同一进程内不同 task 共享 ORT session, ORT 自己跑 batch=2 的 LSTM (TitaNet) + LLM. 完全绕开多进程 cudnn race.

GitHub 参考:
- vLLM 的 continuous batching: https://github.com/vllm-project/vllm
- llama.cpp 的 batched API (`llama_batch_init`): 项目 vendor 自带 llama.cpp, 看看能否复用
- pyannote-audio 的 batched inference: https://github.com/pyannote/pyannote-audio

风险: 改动整个 worker pool 架构, 一天内做完风险大. **fallback 方案**.

### 候选 4: TensorRT-LLM
nvidia 官方 multi-request batching 服务, 替换 llama.cpp. 但 sprint 里上次确认 TRT EP 不支持 Qwen3-ASR encoder (`docs/开发/gpu加速/2026-05-21-3060-CUDA移植与优化.md` 已记). TRT-LLM 仍是单独 LLM serving, 跟 ASR encoder ORT 共存还是有 cuda context 跨进程问题. **风险高, 不推荐先走**.

### 候选 5: 切回 sherpa backend + pool=2 (退而求其次)
sherpa CPU backend 跟 LLM CUDA 共存 OK (production 在用), pool=2 跑 8 vCPU 上各 num_threads=4 也 work. 但单任务 RTF 从 ort_cuda 的 0.047 退到 sherpa 的 0.083. **如果其他方案都失败, 这是底线**.

## 自主推进准则 (跟上次一致)

**用户已经明确授权**: 一鼓作气推到底, **过程中可以任意探索**, 只关心最终结果. 不要在每个小决策上停下来问.

### 什么时候 **不要** 中途询问
- ✅ 跑 ssh 远端命令拿数据, 直接跑
- ✅ 测试新方案 (MPS daemon / Triton / batched), 直接试
- ✅ 选哪个候选, 自己拍板, 不行换下一个
- ✅ commit message 风格仿 `git log --oneline -20` 现有风格
- ✅ 阅读 GitHub 仓库, 直接 git clone 或 webfetch

### 什么时候 **必须** 停下来
- ❌ 所有 5 个候选方案都试过失败 (此时给用户清晰的失败原因 + alternatives)
- ❌ 远端机器宕机 / 模型权重坏
- ❌ 发现需要 buy 一张新 GPU 才能解决 (硬件预算决策超出代码范围)

简单原则: **能往下推就往下推, 不能往下推就讲清楚阻塞 + 选项**.

## TDD 流程 (严格执行, 用户原话: "先红再绿再 commit")

跟上一个 sprint 一致. 每个改动单元都是 "红 → 绿 → commit":
1. 写测试 / 写 reproducer 脚本 → 跑 → 红 (确认问题或确认假设)
2. 改代码 / 配置环境 → 跑 → 绿 (问题解决)
3. **立刻 commit** — 包含测试 + 最小 impl

推荐的粗粒度 commit 拆分 (按候选 1 走的话):

1. `chore(env): 启 nvidia-cuda-mps daemon + 验证 pgrep mps` (运维, 不入仓但落档 README)
2. `test(concurrent): 复跑 _remote_concurrent_probe.sh 在 MPS daemon 下不撞`
3. `perf(concurrent): 2 并发 1800s ort_cuda + MPS, wall + RTF + 资源数据落档`
4. `docs(diarize/ort): 落档 MPS 方案 + 并发吞吐 / 资源对比表`

如果走候选 3 (架构改造):
1. `test+feat(pool): 单进程 task queue 取代多进程 pool`
2. `test+feat(diarize/ort): batched TitaNet embedding`
3. `test+feat(asr): 单 ORT session 复用 (跨 task)`
4. `perf(concurrent): 2 并发实测落档`

## Acceptance Criteria

| # | 指标 | 目标 | 怎么验 |
|---|---|---|---|
| 1 | 2 个 1800s 任务同时启动 (file barrier 同步) | 都跑完不撞 | `bash scripts/_remote_concurrent_probe.sh`, 看 WORKER=1/2 都出 wall 行 |
| 2 | 每任务 RTF | **< 0.1** | log 里 RTF=... 字段, 两个 worker 都 < 0.1 |
| 3 | 总 wall (两个完成) | < 160s (vs 单任务 84s) | `/tmp/c_total_wall` |
| 4 | GPU 显存 peak | < 10 GB (留 2GB 系统) | nvidia-smi 采样 |
| 5 | Mac 路径零变化 | 既有 unit + integration 全绿 | `venv/bin/python -m pytest tests/unit/` 在 mac 上 (FUNASR_RUN_INTEGRATION 不开) |
| 6 | 落档 | `docs/开发/gpu加速/2026-05-23-CUDA并发突破.md` | 含方案、数据、对比、known caveat |

## 不在范围内 (避免 scope creep)

- ❌ 把 cluster_centroid_merge 60s over-merge 修了 (是另一个 sprint 范围)
- ❌ 把 ASR encoder 改 TRT-LLM (太大, 上次 sprint 已确认 TRT 不支持)
- ❌ 改 Mac 路径行为 (production 不动)
- ❌ pool_size=3 / >2 (上限确认 2 即可)

## 部署 / commit 约定 (再次强调)

- **独立维护, 不开 PR** (`feedback_no_pr_workflow` 用户 memory)
- **不要主动 push** 除非用户明确说. `git commit` OK, `git push` 等用户指示.
- Mac 是 production, Linux 是 dev. 改动 Mac 路径要零回归.
- 部署不走 docker (`project_funasr_no_docker` 用户 memory).

## 远端机器使用 tips

```bash
# 同步本地改动到远端
rsync -av --exclude=venv --exclude=models --exclude=temp --exclude='__pycache__' \
  src/ scripts/ tests/ docs/ <user>@<dev-host>:~/Dev/projects/funasr_spk_server/

# 远端跑命令 (LD_LIBRARY_PATH 必须 export, 不然 ORT silent fallback CPU)
ssh <user>@<dev-host> 'bash -lc "cd ~/Dev/projects/funasr_spk_server && \
  source venv/bin/activate && \
  bash scripts/_remote_concurrent_probe.sh"'

# 远端长跑用 nohup + 输出文件, SSH pipe 会 buffer stdout
ssh <user>@<dev-host> 'cd ~/Dev/projects/funasr_spk_server && \
  nohup bash scripts/_remote_xxx.sh > /tmp/xxx.log 2>&1 &'
# 然后从本地 ssh tail /tmp/xxx.log 看进度
```

## 起手第一个 commit + 整体节奏

### 起手 (候选 1 MPS 路线)

```bash
# 远端确认 MPS daemon 二进制存在 + driver 支持
ssh <user>@<dev-host> 'which nvidia-cuda-mps-control && nvidia-cuda-mps-control -d && pgrep -f mps'
```

如果 daemon 起得来:
1. 跑现有 `_remote_concurrent_probe.sh` 看 2 worker 还撞不撞
2. 撞 → MPS 没解决问题 → 跳候选 3 单进程架构改造
3. 不撞 → 测 RTF / 资源, 落档 commit

### 整体节奏

按 "推荐 commit 拆分" 串行推, 一鼓作气推到 acceptance criteria 全过. 最末一个 commit 落档后给用户 summary:
- 方案名 + 改了什么 (代码 / 配置 / 部署 / 文档)
- 实测 RTF 数据 (60s/300s/1800s × 1 并发 / 2 并发)
- 资源占用 (GPU / CPU / 内存)
- 总 commit 数
- 已知 caveat (如果 MPS 在某些条件下不工作, 写明)

**只在以下情况向用户汇报**:
- 全部 acceptance criteria 通过, 准备明早交付
- 5 个候选方案全失败, 需要重新选 scope
- 发现 scope 严重偏离 (比如要买新 GPU)

否则就是 — **跑测试 → 改方案 → 测试通过 → commit → 下一个**.

---

**Good luck. 明天早上前给一个明确可行方案 (or 明确不可行的根因). RTF < 0.1 with concurrency=2 是硬指标.**
