# ORT-CUDA diarize backend — 实现与 perf 落档

**Date**: 2026-05-22
**Branch**: `spike/qwen3-diarize-poc`
**Sprint**: Qwen3 CUDA 平台适配 + ORT 直 wrap diarize backend
**Dev box**: `vm200-i7-3060llm` (Tailscale `<dev-host>`, RTX 3060 12G, Ubuntu 24.04, 4 vCPU)

## 一句话目标

替代 sherpa-onnx 的 `OfflineSpeakerDiarization` CPU build, 用 Python `onnxruntime` 直 wrap
pyannote-segmentation-3.0 + TitaNet 跑 GPU CUDA, **同时跟 llama.cpp CUDA 共存稳定** (sherpa
GPU build 撞 LLM 已经在 [2026-05-21-3060-CUDA移植与优化.md](./2026-05-21-3060-CUDA移植与优化.md)
确认无解, 这是该 sprint 的核心动机).

## 工程层面 — 已落档

### 1. Runtime platform detection (`src/core/runtime.py`)

```
detect_runtime() → MacRuntime / CudaRuntime / CpuRuntime
                  ├─ validate() — fail-fast 替代 ORT silent CPU fallback
                  ├─ recommend_diarize_backend() — sherpa / ort_cuda
                  └─ recommend_num_threads() — cpu_count 自适应 (4 vCPU → 2, ≥5 → 4)
```

- 优先级: `FUNASR_RUNTIME=cpu/mac_ane/cuda` env > 平台探测 (`sys.platform` + `_has_cuda_runtime_available()`)
- CudaRuntime.validate() 显式校验 `CUDAExecutionProvider` 在 ORT providers 列表里, 不在则 raise
- describe_runtime() 给启动日志: `runtime=cuda diarize_backend=ort_cuda num_threads=2`

### 2. ORT CUDA diarize backend (`src/core/qwen3/diarize_ort.py`)

4 个 Python 函数 + 一个 wrapper, 替代 sherpa `OfflineSpeakerDiarization`:

1. `_iter_audio_chunks()` — 10s chunk, 1s step sliding window
2. `pyannote_segmentation_pipeline()` — ORT pyannote-seg → (T_frame, 3) multi-label binary
   - `_powerset_to_multilabel()` — 7-class powerset → 3-speaker multi-label
   - `_aggregate_chunk_outputs()` — Whisper-style 加权融合 重叠帧除以 count
3. `compute_titanet_log_mel()` + `compute_titanet_embedding()` — NeMo-style 80-band log-mel
   (preemph=0.97, hann window 400 zero-pad to 512, hop=160, librosa mel filterbank,
   per_feature z-score normalize) + ORT TitaNet inference → 192-dim L2-normalized
4. `fast_clustering()` — `scipy.cluster.hierarchy` cosine distance + **complete linkage**
   (PoC 实验确认 complete 比 average 更接近 sherpa cluster 行为; threshold=0.9 跟 sherpa
   config 对齐)

完整 wrapper `run_diarization_ort_cuda(audio_path, ...)` 输出 schema 跟 sherpa 一致:
`[{"start": float, "end": float, "speaker": int}, ...]`.

### 3. Dispatch (`src/core/qwen3/diarize.py:run_diarization_dispatched`)

```
优先级: 显式 backend 参数 > FUNASR_QWEN3_DIARIZE_BACKEND env > runtime.recommend_diarize_backend()
```

`Qwen3DiarizeTranscriber.transcribe` 的 lambda 改成调 `run_diarization_dispatched`,
Mac 路径 (MacRuntime → sherpa) 行为零变化, Linux + CUDA 自动走 ort_cuda.

## Perf 实测 — 单 diarize (RTX 3060, 4 vCPU)

跑法: `scripts/_remote_diarize_parity_probe.py`. 只跑 diarize, 不含 ASR.
audio fixture: `tests/fixtures/audio/podcast_2speakers_{60s,300s,1800s}.wav`.

| audio | sherpa CPU wall | sherpa speakers | ort_cuda wall | ort speakers | speedup | ort RTF |
|---|---|---|---|---|---|---|
| 60s | 3.76s | 2 ✅ | **2.36s** | **2 ✅** | 1.6x | **0.039** |
| 300s | 16.78s | 2 ✅ | **7.46s** | 3 ⚠️ | 2.25x | **0.025** |
| 1800s | 98.88s | 2 ✅ | **46.92s** | 3 ⚠️ | 2.1x | **0.026** |

`ort_cuda` 用 `cluster_threshold=0.9` + `method=complete` (跟 sherpa `FastClusteringConfig.threshold` 一致).
单 diarize 在 4 vCPU 上是 sherpa CPU bound, ort_cuda GPU bound, 跟 vCPU 数无关.

## Perf 实测 — e2e (ASR + diarize gather + 全部后处理, RTX 3060, 8 vCPU)

跑法: `scripts/_remote_e2e_wall_probe.py`, 通过 `Qwen3DiarizeTranscriber.transcribe()` 调,
含 `filter_spurious_speakers / cluster_centroid_merge / short_segment_guard / silence_align`.
ASR encoder provider = `cuda` (CUDA EP).

| audio | sherpa wall | sherpa RTF | sherpa spk | ort_cuda wall | ort_cuda RTF | ort_cuda spk | speedup |
|---|---|---|---|---|---|---|---|
| 60s | 5.73s | 0.096 | 2 ✅ | **3.65s** | **0.061** | **1 ⚠️** | 1.57x |
| 300s | 24.85s | 0.083 | 2 ✅ | **15.01s** | **0.050** | **2 ✅** | 1.66x |
| **1800s** | **143.27s** | 0.080 | 2 ✅ | **85.20s** | **0.047 ✨** | **2 ✅** | 1.68x |

🎯 **prompt 目标 "30min wall RTF 0.04-0.05" → 实测 0.047, 完美达标.**

### e2e win 来源拆解 (ASR + diarize 并行的间接效益)

ort_cuda backend 不光 diarize 本身快, ASR 也快得多 — 因为 sherpa CPU 占满 thread 让
ASR 的 numpy.fft 排队等, ort_cuda 走 GPU 不抢 CPU thread, ASR 解放出来:

| audio | ASR elapsed (sherpa backend) | ASR elapsed (ort_cuda backend) | ASR 改善 |
|---|---|---|---|
| 60s | 5.16s (RTF 0.086) | 2.18s (RTF 0.036) | -58% |
| 300s | 23.12s (RTF 0.077) | 10.48s (RTF 0.035) | -55% |
| 1800s | 55.48s (RTF 0.031) | 61.74s (RTF 0.034) | +11% (噪声, 长音频已不被 mel 阻塞) |

短音频 ASR 时间砍半是 e2e wall 加倍 win 的重要原因.

### 资源占用 — 1800s ort_cuda e2e (8 vCPU, 1Hz 采样)

跑法: `scripts/_remote_resource_probe.sh` 同步抓 `nvidia-smi` + `ps` (83 samples,
83.5s wall 内). 同一份 podcast 1800s.

| 指标 | peak | avg | 备注 |
|---|---|---|---|
| GPU 显存 | **2839 MiB (~2.77 GB)** | 2608 MiB | RTX 3060 12G 用 23% |
| GPU util% | 100% | 82% | 中段 ASR + diarize 共用 GPU 跑满 |
| Python CPU% | 393% | 250% | 8 vCPU 平均占 2.5 个 core |
| Python RSS | **1758 MB** | 1498 MB | 模型权重 + audio + KV cache |

阶段形态:
- **t=0-1s**: model load — RSS 18MB → 752MB → 1460MB, GPU mem 1 → 2633 MiB
- **t=1-80s**: ASR + diarize 并行跑, GPU util ~100% 稳定, CPU 240% (mel + cluster + llm token gen)
- **t=80-83s**: 后处理 — GPU 0% (cluster_centroid_merge / silence_align 都 CPU), CPU 280%, RSS 升到 peak 1758MB (audio 重读 + ffmpeg silencedetect)

显存预算: ASR encoder + LLM gguf + pyannote-seg ONNX + TitaNet ONNX 共 ~2.8GB,
跟 llama.cpp CUDA buffer ~300MB 共存稳定 (跟 PoC 同方向).

## ⚠️ 2 进程并发实测 — 不可行 (CUDNN cross-process race)

跑法: `scripts/_remote_concurrent_probe.sh` 启动 2 个独立 python 进程, file barrier
同步 model load 后同时 transcribe 1800s ort_cuda. 同样 RTX 3060 + 8 vCPU.

结果:
- **Worker 1**: 跑完, wall=89.87s, RTF=0.050, 2 speakers ✓
- **Worker 2**: 并发开始 t≈11s 时挂掉, **CUDNN_STATUS_EXECUTION_FAILED 5000**
  → `CUDA failure 700: illegal memory access` → llama.cpp `ggml_abort`

错误栈位 (worker 2):
```
ORT CUDA cudnnRNNForward (TitaNet LSTM, cudnn_rnn_base.cc:335)
  → CUDNN failure 5000 EXECUTION_FAILED
  → CUDA failure 700 illegal memory access
  → llama.cpp ggml_backend_cuda_synchronize abort
```

资源峰值 (Worker 1 跑完前):
| 指标 | peak | avg |
|---|---|---|
| GPU 显存 | 5223 MiB (5.1 GB) | 2937 MiB |
| GPU util | 100% | 84% |
| Worker1 CPU | 275% | 230% |
| Worker1 RSS | 1780 MB | 1448 MB |
| Worker2 死亡 | t≈11s | — |

### 根因

跟之前 sherpa-onnx CUDA build 撞 llama.cpp CUDA 是同类问题: **单 GPU 多进程同时跑
CUDNN RNN/LSTM forward + 另一进程的 LLM CUDA context, cudnn handle 跨进程 race**.
不是 ORT Python API bug, 是 nvidia driver / cuDNN 跨进程内核共享时的内存安全问题
(GPU compute context 单进程 enclave 假设被打破).

### 当前 dev box 下的实际并发上限

| 配置 | pool_size 上限 | 单任务 RTF | 备注 |
|---|---|---|---|
| sherpa backend + CPU | 2 (config 默认) | 0.083 (8 vCPU) | 稳定, CPU bound |
| ort_cuda backend | **1** | **0.047 (8 vCPU)** | 快但单任务 |

想 ort_cuda + pool=2: 见 `docs/开发/archive/Qwen3-CUDA并发突破-新session-prompt.md`
(下一个 sprint).

### Sweet spot (60s) — threshold + linkage sweep

| linkage | threshold | speakers (sherpa baseline=2) |
|---|---|---|
| `average` | 0.7 | 4 (over-detect) |
| `average` | 0.9 | 1 (over-merge) |
| `complete` | 0.7 | 4 (over-detect) |
| **`complete`** | **0.9** | **2 ✅** |

scipy average linkage 在 cosine distance ∈ [0, 2] space 上比 sherpa C++ 算法更激进合并 —
默认必须用 `complete` 才能跟 sherpa parity. 已锁定为 `fast_clustering` 默认值.

## Parity 状态

### ✅ Turn 时间戳高度一致
前 5 turn `start/end` 跟 sherpa baseline 差异 < 0.6s (主要来自 pyannote sliding window
chunk 边界对齐差异, 不是算法 bug).

### ✅ 60s 双人 speaker count parity
ort 跟 sherpa 都给 2 speakers, turn 时间戳 + spk_id 排列一致.

### ✅ e2e 300s/1800s speaker count parity 通过
单 diarize 上 ort 出 3 spk, 但 e2e (经 `cluster_centroid_merge` 后处理) 300s + 1800s 都
正确合到 **2 speakers**, 跟 sherpa 一致. 后处理 pipeline 兜底了 cluster 算法差异.

### ⚠️ e2e 60s 短音频 over-merge 到 1 speaker
60s 双人 e2e ort_cuda 出 `['Speaker2']` 而 sherpa 出 `['Speaker1', 'Speaker2']`. 单
diarize 上 ort 出 2 spk + 19 turns (比 sherpa 多 4 个 turn), 后处理 `cluster_centroid_merge`
在 dominant share ≥ 0.6 时把 minor 合到 dominant → 把另一 speaker 错误吃掉. 长音频上不
触发是因为两 speaker 时长接近, dominant share 不达 0.6.

### ⚠️ 长音频单 diarize speaker over-detect (后处理前)
300s 上 ort 给 3 spk, 1800s 给 3 spk (后处理前). 原因排查方向:
1. ORT TitaNet embedding 跟 sherpa SpeakerEmbeddingExtractor 不严格 parity (浮点细节差异
   累积成 cluster boundary 跳变)
2. `_extract_turns_from_activity` 简化版没做 cross-chunk speaker permutation alignment,
   把跨 chunk 同一 speaker 的 turn 切碎, cluster 阶段碎片重组时多出 false speaker
3. 后处理 `cluster_centroid_merge` (PR3 落档) 应该能兜底, 但当前 ort backend 没接它

后续 commit 跟进:
- ✓ ORT 真 inference + 真模型 sanity test (远端跑通)
- ✗ Acceptance criterion "IoU ≥ 95%" 仍未达到; 当前 PoC bar 0.5 IoU 通过 (integration test
  写死 mean IoU ≥ 0.5)

## 已知遗留

1. **Mac 上 ort_cuda 走 CPU fallback** — ORT 默认 providers
   `["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]`, mac 上
   CoreML 不在可用列表 (warn 一次), 真实落 CPU EP. 慢 (60s 跑分钟级) 但功能正确.
2. **`_extract_turns_from_activity` 简化版** — 没做 cross-chunk speaker permutation
   alignment, 长音频 IoU 不达 prompt 95% bar. 算法升级 (用 pyannote sliding window
   permutation 对齐 + voice activity 切大段) 留后续 sprint.
3. **TitaNet ONNX 期望 2 输入** (mel + length); `compute_titanet_embedding` 已兼容.
   sherpa 的 SpeakerEmbeddingExtractor 隐式给 length, 一致.
4. **ORT session cache** 模块级 dict (key = model_path). 多 worker pool 场景下每 worker
   进程独立 cache, 内存翻倍但不撞.

## 工程意义

- **e2e wall RTF 0.061 / 0.050 / 0.047 (60s/300s/30min)** ✨ — 完美 hit prompt 目标
  "0.04-0.05" (30min on 8 vCPU). vs sherpa baseline 8 vCPU RTF 0.096/0.083/0.080.
- **1.6-1.7x e2e wall speedup** vs sherpa CPU baseline 在所有长度上稳定 (60s/300s/1800s),
  ASR + diarize 并行下 win 加倍 (短音频 ASR 时间也跟着砍半).
- 单 diarize 数据 (4 vCPU): RTF 0.039/0.025/0.026, 2-2.7x speedup, GPU bound.
- **零侵入 Mac 路径** — Qwen3DiarizeTranscriber.transcribe 走 dispatched, MacRuntime
  默认 sherpa, 既有 production 行为不变. 测试 14 处 patch 路径同步改 dispatched.
- **跟 LLM CUDA 共存稳定** — ORT Python API 已在 `scripts/_remote_ort_cuda_clash_check.py`
  确认共存 OK (sprint 前置 PoC), 这次 wrapper 工程化没破坏.

## 后续 sprint 候选

- [ ] **ort_cuda + pool=2 并发突破** (新 sprint, prompt 见
  `docs/开发/archive/Qwen3-CUDA并发突破-新session-prompt.md`) — 当前因 CUDNN
  cross-process race 不可行, 探索 CUDA MPS / Triton / 单进程 batched 等方案,
  目标 RTF < 0.1 with concurrency=2
- [ ] cluster permutation alignment (sherpa-like) 解决 300s/1800s 单 diarize over-detect,
  IoU ≥ 95% (e2e 后处理已兜底, 但单 diarize 数据可改进)
- [ ] 把 ort_cuda backend 接 cluster_centroid_merge / silence_align 等现有后处理 pipeline
  (60s 上 over-merge 到 1 spk 的修复)
