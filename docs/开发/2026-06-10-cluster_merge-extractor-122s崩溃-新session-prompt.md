# cluster_merge embedding extractor 122.88s 段长崩溃 — 新 session 交接

> 背景 session：2026-06-10 ort_cuda 短音频 under-detect 修复验收时发现
> （结案档 `2026-06-10-ort_cuda短音频under-detect-修复结案.md` 的遗留事项）。
> 本文档自含，可直接作为新 session 的开题输入。

## 现象

cluster_merge（后处理 pipeline 第 2 层，over-detect 治理的关键层）算 centroid
时，对超过 **122.88s** 的单个 diarize turn 跑 TitaNet embedding 必崩：

```
RuntimeError: Non-zero status code returned while running Where node.
Name:'/encoder/encoder/encoder.0/mconv.3/Where_1'
... BroadcastIterator::Init ... Attempting to broadcast an axis by a
dimension other than 1. 12288 by 23313
```

生产 transcribe 对该层有 try/except（`qwen3_transcriber.py` ~L577）：**不崩
task，但该 task 静默丢掉整层 cluster_merge**（仅 warning 日志），over-detect
防护失效。

## 已有诊断数据（2026-06-10，已钉死，不用重查）

1. **阈值精确**：≤122s OK，**≥123s 必崩**（Mac CPU 二分实测）。12288 帧 ×
   10ms hop = 122.88s —— 模型图里 Where 节点的 mask 维硬编码 12288 帧。
2. **根因在模型导出产物**，不在 sherpa wrapper：同一个
   `models/qwen3_diarize/sherpa/nemo-titanet/model.onnx`（即
   `config.qwen3.embedding_model`），sherpa `SpeakerEmbeddingExtractor` 与
   我们的 ORT 直跑（`compute_titanet_embedding`，numpy mel）在**同一边界**
   崩 —— 12288 是 TitaNet mconv/SE 模块 mask 的导出期固定值。
3. **影响面（确认过的）**：
   - 长音频评测 `2spk_60min`（吴明辉×程曼祺访谈，有 >4min 的单人长 turn）：
     两轮评测（旧代码轮 + 新代码轮）cluster_merge **全部 4 次崩**——这条音频
     上该层**从来没真正生效过**，spk=2 的正确结果是 filter_spurious 独自扛的。
   - 短音频矩阵 1spk/2spk 的 300s/600s 切片稳定复现。
   - **diarize_ort 主路径不受影响**：per-(chunk,speaker) embedding 输入按
     构造 ≤10s（pyannote chunk 窗），永远到不了 122.88s。
   - 触发条件 = 任何含 **>122.88s 不间断单人 turn** 的音频（独白/访谈常见，
     sherpa 与 ort 两个 diarize backend 的 turn 都会触发）。
4. 崩溃调用链：`apply_cluster_centroid_merge` →
   `cluster_merge.py:53 build_centroids` →
   `qwen3_transcriber.py:98 extractor_fn`（sherpa extractor.compute）。

## 关键事实（避免误判方向）

- 这是**既有问题**，与 2026-06-10 的 under-detect 修复无关（旧代码轮评测
  同样 4 次崩，时间线上早已存在）。
- centroid 不需要整段音频：120s 的 embedding 对 192 维 centroid 已绰绰有余，
  截断/分块平均在语义上无损。
- `build_centroids` 已有 `max_per_speaker=30` 的**段数**上限，但没有**段长**
  上限。
- extractor 由 `build_embedding_extractor_fn(cfg_like)` 构造，per-worker
  lazy singleton 复用（`_ensure_embedding_extractor_fn`）。

## 候选修复方向（按建议优先级）

1. **A. extractor_fn 端段长上限（推荐，工程量小，一处改动全局生效）**：
   `build_embedding_extractor_fn` 返回的 callable 里对 `end - start > ~120s`
   的请求做：
   - 简单版：截取段**中间** 120s（避开段首尾的切点污染），或
   - 讲究版：切 ≤120s 窗逐窗 embedding 后平均再 L2-normalize。
   单测 mock extractor 验证长段不再触发底层调用超限；matrix/eval 复跑确认
   warning 消失且 spk_count 不回归。
2. **B. build_centroids 内 per-段容错**：单段失败跳过该段（centroid 用剩余
   段算），不再让整层 cluster_merge 因一段阵亡。与 A 不冲突，可一起做
   （A 治根，B 兜底）。
3. **C. 重导出 TitaNet ONNX（重，另案评估）**：用 NeMo 重导出去掉 12288
   固定 mask（动态 axes）。改模型 artifact，需要全量精度回归 + 模型分发
   （`scripts/download_qwen3_models.sh`），除非 A/B 不够用否则不动。

## 验收标准建议

- 长音频评测 `2spk_60min` 跑 default 配置：日志无 "cluster_merge 失败"，
  且能看到 cluster_merge 真实 merge events；5 段 spk_count 回归仍全对
- 短音频矩阵（`scripts/_remote_short_audio_matrix.py`）：1spk/2spk
  300s/600s 不再打 "[warn] cluster_merge 失败"，12 组 parity 不回归
- 单测：>122.88s 段走截断/分块路径的行为钉死（mock extractor 断言收到的
  每次调用 ≤ 上限）

## 复现入口 / 环境速查

- 最小复现（Mac 本地即可，~30s）：
  ```python
  from src.core.config import config
  from src.core.qwen3.diarize import _load_audio_mono_16k
  from src.core.qwen3_transcriber import build_embedding_extractor_fn
  import numpy as np
  fn = build_embedding_extractor_fn(config.qwen3)
  audio, _ = _load_audio_mono_16k("tests/fixtures/audio/podcast_2speakers_60s.wav")
  fn(np.tile(audio, 3), 0.0, 123.0)   # ≥123s → RuntimeError; 122s OK
  ```
- 真实场景复跑：远端 3060（`ssh <user>@<dev-host>`，项目
  `~/Dev/projects/funasr_spk_server`）跑
  `scripts/_remote_diarize_accuracy_eval.py`，看 2spk_60min 段的
  "cluster_merge 失败" warning；或 `scripts/_remote_short_audio_matrix.py`
- 相关代码：`src/core/qwen3/cluster_merge.py`（build_centroids / 入口）、
  `src/core/qwen3_transcriber.py`（build_embedding_extractor_fn L64 /
  apply_cluster_centroid_merge_to_turns L105 / 生产 try/except L577）
