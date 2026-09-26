# ort_cuda 短音频 under-detect 调查与修复 — 新 session 交接

> **已结案（2026-06-10）**：根因为 ort 复刻跨 chunk 平均 pyannote logits（slot
> 混叠）+ TitaNet 拿错 ONNX 输出（logits 当 embedding）两层叠加。修复与验收
> 见 `2026-06-10-ort_cuda短音频under-detect-修复结案.md`。本文档仅留档。
>
> 背景 session：2026-06-10 diarize 开关 3060 验收时发现。
> 本文档自含，可直接作为新 session 的开题输入。

## 现象

`tests/fixtures/audio/podcast_2speakers_60s.wav`（60 秒双人播客）：

- Mac sherpa 路径：**2 speaker** ✓（主 87.8% / 次 12.2%）
- 3060 ort_cuda 路径：**1 speaker** ✗
- `tests/integration/test_diarize_ort_parity.py::test_per_speaker_iou_above_threshold`
  （同一条音频，sherpa vs ort 同机对比）**在 Mac CPU 上也挂**（mean IoU 0.22）
  → 说明不是 CUDA EP 数值问题，是 **ort 复刻聚类实现 vs sherpa C++ 在短音频上的行为差异**。

## 已有诊断数据（2026-06-10，3060 实测，脚本见文末）

分层归因，第二个说话人是这样丢的：

```
raw(ort_cuda):         clusters=2  sizes=[55.9s, 1.8s]  shares=[96.8%, 3.2%]
after filter_spurious: clusters=1  ← 1.8s < 2.0s 绝对阈值，被当"假说话人"归并
after cluster_merge:   clusters=1  （此层无关）
```

两层叠加：

1. **根因（聚类层）**：ort_cuda 的 raw 聚类把第二人绝大部分 turn 错归进第一簇，
   只剩 1.8s 残留（sherpa 给第二人 ~12% share，约 7s+）。
2. **顺手补刀（守门层）**：`filter_spurious_speakers` 的 `min_speaker_total=2.0s`
   绝对阈值把 1.8s 残留清掉 → 输出 1 speaker。

## 关键事实（避免误判方向）

- **长音频上 ort_cuda 没有这个问题**：2026-05-22 精度评测
  （`docs/开发/gpu加速/2026-05-22-cuda-diarize-accuracy.md`）在 1/2/3+/4/6 spk
  的 16–60min 评测集上 ort_cuda spk_count **全对**，与 Mac sherpa 完全一致。
  **短音频（≤ 数分钟）是该评测的盲区**，本课题本质是"短音频聚类质量"。
- diarize 开关分支（spike/qwen3-diarize-poc 2026-06-10 部分）对聚类链路
  （diarize_ort.py / cluster_merge.py / filter_spurious）**零改动**，与本课题无关。
- ort_cuda 自建的动机（sherpa CUDA build 与 llama.cpp 撞 segfault）依然成立，
  "换回 sherpa" 不是选项；Mac 走 sherpa 不受影响。

## 评测资产（现成，不用重建）

- **标定评测集**（远端 3060 `tmp_long_audio/eval_set/`，~3.5h，真实人数已人工标定）：
  1spk_real(16min) / panel 3+(34min) / 2spk(60min) / 4spk(44min) / 6spk(60min)，
  标定见 `tmp_long_audio/eval_set/README.md`
- **ablation 评测框架**：`scripts/_remote_diarize_accuracy_eval.py`
  （1 次 ASR+diarize 缓存，5 配置后处理 ablation 秒级复跑，全套 ~14min）
- **短音频样本**：本地/远端 `tests/fixtures/audio/podcast_2speakers_60s.wav`；
  可从 eval_set 长音频切 1/5/10min 片段扩出"短音频×多人数"矩阵
- **parity 测试**：`tests/integration/test_diarize_ort_parity.py`（IoU 断言现成）

## 候选修复方向（按建议优先级）

1. **A. 聚类实现对照（根因路线）**：对同一条 60s 音频 dump 两边中间产物对比——
   embedding 向量（是否同模型同输入）、距离矩阵、linkage 合并序列。
   ort 复刻在 `src/core/qwen3/diarize_ort.py`（scipy cosine + complete linkage 复刻
   sherpa FastClustering）。重点怀疑：linkage 细节 / 阈值语义（0.9 在两套实现里
   是否等价）/ 短音频下 embedding 窗口数太少时的退化行为。
2. **B. filter_spurious 守门自适应（缓解路线，工程量小）**：
   - 绝对阈值按时长自适应（如 `min(2.0, duration * 0.02)`），或
   - "top-2 簇保护"：raw 第二大簇即使低于阈值也不归并（除非占比 < 某下限）。
   注意：该函数同时服务 sherpa 路径，改动必须跑全量长音频评测防回归
   （单人场景靠它清 3 个 spurious 簇，不能放水）。
3. **C. cluster_threshold 短音频自适应**：0.9 对短音频下调试验（ablation 框架现成）。
4. （兜底，另案）num_speakers per-request 化（TODOS #15）。

## 验收标准建议

- 60s podcast 在 ort_cuda 上 = 2 speaker，parity IoU 测试转绿
- 长音频评测集 5 段 spk_count 回归全对（ablation 框架一键复跑）
- 短音频矩阵（新增 1/5/10min × 1/2/多人）至少 spk_count 全对

## 环境速查

- 3060：`ssh <user>@<dev-host>`，项目 `~/Dev/projects/funasr_spk_server`，
  分支 `spike/qwen3-diarize-poc`，server 起法/坑见该机记忆与 `~/start_cuda.sh`
- 分层诊断脚本（本次用过，远端 `/tmp/diag_60s.py`）：调
  `run_diarization_dispatched` → `filter_spurious_speakers` →
  `apply_cluster_centroid_merge_to_turns` 逐层打 cluster 数/时长分布
