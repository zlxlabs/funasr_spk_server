# Qwen3 word_align 显存 PoC 与落地计划

**Date**: 2026-06-16
**Dev box**: `zlx-vm-work-i7-ubuntu2404-3060llm-dev` / `<dev-host>`
**GPU**: RTX 3060 12GB
**Service**: `funasr-server.service`, `ws://<dev-host>:8867`
**Repo**: `~/Dev/projects/funasr_spk_server`
**Profile**: `FUNASR_PROFILE=cuda_dev`

## TL;DR

当前 `word_align` 的 9GB 级显存占用不是典型线性泄露，更像 ONNXRuntime CUDA allocator / BFCArena / CUBLAS workspace 的高水位常驻。但它在工程上不可接受：`batch_size >= 2` 时服务会把 3060 几乎打满，第二轮请求复现 CUDA/CUBLAS 或 BFCArena 分配失败。

实测后最稳的 CUDA 配置是：

```ini
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=true
Environment=FUNASR_QWEN3_WORD_ALIGN_PROVIDER=cuda
Environment=FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE=1
```

但长期在线的产品形态不应该用全局 profile 强制所有请求都开 `word_align`。推荐把词级时间戳做成 API 级按请求开关，默认关闭；请求开启时优先走 CUDA `batch_size=1`，显存不足或 CUDA 路径不可用时 fallback 到 CPU，最终再考虑独立 sidecar/worker 来释放空闲 VRAM。

远程 8867 服务在本轮 PoC 结束后已恢复到稳定配置：

```ini
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=false
```

恢复验证通过，`word_align=false` 时 RTF 约 `0.0787 / 0.0319`，funasr 进程显存约 `2850 MiB`。

## 背景

`cuda_prod` / `cuda_dev` profile 之前默认开启 `qwen3.word_align_enabled=True`。在 CUDA runtime 下，`word_align_provider=auto` 会解析为 `CUDAExecutionProvider`，并用默认 `word_align_batch_size=16`。

当前实现特点：

- `Qwen3DiarizeTranscriber._ensure_word_aligner()` 按 worker 懒加载 `WordAligner`。
- `WordAligner` 内部复用一个 MMS CTC forced aligner ONNX session。
- 模型路径是 `./models/qwen3_diarize/ctc_forced_aligner/model.onnx`，文件约 1.2GB。
- 词级时间戳只在 JSON 输出路径执行，SRT 不挂 `words`。
- 一旦 CUDA `word_align` session 被加载，显存高水位会常驻在服务进程里。

用户预期 `ASR + diarize + word_align` 可能在 4-5GB 左右。实际观察到默认配置下 funasr 进程常驻约 9GB，导致整卡接近满载。

## 测试环境

同卡还运行另一个服务：

| 进程 | 显存 |
|---|---:|
| CapsWriter | `2864 MiB` |
| funasr, `word_align=false` 触发后 | `~2850 MiB` |

测试音频和脚本：

```bash
cd ~/Dev/projects/funasr_spk_server
venv/bin/python scripts/_remote_diarize_e2e_probe.py \
  --server ws://localhost:8867 \
  --audio tests/fixtures/audio/podcast_2speakers_60s.wav
```

probe 会跑：

- `diarize=true` JSON fresh
- `diarize=false` JSON 缓存投影
- `diarize=false` JSON fresh
- `diarize=false` / `diarize=true` SRT 缓存路径

RTF 表中格式为：

- `diarize=true RTF`
- `diarize=false fresh RTF`

## 实测结果

### 稳定基线：word_align=false

配置：

```ini
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=false
```

结果：

| 指标 | 数值 |
|---|---:|
| funasr idle 显存 | `2364 MiB` |
| funasr probe 后显存 | `2850 MiB` |
| 整卡 probe 后 | `5725 MiB used / 6196 MiB free` |
| RTF, diarize=true | `0.0790` / `0.0787` |
| RTF, diarize=false fresh | `0.0316` / `0.0319` |
| 结果 | e2e probe 全部通过 |

这是当前 8867 的稳定运行配置。

### CUDA word_align, batch=16

配置：

```ini
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=true
Environment=FUNASR_QWEN3_WORD_ALIGN_PROVIDER=cuda
Environment=FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE=16
```

结果：

| 轮次 | funasr 显存 | 整卡剩余 | RTF | 结果 |
|---|---:|---:|---|---|
| 触发前 | `2364 MiB` | `6682 MiB` | n/a | 服务正常 |
| 第 1 轮后 | `8996 MiB` | `50 MiB` | `0.1111 / 0.0430` | 通过 |
| 第 2 轮后 | `9040 MiB` | `6 MiB` | n/a | 240s timeout |

错误日志：

```text
BFCArena::AllocateRawInternal(...) Failed to allocate memory for requested buffer of size 34504704
```

结论：默认 batch=16 不适合 3060 长期在线，尤其不能和 CapsWriter 同卡共存。

### CPU word_align, batch=16

配置：

```ini
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=true
Environment=FUNASR_QWEN3_WORD_ALIGN_PROVIDER=cpu
Environment=FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE=16
```

结果：

| 轮次 | funasr GPU 显存 | RSS | RTF | 结果 |
|---|---:|---:|---|---|
| 触发前 | `2364 MiB` | `1049 MiB` | n/a | 服务正常 |
| 第 1 轮后 | `2850 MiB` | `6039 MiB` | `0.2481 / 0.1735` | 通过 |
| 第 2 轮后 | `3660 MiB` | `6264 MiB` | `0.2081 / 0.1697` | 通过 |

结论：

- GPU 压力可控，没有 9GB 常驻问题。
- RTF 明显变差，CPU/RSS 开销高。
- 适合作为 fallback，不适合作为默认高吞吐路径。

### CUDA word_align, batch=1

配置：

```ini
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=true
Environment=FUNASR_QWEN3_WORD_ALIGN_PROVIDER=cuda
Environment=FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE=1
```

结果：

| 轮次 | funasr 显存 | 整卡剩余 | RSS | RTF | 结果 |
|---|---:|---:|---:|---|---|
| 触发前 | `2364 MiB` | `6682 MiB` | `970 MiB` | n/a | 服务正常 |
| 第 1 轮后 | `5956 MiB` | `3090 MiB` | `3041 MiB` | `0.1177 / 0.0423` | 通过 |
| 第 2 轮后 | `6212 MiB` | `2834 MiB` | `3180 MiB` | `0.0784 / 0.0425` | 通过 |

结论：

- 这是当前最好的 CUDA 候选。
- 显存从 9GB 降到约 6.2GB。
- 第二轮 RTF 接近 `word_align=false` 的 `diarize=true` 基线。
- 同卡 CapsWriter 存在时仍保留约 2.8GB 显存余量。

### CUDA word_align, batch=2

配置：

```ini
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=true
Environment=FUNASR_QWEN3_WORD_ALIGN_PROVIDER=cuda
Environment=FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE=2
```

结果：

| 轮次 | funasr 显存 | 整卡剩余 | RTF | 结果 |
|---|---:|---:|---|---|
| 触发前 | `2364 MiB` | `6682 MiB` | n/a | 服务正常 |
| 第 1 轮后 | `9006 MiB` | `40 MiB` | `0.1121 / 0.0428` | 通过 |
| 第 2 轮中 | `9038 MiB` | `8 MiB` | n/a | 失败 |

错误日志：

```text
CUBLAS failure 3: the resource allocation failed
expr=cublasCreate(&cublas_handle_)
```

结论：`batch=2` 只带来很小 RTF 改善，但显存直接回到 9GB 档位，第二轮失败。不可作为生产配置。

### CUDA word_align, batch=4

配置：

```ini
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=true
Environment=FUNASR_QWEN3_WORD_ALIGN_PROVIDER=cuda
Environment=FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE=4
```

结果：

| 轮次 | funasr 显存 | 整卡剩余 | RTF | 结果 |
|---|---:|---:|---|---|
| 触发前 | `2364 MiB` | `6682 MiB` | n/a | 服务正常 |
| 第 1 轮后 | `9006 MiB` | `40 MiB` | `0.1142 / 0.0428` | 通过 |
| 第 2 轮中 | `9040 MiB` | `6 MiB` | n/a | 失败 |

错误日志：

```text
BFCArena::AllocateRawInternal(...) Failed to allocate memory for requested buffer of size 34062336
```

结论：`batch=4` 和 `batch=2` 一样跨过显存阈值，不可用。

## 为什么 batch=2/4 会接近 batch=16 的显存

这不是逻辑矛盾。ONNXRuntime CUDA 显存不是按 batch 线性增长，存在明显阈值效应：

1. `batch=1` 可以串行复用中间 activation；`batch>=2` 会让更多中间层同时驻留。
2. batch 变化会触发不同 CUDA kernel / CUBLAS 算法，workspace 需求可能非线性增加。
3. ORT BFCArena 按高水位缓存显存，申请过的大块显存不一定还给 driver。
4. 同卡已有 CapsWriter 占约 2.8GB，`batch>=2` 后整卡只剩个位数到几十 MiB，后续请求连 CUBLAS handle 或 34MB buffer 都可能申请失败。

因此：

- `batch=1` 是可用区间。
- `batch>=2` 已经跨过当前 3060 部署的危险阈值。
- 是否“第一轮能跑过”不应作为生产可用标准，第二轮失败就应判定配置不可用。

## 推荐产品设计

### 1. API 级按请求开关

不要再由 server profile 全局决定所有请求都启用词级时间戳。建议请求字段：

```json
{
  "engine": "qwen3",
  "diarize": true,
  "word_align": true,
  "language": "chi"
}
```

默认值：

```json
{
  "word_align": false
}
```

原因：

- 多数转写/字幕请求不一定需要词级时间戳。
- CUDA word_align 即使用 `batch=1`，首次触发后也会让 funasr 从约 2.8GB 常驻到约 6.2GB。
- 让调用方显式选择成本，服务端才能做显存预算。

### 2. CUDA 路径固定 batch=1

3060 12GB + CapsWriter 同卡的当前推荐：

```ini
FUNASR_QWEN3_WORD_ALIGN_PROVIDER=cuda
FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE=1
```

禁止将 `batch_size=2/4/16` 作为 3060 默认生产配置。

### 3. CPU fallback

建议 fallback policy：

1. 请求未开启 `word_align`：不加载 aligner。
2. 请求开启 `word_align`：
   - GPU 显存余量足够且 CUDA aligner 可用：走 CUDA `batch=1`。
   - GPU 显存不足或 CUDA aligner 正忙：走 CPU word_align。
   - CPU 也失败：返回正常 ASR/diarize 结果，但不挂 `words`，并在 metadata/raw_result 里记录原因。

注意：最好在 CUDA 执行前做显存 preflight，不要等 CUDA OOM 后再 fallback。OOM 后 ORT/CUDA allocator 可能已经把进程留在不健康状态。

### 4. 独立 worker / sidecar

长期最稳的结构是把 word_align 从主 ASR 服务里拆出去：

- 主服务常驻 `ASR + diarize`，保持约 2.8GB。
- word_align sidecar 按需启动 CUDA session。
- 空闲一段时间后退出进程，真正释放 VRAM。
- 主服务通过队列或 RPC 调 sidecar，失败时 fallback CPU 或返回无 words 结果。

这个方案工程量更大，但能解决“偶尔开一次词级时间戳，主进程长期多占 3GB+”的问题。

## 落地任务

### P0：避免默认踩坑

- [ ] 修改 CUDA profile 或部署 override，避免 `cuda_prod` / `cuda_dev` 默认 `word_align_enabled=True + batch=16`。
- [ ] 若短期仍要全局开启，强制使用 `word_align_provider=cuda` + `word_align_batch_size=1`。
- [ ] 更新部署文档，标注 3060 上 `batch>=2` 不可用。

### P1：API 级 word_align 开关

- [ ] 在请求 schema 增加 `word_align: Optional[bool]`。
- [ ] WebSocket handler 解析请求级 `word_align`。
- [ ] `TranscriptionTask` 或等价任务对象携带请求级 `word_align`。
- [ ] `Qwen3DiarizeTranscriber.transcribe(...)` 支持 per-request override。
- [ ] JSON metadata 回显本次请求的 effective `word_align`，不能只读全局 config。
- [ ] 保持兼容：旧客户端不传字段时默认 `false`。

### P1：缓存 key 修正

当前缓存已有 `qwen3+wa:<lang>` 维度，但落地 API 开关后必须确认：

- [ ] cache key 使用“本次请求 effective word_align”，不是全局 config。
- [ ] `language` 仍参与 `word_align` cache key。
- [ ] `word_align=false` 不能命中带 words 的结果后误报无 words。
- [ ] `word_align=true` 不能命中无 words 的旧结果后误以为完成对齐。
- [ ] `diarize=false` 的 projection 逻辑继续保留 words。

### P1：CUDA batch 和 provider 配置治理

建议配置项区分 CUDA 和 CPU：

```python
word_align_enabled_default: bool = False
word_align_provider_policy: str = "cuda_then_cpu"
word_align_cuda_batch_size: int = 1
word_align_cpu_batch_size: int = 16
```

落地项：

- [ ] 将 3060 CUDA 默认 batch 调为 1。
- [ ] CPU fallback 保留 batch 16 或独立调参。
- [ ] 启动日志打印 `word_align_provider_policy` 和 batch 参数。
- [ ] 避免仅一个 `word_align_batch_size` 同时控制 CPU/CUDA，导致两边无法独立优化。

### P2：显存 preflight

目标：不要在显存不足时创建 CUDA word_align session。

- [ ] 增加 GPU memory probe，优先用 NVML，退化到 `nvidia-smi`。
- [ ] CUDA aligner 未加载时，free VRAM 低于阈值则不加载 CUDA aligner。
- [ ] CUDA aligner 已加载时，free VRAM 低于阈值则拒绝新 CUDA word_align 请求或转 CPU。
- [ ] 记录每次请求前后 VRAM delta，便于线上追踪。

阈值需要二次验证。基于当前数据，保守起点可以是：

- CUDA word_align 未加载：要求 free VRAM 至少 4.5GB。
- CUDA word_align 已加载：要求 free VRAM 至少 1.5-2GB。

这些值不是最终标准，只是防止 3060 进入个位数 MiB 剩余显存的保护线。

### P2：错误处理和降级

- [ ] 捕获 `BFCArena Failed to allocate`、`CUBLAS resource allocation failed` 等 ORT/CUDA 错误。
- [ ] 请求级返回中标记 `word_align_error`。
- [ ] CUDA 失败后不要继续在同一请求里反复重试 CUDA。
- [ ] 评估 CUDA OOM 后是否需要重建 aligner 或重启 word_align worker。

### P3：独立 word_align worker

- [ ] 设计 word_align sidecar 协议：输入 audio path + ASR chunks + language，输出 words + stats。
- [ ] 支持 idle TTL 自动退出，释放 CUDA VRAM。
- [ ] 主服务 fallback：sidecar 不可用时 CPU 或无 words。
- [ ] 增加 sidecar 生命周期日志和健康检查。

## 测试计划

### Unit tests

- [ ] 请求 schema 默认 `word_align=false`。
- [ ] 请求 `word_align=true` 时 metadata 回显 true。
- [ ] cache key 按 effective word_align + language 区分。
- [ ] `diarize=false` projection 保留 words。
- [ ] fallback 后 metadata/raw_result 记录 provider 和错误原因。

### Integration tests

- [ ] 本地 mock aligner：验证 API 开关只影响本次请求。
- [ ] 远端 3060：`word_align=true provider=cuda batch=1` 两轮 60s probe。
- [ ] 远端 3060：`word_align=true provider=cpu` fallback probe。
- [ ] 远端 3060：显存不足时不加载 CUDA aligner。

### Performance guard

以 60s `podcast_2speakers_60s.wav` 为 smoke test：

| 配置 | 期望 |
|---|---|
| `word_align=false` | RTF 约 `0.08 / 0.03`，funasr 约 `2.8GB` |
| CUDA `word_align=true batch=1` | 两轮通过，funasr 不应接近 `9GB` |
| CUDA `batch>=2` | 不作为 production guard，只保留为反例数据 |
| CPU fallback | 正确产出 words，RTF 允许明显变慢 |

## Reproducer

临时切 CUDA batch=1：

```bash
sudo tee /etc/systemd/system/funasr-server.service.d/10-no-word-align.conf >/dev/null <<'EOF'
[Service]
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=true
Environment=FUNASR_QWEN3_WORD_ALIGN_PROVIDER=cuda
Environment=FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE=1
EOF

sudo systemctl daemon-reload
sudo systemctl restart funasr-server.service
```

跑 probe：

```bash
cd ~/Dev/projects/funasr_spk_server
venv/bin/python scripts/_remote_diarize_e2e_probe.py \
  --server ws://localhost:8867 \
  --audio tests/fixtures/audio/podcast_2speakers_60s.wav
```

看显存：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
```

恢复稳定配置：

```bash
sudo tee /etc/systemd/system/funasr-server.service.d/10-no-word-align.conf >/dev/null <<'EOF'
[Service]
Environment=FUNASR_QWEN3_WORD_ALIGN_ENABLED=false
EOF

sudo systemctl daemon-reload
sudo systemctl restart funasr-server.service
```

## 当前决策

短期：

- 8867 长期在线默认保持 `word_align=false`。
- 若需要全局开启词级时间戳，只允许 CUDA `batch=1`。
- `batch=2/4/16` 在这台 3060 上判定为不可用。

中期：

- 做 API 级 `word_align` 开关。
- 做 CUDA `batch=1` + CPU fallback。
- 做 cache key 和 metadata 的 request-level 修正。

长期：

- 拆独立 word_align worker/sidecar，让偶发词级时间戳请求不污染主 ASR 服务的长期 VRAM baseline。

---

## 工程评审结论（2026-06-16 /plan-eng-review + codex 外部声音）

### 锁定范围：P0 + P1 + CPU fallback + OOM poison

P2（VRAM preflight）和 P3（sidecar）推迟，已进 `TODOS.md` #17 / #18。
推迟依据：所有 profile 都是 `qwen3_pool_size=1`，word_align 并发已被序列化为 1，
preflight 拦的「第二个并发 CUDA 会话」在 pool=1 下不会出现。

### 锁定的设计决策（实现按此执行，不再重议）

| # | 决策 | 内容 |
|---|---|---|
| 1A | effective word_align 算一次 | 优先级链（请求 > config 兜底）**在任何 early cache 查询之前**解析一次（codex #2：chunked upload 在 `websocket_handler:454` 有早返回，必须在 `FileUploadRequest` 构造/session 回填层解析，不能只在 enqueue），写进 `TranscribeOptions.word_align` 确定 bool；transcribe/cache/metadata 全读它，删掉 3 处 config 读 |
| 2A | SRT 强制降 +wa | 缓存维度有效 word_align = `effective AND output_format=='json'`；`compute_cache_engine` 收 output_format；SRT 即使 word_align=true 也降回纯 `qwen3` tag；metadata.word_align 反映 delivered（SRT=false） |
| 1B | config 只加 cuda batch | 加 `word_align_cuda_batch_size: int = 1`，保留 `word_align_batch_size: int = 16`（CPU/默认）；**不加 provider_policy 字段**；env 走 `_override_if_set` |
| 2A(CQ) | CPU fallback + poison | CUDA 失败→重建 CPU（batch=16）重试；CPU 也失败→无词 + `word_align_error`；**OOM 后 poison 该 pool**（共享 flag，非单实例 — codex #8）直走 CPU；poison 时 `del` CUDA session + gc 尝试回收 VRAM，**前后打 nvidia-smi delta 到日志**（不当保证） |
| A | 错误分类穿透 | `align_chunks` 现状逐窗 catch 吞 OOM（`word_align.py:164`），fallback 触发不了。加资源错误判定（ORT Fail/BFCArena/CUBLAS），资源类错误**绕过逐窗 catch、立即上抛**触发 poison+CPU；普通对齐失败仍逐窗跳过 |
| B | 失败不写 +wa | 请求 word_align 但 segments 实际无词（两 provider 都失败）→ **不写 +wa 缓存行**（避免 exact-hit 永久毒化该文件）；只缓存成功对齐 |
| C | 跨引擎回退排除折维行 | `get_cached_result` 的 file_hash 跨引擎回退只命中「裸 engine」行，排除 `+wa`/`+nospk` 折维行（codex #4：补全 strict 隔离的反向漏洞，双向对称） |

### 实现细节（codex 补充，归入实现，不单列决策）
- `_ensure_word_aligner` 不能再是简单 singleton（codex #10）：需 cuda + cpu 两套 session 变体
- poison 是 **in-proc CUDA pool 概念**（codex #9）：Mac file worker 跑完即退，poison 对其无意义
- metadata 需新增 `word_align_error` 字段，且**缓存命中出口**也要能回显（codex #11，fresh stats 在 raw_result，命中路径需另取）
- delivered 语义（codex #12）：`word_align=true` 仅当 segments 实际挂上词；部分窗口失败仍记 true（有词）+ stats 记 failed_windows
- P0 必须确保 config 兜底显式 false everywhere（codex #13），否则 per-request 默认不是真 OFF

### 失败模式（每条新路径 + 是否有测试/错误处理/对用户可见）

| 路径 | 失败方式 | 测试 | 错误处理 | 用户可见 |
|---|---|---|---|---|
| effective 解析 | early cache 用未解析值 → 缓存/metadata 错 | ✅(决策1A) | 解析前置 | 是 |
| SRT +wa | output_format 未透传 → tag 错 | ✅ | 确定性 | 是 |
| CUDA OOM 触发 | 逐窗吞 OOM → fallback 不触发（**曾是 silent 致命**）| ✅ | 错误分类穿透 | 是 |
| CPU 也失败 | 返回无词 | ✅ | metadata error | 是（清晰） |
| 失败缓存 | 无词存 +wa → 文件永久毒化（**曾是 silent**）| ✅ | 失败不写 +wa | 是 |
| 反向污染 | no-wa 请求命中 +wa 行（**曾是 silent**）| ✅ | 跨引擎排除折维 | 是 |
| poison 范围 | pool>1 时单实例 poison 不传染 | ✅ | 共享 flag | 是 |
| dispose 无效 | ORT 不还显存 | n/a | 日志 delta + TODO #18 | 是（日志） |
| 老 .task 文件 | 无 word_align 字段 → 崩 | ✅(REGRESSION) | 默认兜底 | 是 |

3 个曾是「silent + 无处理」的致命 gap（OOM 不触发 fallback / 失败缓存毒化 / 反向污染）均已被评审决策覆盖。

### 实现并行化（git worktree）

```
Phase 1（先行，互相独立可并行）:
  Lane 0: config P0 (config.py)                        — 独立
  Lane A: schema + enqueue 解析 + worker 序列化           — 基础，后续全依赖
          (schemas.py, task_manager.py, websocket_handler.py, qwen3_worker_process.py)

Phase 2（Lane A 合并后，三路并行 worktree，文件不重叠）:
  Lane B: cache 正确性 (database.py)                     — SRT降+wa / 失败不写 / 跨引擎排除
  Lane C: transcriber fallback+poison+错误分类            — qwen3_transcriber.py + qwen3/word_align.py
  Lane D: metadata (result_projection.py)                — per-request 回显 + word_align_error

冲突标记: B/C/D 文件不重叠，低冲突。Lane B 依赖但不修改 result_projection（仅 import），与 Lane D 安全并行。
执行顺序: Lane 0 ∥ Lane A → 合并 A → Lane B ∥ Lane C ∥ Lane D → 合并。
```

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | ISSUES_FOUND | 15 findings, 3 critical new folded |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES_OPEN→RESOLVED | 8 issues, 3 critical gaps (all addressed) |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (后端) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | n/a |

- **CODEX:** 外部声音确认全部架构方向，挖出 5 个新问题，3 个 CRITICAL（OOM 不触发 fallback / 失败缓存毒化 / 反向缓存污染）已纳入决策。
- **CROSS-MODEL:** 无冲突 — codex 与本次评审同向，仅向深处补充。
- **UNRESOLVED:** 0
- **VERDICT:** ENG CLEARED（范围已缩，7 个设计决策 + 3 个 codex critical 全部锁定）— 可进入实现。
