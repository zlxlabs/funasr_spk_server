# ASR 引擎架构（CLAUDE.md 规则摘录）

> 下文自仓根 `CLAUDE.md`（基线 `f0c9e13`）按原标题、原正文搬出，本卡不改写。对应定案仍以 `docs/开发/` 既有日期文档为准，本文件只承担规则瘦身摘录。
>
> 核销表：`docs/开发/2026-08-28-CLAUDE规则瘦身核销表.md`

# ASR 引擎架构

## 当前现状（2026-05-23）

两个引擎并行接入生产，dispatch 走轻量函数路由（**不是 ABC 抽象**）：

- **FunASR**（生产稳定）: `src/core/funasr_transcriber.py`，MPS GPU 加速。canonical 为**句级** segments（引擎层不再合并）；JSON 出口在 serve 投影层 `merge_segments_view` 做同说话人相邻句合并 + `segment_merge_max_span_sec`（默认 120s）上限（issue #1）
- **Qwen3**: `src/core/qwen3_pool_transcriber.py`（**runtime-aware 池 dispatch**, 见下文）+ `src/core/qwen3_transcriber.py`（单例）+ `src/core/qwen3/asr.py`（引擎构造）。Mac 上 frontend ONNX 走 CoreML ANE（`onnx_provider="COREML_ANE_FE"`），见 `spikes/qwen3_mac_hw_accel/SUMMARY.md`
- **Dispatch**: `src/core/transcriber_dispatch.py` 的 `resolve_transcriber()` 按 engine 名分支
- **引擎选择优先级**: `upload_request.engine` > `config.transcription.default_engine`（env `FUNASR_DEFAULT_ENGINE`）> `funasr`
- **缓存隔离**: 缓存 key 按 `(file_hash, engine)` 区分，跨引擎不命中；engine 维折 word_align / diarize 状态（`qwen3+wa:<lang>+nospk` 形态，见「diarize 开关」节），折维参数统一走 `database.cache_params_for(task)`（word_align 维读 `options.word_align` 非 config，且 `output_format=srt` 强制降 +wa；写入用 `cache_save_engine_for` 对齐失败时降 +wa 防毒化；跨引擎回退排除折维行，见 pipeline 5.5 词级时间戳节）

## Runtime + Diarize backend 抽象

`src/core/runtime.py` 的 `detect_runtime()` 返回 `MacRuntime / CudaRuntime / CpuRuntime`，按 `sys.platform` + `onnxruntime CUDAExecutionProvider` 探测自动选；`FUNASR_RUNTIME=cpu/mac_ane/cuda` env 强制 override。每个 runtime 暴露：
- `validate()` — `CudaRuntime` 显式 assert CUDA EP 在 ORT providers 列表，缺则 fail-fast（替代 ORT silent CPU fallback）
- `recommend_diarize_backend()` — Mac/Cpu → `sherpa`, Cuda → `ort_cuda`
- `recommend_num_threads()` — Mac 固定 4, Linux 按 `cpu_count()` 给 2/4

Qwen3 diarize 有 **两个 backend 实现**，通过 `src/core/qwen3/diarize.py:run_diarization_dispatched` 路由：

- **`sherpa`**（默认，Mac/Cpu）: `src/core/qwen3/diarize.py:run_diarization`，sherpa-onnx `OfflineSpeakerDiarization` + sherpa C++ FastClustering
- **`ort_cuda`**（CUDA 平台默认）: `src/core/qwen3/diarize_ort.py:run_diarization_ort_cuda`，Python `onnxruntime` 直 wrap pyannote-segmentation-3.0 + TitaNet + scipy 复刻 FastClustering（cosine + complete linkage）。**pipeline 结构 1:1 移植 sherpa C++**（per-chunk 独立 argmax + per-(chunk,speaker) embedding，2026-06-10 修短音频 under-detect 时重写，见 `docs/开发/2026-06-10-ort_cuda短音频under-detect-修复结案.md`；⚠️ pyannote slot 是 chunk 局部的，禁止跨 chunk 平均 logits；TitaNet ONNX 双输出必须取 `embs` 不是 `logits`）。**8 vCPU + RTX 3060 上 30min wall RTF 0.047 vs sherpa CPU 0.080**，详见 `docs/开发/gpu加速/2026-05-22-ORT-CUDA-diarize-backend.md`
- **优先级**: 显式 `backend` 参数 > `FUNASR_QWEN3_DIARIZE_BACKEND` env > `runtime.recommend_diarize_backend()`
- **为什么自建 ort_cuda 而不是用 sherpa CUDA build**: sherpa-onnx CUDA build 的 C++ wrapper 跟 llama.cpp CUDA 撞 segfault，ORT Python API 不撞（`scripts/_remote_ort_cuda_clash_check.py` 验证过）

## Qwen3 池 dispatch (runtime-aware)

`src/core/qwen3_pool_transcriber.py:get_qwen3_pool_transcriber()` 按 `detect_runtime()` 分发到两套池：

- **`cuda` runtime** → **`Qwen3InProcPool`** (`src/core/qwen3_inproc_pool.py`)
  - 单进程内 N 个 `Qwen3DiarizeTranscriber` 实例，`asyncio.Queue` 调度 acquire/release
  - 共享同一 cuda context，**race-free** — 避开 multi-process 跨进程 CUDNN/cuda buffer race
  - RTX 3060 + 8 vCPU 实测 `pool_size=2` 跑 1800s × 2 并发: TOTAL_WALL 142s, 每 task RTF 0.079
  - 详见 `docs/开发/gpu加速/2026-05-23-CUDA并发突破.md`
- **其他 runtime (Mac/CPU)** → **`Qwen3PoolTranscriber`** (file-based multi-process pool, 历史路径)
  - `FileBasedProcessPool` 派发到 `qwen3_worker_process.py` subprocess
  - Mac MPS 上行为 100% 不变

**为什么不统一一套池**: CUDA 多进程下 cuDNN handle 跨进程 race 撞死 worker (实测 MPS 任何 thread% 设置都不解); Mac 多进程下 sherpa CPU 没此问题, multi-process 隔离反而更稳. **rule-of-two backends, runtime 自动选**.

`pool_size` 两套共用 `config.transcription.qwen3_pool_size` (env `FUNASR_QWEN3_POOL_SIZE` 覆盖). 单例缓存在 `_qwen3_pool_singleton`, 测试用 `reset_qwen3_pool_singleton()` 清.

## Qwen3 后处理 pipeline

`Qwen3DiarizeTranscriber.transcribe` 在 ASR + diarize 后串联多层后处理（顺序固定，第 1–5 + 5.5 + 5.7 层各有 config flag + env override 可关，第 6 层是无条件的输出层规范化）。**per-request `options.diarize=False` 时走精简管线**（见下文「diarize 开关」节）：

1. **`filter_spurious_speakers`** — 丢掉总时长太小的"假说话人"，把碎片归到时间最近的有效 speaker
2. **`apply_cluster_centroid_merge`**（PR3，`cluster_merge_enabled`）— 多人场景把过聚的 cluster 合并；用 sherpa embedding extractor 算 centroid。dominant share ≥ 0.6 时还会用更宽松的 `cluster_merge_dominant_minor_threshold`（默认 0.5）把跟 dominant 接近的 minor cluster 也合到 dominant（兜底拦截解码器漂移引入的中长噪声 cluster，见 `docs/开发/archive/spk-over-detect-归因调研结果.md`）。⚠️ extractor 带 120s 段长上限（`MAX_EXTRACTOR_SEGMENT_SEC`，TitaNet ONNX 导出图 12288 帧 mask 硬上限 = 122.88s，超限段切等宽窗逐窗 embedding 平均；`build_centroids` 另有 per-段容错兜底，见 `docs/开发/2026-06-10-cluster_merge-extractor-122s段长崩溃-修复结案.md`）
3. **`merge_asr_chunks_and_diarize`** — 按 Qwen3 内部 40s chunk 时间窗切文本到 diarize turn
4. **`apply_short_segment_guard`**（PR4，`short_segment_guard_enabled`）— drop 微短段 / ABA 抖动平滑 / 合并连续同 speaker
5. **`apply_silence_align_to_segments`**（spike 405abf6，`silence_align_enabled`）— ffmpeg silencedetect + snap-to-silence 把段切点吸附到最近静音中点，60s podcast +19pp / 60min long +33pp 对齐率，RTF 影响 <1%，见 `spikes/qwen3_silence_align/SUMMARY.md`
5.5. **word_align 词级时间戳**（**per-request 开关 `options.word_align`，全 profile 默认关**，2026-06-16 显存落地，定案: `docs/开发/gpu加速/2026-06-16-Qwen3-word-align显存PoC与落地计划.md`）— MMS-300M CTC-FA（`src/core/qwen3/word_align.py`，deskpai `ctc-forced-aligner` ONNX）按 ASR chunk 逐窗口对齐，把每个词的绝对秒时间**增量挂进** `segment.words`（`attach_words_to_segments` 最大时间重叠归段），**不替换段边界**。挂在 silence_align 之后、relabel 之前（干净段上挂词）。**JSON-only**（SRT 不带词，跳过省 RTF）。
   - **开关语义（决策 1A）**：`FileUploadRequest.word_align: Optional[bool]=None`（None=未指定跟随 config 兜底），`resolve_word_align(请求 > config.word_align_enabled 兜底)` 在 `task_manager.create_task` + 分片 session 解析成 effective bool 写进 `TranscribeOptions.word_align`，transcribe/cache/metadata **全读它一个字段**（不再各自读 config）。**transcribe 读 `options.word_align` 而非 `self.word_align_enabled`**。
   - **CUDA 显存 + fallback（决策 2A-CQ/A/4A）**：CUDA word_align session 显存高水位常驻（3060 batch>=2 撞 BFCArena/CUBLAS OOM），故 `word_align_cuda_batch_size` 锁死 1（CPU 仍 16）。`_word_align_segments` 封装 **CUDA OOM → poison pool（`Qwen3DiarizeTranscriber._cuda_word_align_poisoned` class attr 进程/pool 级共享）+ dispose CUDA session（打 nvidia-smi delta，不当保证）+ 转 CPU（batch=16）重试**；CPU 也失败→段不带词。poison 后该进程余生 word_align 直走 CPU，重启恢复。资源错误判定 `is_resource_error`（BFCArena/CUBLAS）**穿透** `align_chunks` 逐窗 catch 才能触发 fallback（普通逐窗错误仍跳过）。
   - **缓存折维（决策 2A/B/C）**：`compute_cache_engine` 收 `output_format`，有效 word_align = `enabled AND json`（**SRT 即使请求 word_align=true 也降回裸 tag**，该行无词）；**对齐全失败→`cache_save_engine_for` 降 +wa 存 base tag**（不毒化文件）；`get_cached_result` 跨引擎回退 `engine NOT LIKE '%+%'` 排除折维行（防反向污染）。
   - **metadata（决策 2A/codex #12）**：`metadata.word_align` 反映**实际交付**（qwen3 AND options.word_align AND json AND words 实际挂上），失败附 `word_align_error`。
   - 语言来源：per-request `language` 字段（ISO 码 chi/eng/jpn/kor…）> config `word_align_language` 兜底。逐 window fallback：某 chunk 对齐失败→该段 words=None，段照常出，stats 记失败数。RTF 代价：Mac CPU +0.166（+17%），RTX 3060 CUDA 仅 +0.011（~1%）。模型预下到 `word_align_model_path`（`scripts/download_qwen3_models.sh --word-align`）。WordAligner per-worker 单例（primary + CPU fallback 两实例）。见 `spikes/qwen3_word_timestamp/SUMMARY.md`
   - **显存安全（TODOS #17 preflight + #18 sidecar 已落地，定案: `docs/开发/gpu加速/2026-06-16-word-align显存安全-评审定案与落地计划.md`）**：
     - **#17 VRAM preflight（Lane 1）**：`src/core/gpu_mem.py:free_vram_mib()`（nvidia-smi 读 free，尊重 `CUDA_VISIBLE_DEVICES`，探不到→None）+ `has_headroom()`。`_word_align_segments` 在加载 CUDA aligner **前**探显存，`free < word_align_preflight_free_mib`（默认 4608，env 可覆盖）→ 直走 CPU（不等 OOM）；探不到/preflight 关/非 CUDA → 不误杀照走（**preflight 不替代 OOM fallback**，TOCTOU codex #11）。`used_vram_mib()` 给 poison dispose delta。
     - **#18 CUDA sidecar（Lane 2，仅 cuda runtime/sidecar_enabled）**：`src/core/qwen3/word_align_sidecar.py` —— CUDA word_align 拆**长驻独立进程**，idle TTL（`word_align_sidecar_idle_ttl_sec` 默认 90s）无请求自杀**真正释放 VRAM**（ORT BFCArena 唯进程退出可还）。**Unix domain socket** request/response（length-prefixed JSON，per-PID `/tmp` socket），传 audio_path + chunks JSON（sidecar 自读文件）。**瘦入口**只 import `word_align.py` + `audio_io.py`（不拖 sherpa/ASR，codex #9）。client `WordAlignSidecarClient` 进程全局单例（codex #6），lazy spawn/复用/respawn/超时杀（codex #4 杜绝双跑）/OOM 退休（codex #8）。`_word_align_via_sidecar` 降级链：preflight 不足 / sidecar 超时·资源错误·不可用 → 主进程 CPU；普通对齐错误 → 无词；**cuda runtime 下进程内永不建 CUDA session**（codex #7 硬切）。`provider=cuda_sidecar`。`word_align_sidecar_enabled=false` 退回 #17 进程内路径。
     - 音频加载抽到 `src/core/qwen3/audio_io.py:load_audio_mono_16k`（DRY，diarize/sidecar 共用，`diarize._load_audio_mono_16k` 为别名）。
5.7. **nospk 分层切段**（`nospk_split_enabled`，**仅 diarize=False 分支执行**）— `src/core/qwen3/segment_split.py:split_long_segments`：diarize=false 没有 turn 边界、段 = ASR ~40s chunk，对超过 `nospk_split_max_segment_sec`（默认 12s）的段做两层 fallback 切分：有 `segment.words`（word_align 挂的）→ 词隙中点精确切；无 words → 静音中点切（**SRT 路径因 word_align JSON-only 永远走静音 fallback**，设计内不变量 T-D #5）；无候选硬切目标等分点保 max 上界。文本归属沿用现役 char-ratio 机制（标点优先）逐字无损，`nospk_split_min_segment_sec`（默认 1.5s）兜底最小片时长（吸收 short_segment_guard 的通用清理职责）。transcriber 薄 wrapper `apply_nospk_split_to_segments` 照 silence_align 形状（关/空/异常→fallback to input），挂 word_align 之后
6. **`relabel_segments_by_duration_desc`**（commit ceb9fa1，**无 config flag**；diarize=True 无条件执行，diarize=False 分支跳过）— 输出层 Speaker ID 稳定化：把内部 raw cluster int 按 speaker 总时长降序重映射成 0/1/2/…，让下游 `f"Speaker{i+1}"` 渲染出的 **Speaker1 始终是说话最多的主说话人**；底层 raw cluster int 由算法决定（ort_cuda / sherpa 互不兼容、跨平台不一致），不重映射会让客户端看到 "Speaker7" / "Speaker55" 这种漂移编号。平局按原 int 升序 tie-break 保证 deterministic。实现见 `src/core/qwen3/merge.py:relabel_segments_by_duration_desc`

加新后处理层：照 `apply_silence_align_to_segments` 的 helper 形状（`(enabled / 空 / 异常)→fallback to input`），挂到 transcribe 流程内同时给 stats 日志，配 5 个 config 字段就行。

## diarize 开关（2026-06-10 落地，定案: `docs/开发/2026-06-10-diarize开关API-设计定案.md`）

**API 语义引擎无关，实现策略引擎相关**：`FileUploadRequest.diarize: bool = True`，`diarize=false` ⇒ 响应不含说话人区分（JSON `speaker=null` + `speakers=[]`，SRT 无 `SpeakerN:` 前缀），默认 true 完全向后兼容。

- **options 穿透（E3/D1）**：`language` + `diarize` 收进 `TranscribeOptions`（`src/models/schemas.py`），`TranscriptionTask.options` 嵌套（平铺 language 已删）。整体穿透 schema → handler（含分片 session 回填）→ task_manager → 两套 pool → worker → `transcribe(options=...)`。file-based pool 用 `model_dump()` 写 .task JSON，worker 解析回 TranscribeOptions（老任务文件平铺 language 兜底）。**funasr 任务文件不写 options**（协议钉测试）。
- **qwen3 = 真跳层（D2/D5）**：跳 diarize + filter_spurious / cluster_merge / short_guard / relabel，段来自 ASR ~40s chunk，超长段走 nospk 分层切段（pipeline 5.7 层）。**内部 `Segment(speaker:int)` 永不为 None**，null 只在出口转换层出现。
- **funasr = 出口投影（D4）**：cam++ 提取无 per-call 开关，照算后由 serve 层投影抹 speaker；**缓存免折维**——存一行 diarized，serve 时按需投影，一行通吃两种请求。
- **缓存折维（D9）**：`compute_cache_engine` 字符串折维，顺序固定缺省不写：`qwen3` / `qwen3+wa:<lang>` / `qwen3+nospk` / `qwen3+wa:<lang>+nospk`（2 维 4 形态；**维度 >3 升级结构化 variant**）。折维 tag 一律禁 cross-engine。折维参数收拢在 `database.cache_params_for(task)` / `cache_params(engine, options)`，**不要手写**。
- **投影双出口（D3+T-A）**：纯函数 `src/core/result_projection.py`（`project_result_nospk` + `merge_segments_view` + `segments_to_srt_text` + `build_result_metadata`）。出口 1 = `db_manager.get_cached_result`（exact `+nospk` miss → 同引擎同 wa-tag diarized 行现场投影，标 `projected:true`，**不回写**——缓存永远只存真算结果；SRT nospk 旁路 funasr raw 路径从投影 segments 重渲染）；出口 2 = task_manager fresh 结果（缓存先存引擎真算结果再投影）。坏行具名异常当 miss + warn，禁 catch-all。
- **funasr segment 合并视图（issue #1）**：引擎/缓存存句级；JSON 出口（fresh + 缓存命中，`cached_engine == funasr`）`merge_segments_view`（gap + max_span cap）；**先 merge 后 nospk**。nospk SRT 旁路不过 merge（句级渲染）。config：`segment_merge_gap_sec=3.0` / `segment_merge_max_span_sec=120.0`。
- **metadata 回显（E2）**：`TranscriptionResult.metadata = {engine, diarize, word_align, language, projected}`（+ word_align 请求但失败时附 `word_align_error`；funasr+JSON 附 `segment_merge_max_span_sec`），serve 层组装（fresh 出口 + 3 个缓存命中出口），**save_result exclude 不入库**（projected 是请求级属性）。合并优先级：request > 分片 session 回填 > config > 引擎默认。`word_align` 反映**实际交付**（delivered: qwen3 AND options.word_align AND json AND words 实际挂上），非"请求想要"。
- **可观测性**：per-task 日志带 diarize 生效值；`db_manager.projected_serves` 计数进 `get_cache_stats()`；切段 stats 进 raw_result.nospk_split + 日志。
- **部署顺序**：老 server Pydantic 忽略未知 diarize 字段 → **server 先升级、客户端后启用**（部署假设有单测钉死）。
- **NOT in scope**：mode 三档枚举 / num_speakers per-request（TODOS #15，diarize=false 时闲置打 info 日志）/ funasr 启动不加载 cam++（TODOS #16）/ 词级替换式 merge（TODOS #14）。

### ⚠️ Qwen3 worker audio 转换的边界

`src/core/qwen3_worker_process.py` 对 sherpa-supported 格式（`{.wav, .flac, .ogg, .mp3, .opus}`）**跳过** ffmpeg `convert_to_wav`，仅 m4a/aac/mp4/mov/webm 走转码（sherpa libsndfile/librosa fallback 读不了）。**不要为了"统一"把所有格式都 ffmpeg 转 wav** — 调研（`docs/开发/archive/spk-over-detect-归因调研结果.md`）证明 ffmpeg→wav 即使保持 16kHz mono 也会改变 audio 字节，通过 pyannote+TitaNet embedding 放大，触发 sherpa diarize FastClustering over-detect（60min-2spk 真实场景 → 4 spk）。

