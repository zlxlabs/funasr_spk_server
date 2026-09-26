# 交接 prompt — TODOS #17 + #18 word_align 显存安全课题

> 新 session 直接读本文件开工。本文件自洽,不依赖任何历史对话上下文。
> 配套阅读(可选): `docs/开发/gpu加速/2026-06-16-Qwen3-word-align显存PoC与落地计划.md`(显存 PoC 全数据 + 评审结论)。

## 一句话任务

让 Qwen3 word_align(词级时间戳,MMS-300M CTC-FA ONNX)在共驻 3060 上**显存安全**。两件事 **#17 VRAM preflight 探针** + **#18 独立 sidecar 进程**——**当一个课题设计,分两个增量发**(见下「课题边界与发布策略」)。

## 课题边界与发布策略(重要,先看)

#17 和 #18 是同一个目标(word_align 显存安全)的两层,**一次设计、分两次发**:

- **为什么一起设计**:同一子系统、同一批文件;**#17 的显存探针就是 #18 sidecar 路由器的内部组件**——sidecar 也要先决定「拉 CUDA sidecar 还是走 CPU」,这个决定正需要 `free_vram_mib()` + 阈值 gate。接口只想一遍,别分两次重复加载心智。**用一次 `/plan-eng-review` 把 probe 接口 / 阈值 / sidecar 协议 / 生命周期 / 降级链一起过。**
- **为什么分两次发**(make the change easy, then make the easy change):
  - **Lane 1 = #17**:探针 `free_vram_mib()`(NVML 优先,退化 nvidia-smi)+ **进程内 preflight gate**(加载 CUDA word_align session 前探显存,不够→走 CPU,不等 OOM 后才 fallback)。小、低风险、几小时可 ship,**立刻降当前 OOM 风险**。
  - **Lane 2 = #18**:sidecar 进程,**复用 Lane 1 的探针**(不返工——探针是共享原语,sidecar 路由器调它)。大、结构性(新进程 + IPC + 生命周期),设计落地都重。
  - **附带好处**:#17 单独发后能用数据验证「#18 到底急不急」——若 preflight + CPU 降级后 OOM 基本绝迹,#18 可降优先级;若仍常撞顶,再上 sidecar 有数据支撑。
- **共享原语**:`free_vram_mib()` / 阈值常量 / runtime gate 两条 lane 共用,Lane 1 建好 Lane 2 直接复用。现有 `src/core/qwen3_transcriber.py:_gpu_mem_used_mib()`(目前仅 poison 时日志观测用)可升级成正式探针。

### #17 preflight 起步阈值(需二次标定,非最终)
- CUDA word_align session **未加载**:要求 free VRAM ≥ ~4.5GB(session ~3.4GB + 余量)。
- CUDA word_align session **已加载**:要求 free ≥ ~1.5-2GB 再接新 CUDA word_align。
- 低于阈值 → 不加载/不路由 CUDA,走 CPU。preflight 在 CUDA 执行**前**做,不要等 OOM 后 fallback(OOM 可能已把 CUDA 上下文弄不健康)。

## #18 sidecar 一句话

把 word_align 从主 ASR 服务进程拆成**独立 sidecar 进程**:按需启动 CUDA session,空闲一段(idle TTL)后**退出进程真正释放 VRAM**,主服务通过 RPC/队列调用,不可用时降级 CPU 或返回无词。

## 为什么要做(已实测确认的痛点)

word_align 的 CUDA ONNX session 一旦加载,显存**赖着不走**——这是 ORT BFCArena 高水位分配器的特性,**唯一可靠的释放是进程退出**。

2026-06-16 在 3060 dev box(`ws://<dev-host>:8867`,与 CapsWriter 2.86GB 同卡共驻)实测,同一段 60s 音频连续跑 word_align 任务:

```
funasr 进程显存(MiB):
  加载 word_align 后基线  6234
  +2 任务               6790
  +5 任务               7844  ← 阶梯跳一档
  +6/7/8 任务           7844  ← 封顶, 不再涨
  (重启服务才回落到 idle 2110)
整卡 12288, 封顶时仅剩 ~1200 MiB 空闲(还要养 CapsWriter 2864)。
```

结论:同尺寸输入会**阶梯爬升到一档封顶**(不无限涨),但封顶在高位、余量薄。**更长音频会顶出更高封顶**,在这台 12GB 卡 + 同驻 CapsWriter 上很容易 OOM。当前主服务的 word_align session 永久占住 ~3.4-5.7GB baseline,直到重启。sidecar 让偶发的词级时间戳请求用完即退,主 ASR 服务长期 baseline 不被顶高。

## 已经落地的东西(不要重做,sidecar 是它们的上位替代)

per-request word_align 开关 + 进程内显存兜底已在 main 落地(commit `ed4c107` 及其之前),已过 3060 真机 e2e。已有的进程内兜底:

- **CUDA OOM → poison + dispose + 转 CPU**:`src/core/qwen3_transcriber.py:_word_align_segments` + `_poison_cuda_word_align`。第一次 CUDA 资源错误(`is_resource_error` 匹配 BFCArena/CUBLAS)后,`Qwen3DiarizeTranscriber._cuda_word_align_poisoned`(class attr,进程/pool 级)置位,该进程余生 word_align 走 CPU,并 `WordAligner.close()` + gc 尝试回收显存(打 nvidia-smi delta,**不保证真还**——这正是 sidecar 要解决的)。
- **per-request 开关**:`FileUploadRequest.word_align: Optional[bool]` → `resolve_word_align` → `TranscribeOptions.word_align`(决策 1A,`src/models/schemas.py`)。
- **cuda batch 锁死 1**:`Qwen3Config.word_align_cuda_batch_size=1`(batch>=2 在 3060 撞 OOM)。

sidecar 落地后,进程内 poison/dispose 这套「band-aid」可以保留作为 sidecar 不可用时的次级兜底,或简化。

## TODOS #18 原始条目(`TODOS.md`)

```
### 18. 独立 word_align sidecar 进程
- What: word_align 从主 ASR 服务拆独立进程, 按需启动 CUDA session, 空闲 TTL 自动退出
  释放 VRAM; 主服务通过队列/RPC 调用, 不可用时 CPU fallback 或返回无词; 加生命周期日志
  + 健康检查
- Why: poison+dispose 只能"尝试"要回显存, ORT BFCArena 高水位未必真还(文档存疑)。
  唯一可靠的显存释放是进程退出
- Cons: 跨进程协议 + 生命周期管理 + 健康检查, 工程量最大; 主服务多一个故障面
- 依赖: word_align per-request PR(已落地)+ TODO #17(VRAM preflight)
- 优先级: P3
```

## 关键架构 / 文件指引

- **word_align 引擎**:`src/core/qwen3/word_align.py` — `WordAligner`(per-worker 单例,`align_chunks(audio_16k, asr_chunks, language)` → `(words, stats)`)、`build_alignment_session`、`is_resource_error`、`resolve_word_align_providers`。MMS 模型 `models/qwen3_diarize/ctc_forced_aligner/model.onnx`(~1.2GB)。
- **调用点**:`src/core/qwen3_transcriber.py:_word_align_segments`(同步,在 executor 跑;输入 audio_path + asr_result.chunks + segments + effective_lang,输出挂了 words 的 segments + stats)。这是 sidecar 边界的天然切口——把这个函数的内部换成「调 sidecar」。
- **池 dispatch**:`src/core/qwen3_pool_transcriber.py:get_qwen3_pool_transcriber()` 按 runtime 分发:
  - **cuda → `Qwen3InProcPool`**(`qwen3_inproc_pool.py`,单进程内 N 实例)← **sidecar 主要针对这条**(进程不退,显存赖着)。
  - 其他(Mac/CPU)→ `Qwen3PoolTranscriber`(file-based 多进程,worker 跑完即退)← **codex #9:Mac worker 本就每 task 退出,不需要 sidecar**。
- **runtime 抽象**:`src/core/runtime.py:detect_runtime()` / `recommend_word_align_provider()`。
- **config**:`src/core/config.py:Qwen3Config` 的 `word_align_*` 字段 + `PROFILES`。加 sidecar 配置照「加 config 字段的步骤」(CLAUDE.md)。
- **现成 e2e 工具**:`scripts/_remote_word_align_probe.py`(5 场景验收)、`scripts/_remote_word_align_oom_driver.py`(多轮逼 OOM)。

## 必须先决策的设计点(开工前用 AskUserQuestion 跟用户确认,先讲大白话背景再给推荐)

> 注:「#17 与 #18 是否合并」**已定**——一次设计、分两次发(见上「课题边界与发布策略」),不再作为开放决策。下面 4 项是 sidecar(Lane 2)的设计 fork,在那一次 `/plan-eng-review` 里过;Lane 1(#17 探针 + preflight gate)结构简单,起步阈值见上,可先快速落地。

1. **进程模型**:每请求 spawn 一个用完即退的子进程(最简单、显存最干净,但每次付 ~1-2s session build + ASR chunk 数据传输开销)vs 一个长驻 sidecar + idle TTL 到点自杀(省重复 build,但 idle 窗口内仍占显存)。推荐倾向:**长驻 + 短 idle TTL**(如 60-120s),兼顾首请求延迟和空闲释放。
2. **通信协议**:本机 HTTP(localhost,简单、可健康检查)vs Unix domain socket vs stdin/stdout pipe vs 文件队列(沿用 `FileBasedProcessPool` 既有范式)。数据量:传 audio(可传路径,sidecar 自己读文件,避免大 base64)+ ASR chunks(JSON)+ language,回 words + stats。推荐倾向:**传 audio_path(不传字节)+ chunks JSON**,协议用本机 HTTP 或复用 file-based pool 范式。
3. **降级链**:sidecar 不可用 / 启动失败 / 超时 → 主进程 CPU 兜底(复用现有 `_ensure_word_aligner_cpu`)还是直接返回无词 + `word_align_error`?推荐:**sidecar 失败 → 主进程 CPU → 仍失败才无词**。
4. **作用域**:**只对 cuda runtime / `Qwen3InProcPool` 启用 sidecar**(Mac file-based pool worker 每 task 退出,显存本就释放,codex #9)。runtime gate 清楚。

## 工程约束(本项目硬规矩,见 CLAUDE.md)

- **严格 TDD**:先红再绿再 commit,红→绿→commit 是最小单位,不要积累多改动一次性提交。
- **改 FunASR 路径(schemas/database/task_manager/websocket_handler/funasr_transcriber)后必跑 parity**:`FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/`。sidecar 主要动 qwen3 路径,但若碰 schema/options 要跑。
- **测试三层**:`tests/unit/`(mock,毫秒)、`tests/integration/`(真模型,默认 skip 需 `FUNASR_RUN_INTEGRATION=1`)、`tests/manual/`(不收集)。sidecar 协议/生命周期/降级用 unit mock;真 CUDA 释放用 3060 手工 probe。
- **不重复造轮子**:进程管理/RPC 优先成熟方案(`multiprocessing`、`subprocess`、本机 HTTP `aiohttp`/`uvicorn` 已在依赖里?先查),不手搓。
- **低耦合高内聚 + 完备日志 + 注释**。

## 3060 dev box 信息(真机验证用)

- `ssh <user>@<dev-host>`,仓库 `~/Dev/projects/funasr_spk_server`,`FUNASR_PROFILE=cuda_dev`,端口 8867,systemd `funasr-server.service`(`sudo systemctl restart`),日志 `logs/server_cuda.log`(不是 journald)。
- 同卡共驻 CapsWriter(~2.86GB)+ samapi(docker,需要腾显存时 `docker stop samapi`)。**别 OOM 轰共驻服务**。
- 重启服务可把 word_align 显存压回 idle 基线(~2110 MiB)。
- 真机跑验收:`venv/bin/python scripts/_remote_word_align_probe.py --server ws://localhost:8867 --audio tests/fixtures/audio/podcast_2speakers_60s.wav`。

## 验收标准

**Lane 1(#17 探针 + preflight gate)**
- [ ] `free_vram_mib()` 探针(NVML 优先 / nvidia-smi 退化 / 非 CUDA 返回 None),unit 覆盖三路。
- [ ] 加载 CUDA word_align session 前 preflight:free 低于阈值 → 走 CPU(不等 OOM),unit 覆盖。
- [ ] 阈值标定:3060 真机记录加载前/后 free,确认阈值不会误杀也不放过 OOM。
- [ ] 行为可观测:每请求前后 VRAM delta 进日志。

**Lane 2(#18 sidecar)**
- [ ] word_align=true 请求经 sidecar 产出词级时间戳,与进程内 CUDA 结果**一致**(parity)。
- [ ] sidecar 空闲 TTL 到点退出后,**3060 整卡显存回落到主服务 idle 基线**(nvidia-smi 实测,这是 #18 的核心价值)。
- [ ] 主服务长期 baseline 不再被 word_align 永久顶高(连跑多个 word_align 任务,空闲后显存回落)。
- [ ] sidecar 不可用/超时 → 降级链生效(CPU 或无词 + `word_align_error`),主请求不挂。
- [ ] sidecar 路由器复用 Lane 1 的 `free_vram_mib()` 决定走 CUDA sidecar 还是 CPU(不重复造探针)。
- [ ] Mac/CPU runtime 不启用 sidecar(行为不变,parity 绿)。
- [ ] unit 覆盖协议/降级/生命周期;3060 手工 probe 证明显存真释放。
- [ ] 文档同步:CLAUDE.md pipeline 5.5 word_align 节 + 部署文档 + TODOS.md #17/#18 标完成。

## 开工第一步建议

1. 读本文件 + `2026-06-16-Qwen3-word-align显存PoC与落地计划.md` + `src/core/qwen3_transcriber.py:_word_align_segments`/`_gpu_mem_used_mib` + `src/core/qwen3/word_align.py`。
2. 用一次 `/plan-eng-review` 把整个显存安全课题过一遍(probe 接口 / 阈值 / sidecar 4 个设计 fork / 降级链),把上面 4 个 sidecar 决策跟用户敲定(先大白话背景再推荐)。
3. **先 ship Lane 1(#17)**:TDD 落地探针 + preflight gate(小、独立、立刻降 OOM 风险),3060 标阈值。
4. **再 ship Lane 2(#18)**:sidecar 复用 Lane 1 探针,先 unit-mock 协议/降级,再上 3060 验真显存释放。
5. Lane 1 上线后用真机数据复评 #18 优先级(若 OOM 基本绝迹可降级慢做)。
