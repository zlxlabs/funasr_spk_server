# word_align 显存安全（TODOS #17 + #18）— 评审定案与落地计划

**Date**: 2026-06-16
**评审**: `/plan-eng-review`（Step0 范围挑战 + 架构/代码/测试/性能 + codex 外部声音 15 findings）
**课题**: word_align（MMS-300M CTC-FA 词级时间戳）在共驻 3060 上显存安全
**前置**: per-request 开关 + poison + CPU fallback 已落地（commit `ed4c107`），本计划是其**上位替代**
**配套阅读**: `2026-06-16-TODOS17-18-word-align-显存安全-交接.md`（课题边界）、`2026-06-16-Qwen3-word-align显存PoC与落地计划.md`（显存 PoC 全数据）

> 本文是评审**输出**：锁定的设计决策 + 测试覆盖 + 失败模式 + 落地顺序。实现照此执行，不再重议已锁项。

## 实现状态（2026-06-16 落地）

**Lane 1 (#17) + Lane 2 (#18) 代码 + 单测已落地**（严格 TDD，红→绿→commit）。剩 3060 真机 E2E（显存真释放 + 阈值标定）为最终手工验收。

| 件 | 文件 | 单测 | 状态 |
|---|---|---|---|
| 探针 | `src/core/gpu_mem.py` | `test_gpu_mem.py`（14）| ✅ |
| preflight gate | `qwen3_transcriber._word_align_segments` | `test_..._word_align_preflight.py`（5）| ✅ |
| preflight config | `config.py` | `test_config_word_align_preflight.py`（4）| ✅ |
| sidecar 协议+server | `qwen3/word_align_sidecar.py` | `test_word_align_sidecar_protocol.py`（9）| ✅ |
| sidecar client | 同上 | `test_word_align_sidecar_client.py`（9）| ✅ |
| sidecar 入口 | 同上 `run_sidecar` | `test_word_align_sidecar_entry.py`（3）| ✅ |
| sidecar config | `config.py` | `test_config_word_align_sidecar.py`（6）| ✅ |
| 集成路由+降级 | `qwen3_transcriber._word_align_via_sidecar` | `test_..._word_align_sidecar.py`（8）| ✅ |
| audio_io 抽取 | `qwen3/audio_io.py` | `test_qwen3_diarize_audio_fallback.py`（2）| ✅ |

全 unit 682 passed（6 失败为既有 `.env` 污染，与本课题无关）。

**3060 真机 E2E 验收（2026-06-16 实测，全通过 ✅）**

dev box `<dev-host>`（cuda_dev，funasr + CapsWriter 2864MiB 共驻 12GB 卡）：

| 验收项 | 结果 |
|---|---|
| sidecar idle TTL 退出**真释放 VRAM** | ✅ 日志 `idle 90.0s 无请求, 退出释放 VRAM`；整卡 used **9780→6579（释放 3201 ≈ sidecar 持有的 3198）** |
| word_align 经 sidecar 出词 parity | ✅ 359-360 词，`metadata.word_align=true`，无 `word_align_error`，RTF 0.11；probe 5 场景（A-E）全过 |
| 主服务 baseline 不被 word_align 永久顶高 | ✅ word_align 的 ~3198MiB **全在独立 sidecar 进程**（`python -m src.core.qwen3.word_align_sidecar`），主 funasr 进程仅 3704（ASR/diarize 工作集）。**对比**：落地前进程内 CUDA word_align 会把主进程顶到 5956-6212MiB 且只有重启才回落 |
| 生命周期：lazy spawn / idle 退出 / 重启 | ✅ 首请求 lazy spawn；idle 90s 自退；下次请求 `poll()` reap 旧进程 + 重启新 sidecar |
| preflight 放行 + 无 OOM | ✅ free 6936 > 4608 阈值放行；CapsWriter 毫发无损 |

**阈值标定**：3060+CapsWriter 上 sidecar word_align CUDA session ~3198MiB，idle 基线 funasr 2110MiB。起步阈值 `word_align_preflight_free_mib=4608`（~session+1.4GB 余量）真机**既不误杀也没放过 OOM**，保留为验证值（可按需 env 收紧到 ~3700）。

**已知小行为（可接受）**：sidecar idle 自退后短暂变僵尸进程（`Z <defunct>`，**不占 GPU/内存**），下次 word_align 请求经 `_alive()→poll()` reap + respawn；"自退后再无请求"会留一个无资源僵尸到主进程退出（atexit 兜底）。不加 reaper 线程（无资源、下次请求即清，over-engineering）。

剩余（未在本轮真机覆盖，低优先）：sidecar 主动杀（超时/OOM 退休）路径的真机触发——单测已覆盖（mock），真机靠自然 OOM 难构造，留观测。

---

## 0. 一句话

word_align 的 CUDA ONNX session 一旦加载，显存赖着不还（ORT BFCArena 高水位，唯进程退出可释放）。两层防护、**一次设计、分两次发**：
- **Lane 1（#17）**: VRAM preflight 探针 + gate — 加载 CUDA 前探显存，不够直接走 CPU，事前预防（小、低风险、先发）。
- **Lane 2（#18）**: word_align CUDA 拆**长驻 sidecar 进程**，idle TTL 到点退出**真正释放显存**（大、结构性、后发，复用 Lane 1 探针）。

---

## 1. 锁定的设计决策（实现按此，不再重议）

| # | 决策 | 锁定内容 | 依据 |
|---|---|---|---|
| **A1** | sidecar 进程模型 + IPC | **长驻进程 + idle TTL（60–120s）自杀释放显存**；通信用 **Unix domain socket**（stdlib `socket` / asyncio `open_unix_connection`，**零新依赖**），request/response，传 `audio_path` + ASR chunks JSON（不传字节）。**不复用 `FileBasedProcessPool` 类**——其"每任务即退 + 自动 respawn + 启动前清文件 + 固定目录"语义与长驻 idle-TTL 拧着（codex #1/#2/#3/#5/#10/#15）。 | 用户 + codex |
| **A2** | 探针实现 | `free_vram_mib()` **只用 nvidia-smi**（扩展现有 `_gpu_mem_used_mib` 读 `memory.free`），**不引 NVML**（探针是冷路径，每请求一次，子进程开销相对多秒对齐可忽略；NVML 是为零收益花创新额度）。**必须尊重 `CUDA_VISIBLE_DEVICES` / ORT 设备选择**，不读死 nvidia-smi 第一行（codex #12）。非 CUDA / 探测失败 → 返回 `None`，调用方降级"未测、不误杀"。 | 用户（偏离交接文档"NVML 优先"）+ codex #12 |
| **A3** | 降级链 + CPU 兜底位置 | sidecar **只做 CUDA**（单一职责 = 隔离 CUDA 显存）；CPU 路径无显存问题，**留主进程**复用现有 `_ensure_word_aligner_cpu`。降级链：探针够 → CUDA sidecar；探针不足 / sidecar 失败/超时 → 主进程 CPU；CPU 失败 → 无词 + `word_align_error`。**超时必先杀 sidecar（顺带释放显存）再起 CPU**，杜绝 CUDA+CPU 双跑（codex #4）。 | 用户 + codex #4 |
| **A4** | 作用域 | **仅 CUDA runtime / `Qwen3InProcPool` 启用 sidecar**。Mac MPS worker 每 task 即退（显存自然释放）、CPU 路径不碰显存 → 行为 100% 不变，runtime gate 死框回归面。 | 用户 + codex #9 |
| **P1** | 阈值承载 | 阈值进 `Qwen3Config` 字段（`_override_if_set` env 可覆盖 + PROFILES），**保守默认**（未加载 free ≥ 4.5GB / 已加载 ≥ 1.5–2GB），Lane 1 上 3060 真机标定后改准默认。启动日志打印。换卡 / 调 pool_size 只改 env。 | 用户 |

### codex 9 条加固（全折进）

| codex# | 加固 | 落地点 |
|---|---|---|
| #4 | 超时杀 sidecar 再 CPU，杜绝双跑 | A3 降级链 |
| #6 | sidecar 是**进程全局单例**（不挂 transcriber 实例），`pool_size>1` 也只一个 CUDA session | sidecar manager 单例 |
| #7 | Lane 2 后 cuda runtime 下 `_word_align_segments` **不再可达** `_ensure_word_aligner()` CUDA 分支（测试钉死） | 切口改造 |
| #8 | sidecar 模式 OOM = **杀/退休 sidecar（冷却期）**，不再永久 poison 主进程（否则 TTL/重启后仍禁用 sidecar） | poison 语义迁移 |
| #9 | sidecar **瘦入口**：只 import `qwen3/word_align.py` + 极简音频加载，不碰 `qwen3_transcriber.py`（避免拖入 ASR/diarize 机器） | sidecar 模块边界 |
| #11 | preflight 是 **TOCTOU**（探完到建 session 之间显存可能被 CapsWriter/ASR 吃掉）→ **OOM fallback 仍必需**，preflight 只挡明显失败 | plan 显式不变量 |
| #12 | `free_vram_mib()` 尊重 `CUDA_VISIBLE_DEVICES` | A2 |
| #13 | "已加载阈值"太糙（MMS 显存随输入 shape 爬）→ 平阈值 + OOM fallback 兜底；**duration/chunk-aware 阈值收 TODO #19**，数据证明需要再做 | TODO #19 |
| #14 | 缓存 `+wa` 降维看 **has_words（真挂上词）**，非"sidecar 返回成功" | 集成不变量 |

---

## 2. 数据流 / 降级链 / Lane 边界（ASCII）

### 2.1 word_align 路由 + 降级链（Lane 2 终态，cuda runtime）

```
word_align=true 请求 (qwen3, JSON, cuda runtime)
        │
        ▼
  _word_align_segments  (主进程切口, executor)
        │
        ├─ free_vram_mib()  ── None(探不到) ─┐
        │      │                            │
        │   够阈值                       不足阈值
        │      │                            │
        │      ▼                            ▼
        │  ┌─────────────────────┐    主进程 CPU aligner
        │  │ CUDA sidecar (UDS)  │    (_ensure_word_aligner_cpu)
        │  │ 进程全局单例/长驻    │          │
        │  │ idle TTL 自杀释放VRAM│      成功│失败
        │  └─────────────────────┘          │   │
        │      │成功   │OOM/超时/连不上       │   │
        │   挂词✓   杀 sidecar               挂词✓ 无词+error
        │           (释放显存) ──► 主进程 CPU ──► (同上 成功/失败)
        │
        ▼
  has_words? ── 真挂上词 → 缓存 +wa / metadata.word_align=true
             └ 无词      → 不写 +wa(不毒化) / word_align_error
```

不变量（codex #11）：**preflight 不替代 OOM fallback**。TOCTOU 窗口内仍可能 OOM，poison/CPU 兜底永远保留。

### 2.2 Lane 1 → Lane 2 的 gate 迁移（探针是共享原语，非一次性前置）

```
Lane 1 (#17) 终态:                      Lane 2 (#18) 终态 (cuda runtime):
┌───────────────────────────┐          ┌───────────────────────────────┐
│ 主进程 _ensure_word_aligner│          │ 主进程: 不再构造 CUDA aligner   │  ← codex #7 硬切
│  ├ free_vram_mib() gate    │   ──►    │ sidecar router:                │
│  │  够→建 CUDA aligner      │  探针    │  ├ free_vram_mib() gate (复用!) │
│  │  不足→建 CPU aligner     │  迁移    │  │  够→路由 CUDA sidecar        │
│  └ CPU fallback (主进程)    │          │  │  不足→主进程 CPU             │
└───────────────────────────┘          │  └ CPU fallback (主进程, 不变)  │
   探针 free_vram_mib() ───────共享──────┘ └───────────────────────────────┘
   (src/core/gpu_mem.py, 两 lane 同一份)
```

Lane 1 的 gate 不是白写：探针 `free_vram_mib()` + 阈值常量是 Lane 2 sidecar router 的内部组件。gate **落点**从"主进程加载前"迁到"路由 CUDA sidecar 前"。

### 2.3 sidecar 生命周期状态机

```
        spawn (首个 CUDA word_align 请求, lazy)
          │
          ▼
   ┌─────────────┐  请求到达    ┌──────────┐
   │   IDLE      │ ───────────► │  BUSY    │
   │ (TTL 计时)  │ ◄─────────── │ (对齐中)  │
   └─────────────┘  完成,重置TTL └──────────┘
          │ TTL 到期(idle)            │ CUDA OOM
          ▼                          ▼
      self-exit                  回错误状态 → 主进程杀 sidecar
   (释放 VRAM, 干净)             (释放 VRAM) + 冷却期(codex #8)
          │                          │
          └──► 下个请求 lazy respawn ◄┘
```

UDS liveness：连不上 socket = sidecar 死 → 主进程 CPU 兜底或 lazy respawn。无半截文件、无目录撞车（socket 路径 per-PID，codex #15）。

---

## 3. 涉及文件 / 模块

| 文件 | Lane | 改动 |
|---|---|---|
| `src/core/gpu_mem.py` **(新)** | 1 | `free_vram_mib()`（nvidia-smi 读 free，尊重 CUDA_VISIBLE_DEVICES，None 兜底）+ 阈值判定 helper |
| `src/core/qwen3_transcriber.py` | 1→2 | Lane1: `_ensure_word_aligner` 加 preflight gate；迁走 `_gpu_mem_used_mib`。Lane2: `_word_align_segments` 切口 CUDA 分支改"调 sidecar"，硬切主进程 CUDA 路径（codex #7） |
| `src/core/qwen3/word_align_sidecar.py` **(新)** | 2 | sidecar 进程入口（UDS server）+ 主进程侧 client（进程全局单例 manager，lazy spawn / idle TTL / 超时杀 / liveness）。**瘦入口**：只 import `qwen3/word_align.py` + 极简音频加载（codex #9） |
| `src/core/config.py` | 1+2 | `Qwen3Config` 加阈值字段 + sidecar 字段（idle_ttl / enabled）+ `_override_if_set` + PROFILES + print_config |
| `src/core/runtime.py` | 2 | runtime gate helper（仅 CudaRuntime 启用 sidecar） |

新增服务 = 1（sidecar）。未触发复杂度 STOP（< 8 文件 / < 2 新服务）。

---

## 4. 测试覆盖（TDD 红→绿，实现时逐个补齐）

### Lane 1 (#17)
- `gpu_mem.free_vram_mib()`: nvidia-smi 正常解析 free / 不可用→None / 超时→None / 尊重 CUDA_VISIBLE_DEVICES（mock 多卡）/ 非 CUDA→None。**unit mock subprocess**。
- preflight gate: free≥阈值→建 CUDA / free<阈值→建 CPU（不等 OOM）/ probe=None→按现状不误杀。**unit mock probe**。
- VRAM delta 进日志。**unit 验日志**。

### Lane 2 (#18)
- sidecar entry: 收请求→CUDA align→回 words+stats / idle TTL 到→进程退出（fake clock unit + **[E2E 3060] 真显存回落**）/ CUDA OOM→回错误状态（不在 sidecar 内 CPU）。
- sidecar client/router: probe 够→调 sidecar / probe 不足→主进程 CPU / sidecar 超时→**杀 sidecar 再** CPU（mock 超时，验证不双跑）/ 连不上→CPU / **进程全局单例**（pool_size=2 只一个 sidecar，codex #6）。
- UDS 协议: 正常 round-trip / 半包/粘包 framing / 连接拒绝→死判定。**unit**。
- 切口硬切: cuda runtime 下 `_word_align_segments` **不可达** `_ensure_word_aligner()` CUDA 分支（codex #7）。**unit 断言**。
- 集成: has_words 语义保持（sidecar 成功但 0 词→不写 +wa，codex #14）。**unit**。

### REGRESSION（必加，无 AskUserQuestion）
- **[CRITICAL]** Mac/CPU word_align parity 不被 Lane1/2 改坏 → 现有 word_align e2e + `tests/integration` parity（`FUNASR_RUN_INTEGRATION=1`）。
- **[CRITICAL]** sidecar 空闲后**整卡显存回落到主服务 idle 基线**（~2110MiB）→ [E2E 3060] nvidia-smi 实测，这是 #18 核心价值。

### [E2E 3060] must-have
- word_align=true 经 sidecar 出词，与进程内 CUDA parity 一致。
- 连跑多个 word_align 任务，空闲后显存回落（主服务 baseline 不被永久顶高）。
- sidecar 挂掉 → 降级 CPU/无词，主请求不挂。

---

## 5. 失败模式（每条新路径：测试 / 错误处理 / 用户可见）

| 路径 | 失败方式 | 测试 | 错误处理 | 用户可见 | 评级 |
|---|---|---|---|---|---|
| free_vram_mib | nvidia-smi 缺/超时 | ✅ | None→不误杀 | 日志 | ok |
| 多 GPU | 读错卡 free | ✅(mock) | 尊重 CUDA_VISIBLE_DEVICES | 日志 | ok（codex#12 已堵）|
| preflight TOCTOU | 探完被抢→建 session OOM | ✅ | **OOM fallback 保留** | metadata error | ok（codex#11 不变量）|
| sidecar spawn 失败 | 起不来 | ✅ | →主进程 CPU | metadata | ok |
| sidecar 超时 | CUDA 卡住 | ✅ | **杀 sidecar 再 CPU**（不双跑）| metadata | ok（codex#4）|
| TTL 自杀 race | 退出 vs 新请求 | ✅ | UDS 连不上→respawn/CPU | 透明 | ok（UDS liveness）|
| pool_size>1 | 多 sidecar | ✅ | **进程全局单例** | — | ok（codex#6）|
| OOM 退休 | 永久禁 sidecar | ✅ | 杀+冷却，非永久 poison | 日志 | ok（codex#8）|
| 切口残留 | 主进程仍建 CUDA | ✅ | **测试钉死不可达** | — | ok（codex#7）|
| has_words | sidecar 成功 0 词→毒化缓存 | ✅ | 看真挂词，不写 +wa | 缓存正确 | ok（codex#14）|

**0 个 critical gap**（无测试 + 无处理 + silent 的路径）。codex 挑出的 race（TTL/双跑/单例/切口残留）均已被 UDS liveness + 杀进程 + 单例 + 测试钉死覆盖。

---

## 6. 落地顺序（一次设计、分两次发）

```
Lane 1 (#17, 先发, 小/低风险):
  1. src/core/gpu_mem.py: free_vram_mib() + 阈值 (TDD)
  2. config.py: 阈值字段 + env + PROFILES + print_config
  3. qwen3_transcriber.py: _ensure_word_aligner preflight gate
  4. 3060 真机标定阈值 → 改默认值
  ── ship → 用真机数据复评 #18 优先级 (OOM 是否基本绝迹)

Lane 2 (#18, 后发, 复用 Lane 1 探针):
  5. word_align_sidecar.py: UDS sidecar entry (瘦入口) + client manager (单例/TTL/杀/liveness) (TDD unit-mock)
  6. qwen3_transcriber.py: _word_align_segments CUDA 分支→调 sidecar; 硬切主进程 CUDA 路径
  7. config.py + runtime.py: sidecar gate (仅 cuda runtime)
  8. 3060 真机验真显存释放 (idle 后回落 baseline) + parity
  ── ship
```

**worktree 并行**：Lane 1 内部基本顺序（都碰 gpu_mem/config/transcriber）。Lane 2 的 sidecar 模块（新文件）可与切口改造并行起草，但切口依赖 sidecar client 接口，建议先定 client 接口签名再并行。整体 **Lane 1 → 复评 → Lane 2** 顺序，lane 内顺序为主。判定：**接近顺序实现，并行收益有限**（共享 gpu_mem/config/transcriber 三文件）。

---

## 7. NOT in scope（明确推迟）

- **duration/chunk-aware 动态阈值**（codex #13）→ TODO #19。平阈值 + OOM fallback 先兜底，数据证明长音频常 OOM 再做。
- **NVML 探针**（交接文档原写 NVML 优先）→ 砍。冷路径不值得新依赖；未来探针变高频再议。
- **sidecar 跨机 / 多 GPU 调度** → 不做。单机单卡 word_align 足够。
- **Mac/CPU sidecar** → 不做（worker 即退 / 不碰显存，codex #9）。
- **HTTP/gRPC sidecar 协议** → 砍。UDS stdlib 够用，不引 server 框架。
- **sidecar 内 CPU 兜底** → 不做。CPU 无显存问题，留主进程（A3）。

## 8. What already exists（复用，不重建）

| 子问题 | 现成 | 复用方式 |
|---|---|---|
| 读显存 | `qwen3_transcriber.py:_gpu_mem_used_mib()`（nvidia-smi 读 used）| 迁 `gpu_mem.py`，扩展读 free + 多卡 |
| 降级链 | `_word_align_segments`（poison→CPU→无词，过真机 e2e）| sidecar 插同一切口，CPU 分支原样保留 |
| CPU aligner | `_ensure_word_aligner_cpu()` | A3 主进程兜底直接复用 |
| 子进程范式 | `FileBasedProcessPool`（per-task）| **不复用类**（语义错配）；只借鉴"子进程隔离"思路，IPC 改 UDS |
| has_words 缓存 | `database.cache_params_for` / task_manager `+wa` 降维 | 集成保持语义（codex #14）|

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | n/a（纯后端基础设施）|
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | ISSUES_FOUND | 15 findings：1 翻案 A1（FileBasedProcessPool 错配→改 UDS）+ 9 加固全折进 + 1 新 TODO #19 |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 6 决策锁定（A1-A4+P1+9加固），0 critical gap，1 issue→TODO #19 |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a（后端）|
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | n/a |

- **CODEX:** 外部声音翻案了 A1（"复用 FileBasedProcessPool 类"是长驻 idle-TTL sidecar 的错配），用户改判 **Unix domain socket**；另 9 条 race/边界加固全部折进。
- **CROSS-MODEL:** 1 个真张力（A1 IPC），已由用户拍板 UDS 解决；其余 codex findings 为加固，无冲突。
- **UNRESOLVED:** 0
- **VERDICT:** ENG CLEARED — 6 个设计决策 + codex 9 加固全部锁定，0 critical gap。可进入实现（Lane 1 → 复评 → Lane 2）。
