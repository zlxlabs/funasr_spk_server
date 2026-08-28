Always respond in 中文

# 代码设计上的要求
- 各部分功能尽量低耦合，高内聚。
- 各个函数代码做好注释。
- 有完备的日志系统，方便后期调试确认问题。
- 避免写冗余代码，提高项目的可复用性。
- 尽力遵循工程上的最佳实践。适当使用文件夹来增加整个项目的可读性，但不要添加过多无关文件。
- 及时完善 .gitignore 文件。

# 测试约定（PR1 之后启用）

## pytest 套件
- 测试目录三层：
  - `tests/unit/` — 单元测试（mock 外部依赖，毫秒级）
  - `tests/integration/` — 端到端集成（含真实 FunASR 模型）
  - `tests/manual/` — 历史手工脚本（**不在 pytest 收集范围**），仅作复现参考
- 推荐命令：`venv/bin/python -m pytest`（venv 的 pytest binary shebang 漂移，用 `-m pytest` 走 venv python）
- integration 默认 skip，需 `FUNASR_RUN_INTEGRATION=1` 才跑

## TDD 流程
- 新功能 / 改 bug 都先写测试再改代码
- 红 → 绿 → commit 是最小单位，**不要积累多个改动一次性提交**
- 每个 commit message 写清楚改了什么 + 解决什么问题

## Parity 测试
- 改动 FunASR 路径（schemas / database / task_manager / websocket_handler / funasr_transcriber）后必跑：
  ```
  FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/
  ```
- 通过 = 改动安全；失败 = 你改坏了 FunASR 路径，回去查
- golden baseline 在 `tests/fixtures/golden/`，由首次运行自动生成

# ASR 引擎架构

两个引擎并行接入生产，dispatch 走轻量函数路由（**不是 ABC 抽象**）：
- **Dispatch**: `src/core/transcriber_dispatch.py` 的 `resolve_transcriber()` 按 engine 名分支
- **引擎选择优先级**: `upload_request.engine` > `config.transcription.default_engine`（env `FUNASR_DEFAULT_ENGINE`）> `funasr`
- **缓存隔离**: 缓存 key 按 `(file_hash, engine)` 区分，跨引擎不命中；engine 维折 word_align / diarize 状态（`qwen3+wa:<lang>+nospk` 形态，见「diarize 开关」节），折维参数统一走 `database.cache_params_for(task)`（word_align 维读 `options.word_align` 非 config，且 `output_format=srt` 强制降 +wa；写入用 `cache_save_engine_for` 对齐失败时降 +wa 防毒化；跨引擎回退排除折维行，见 pipeline 5.5 词级时间戳节）

现状全貌、Runtime、池 dispatch、后处理 pipeline、diarize 开关：`docs/开发/ASR引擎架构.md`。定案 `docs/开发/2026-06-10-diarize开关API-设计定案.md`；ORT-CUDA `docs/开发/gpu加速/2026-05-22-ORT-CUDA-diarize-backend.md`；池 `docs/开发/gpu加速/2026-05-23-CUDA并发突破.md`；word_align `docs/开发/gpu加速/2026-06-16-Qwen3-word-align显存PoC与落地计划.md`、`docs/开发/gpu加速/2026-06-16-word-align显存安全-评审定案与落地计划.md`。

### ⚠️ Qwen3 worker audio 转换的边界

`src/core/qwen3_worker_process.py` 对 sherpa-supported 格式（`{.wav, .flac, .ogg, .mp3, .opus}`）**跳过** ffmpeg `convert_to_wav`，仅 m4a/aac/mp4/mov/webm 走转码（sherpa libsndfile/librosa fallback 读不了）。**不要为了"统一"把所有格式都 ffmpeg 转 wav** — 调研（`docs/开发/archive/spk-over-detect-归因调研结果.md`）证明 ffmpeg→wav 即使保持 16kHz mono 也会改变 audio 字节，通过 pyannote+TitaNet embedding 放大，触发 sherpa diarize FastClustering over-detect（60min-2spk 真实场景 → 4 spk）。

## 关键架构决策

**为什么不用 ABC / factory / contract test 抽象**（决策档案: `docs/开发/archive/重构计划-ASR引擎抽象.md`）:

PR1 设计阶段曾计划 ABC 抽象 + factory + contract test 体系（PR2 触发），实际工程化（PR2-4）后判定**过度设计**。当前"全局唯一引擎实例 + 薄 dispatch 路由 + per-engine config 隔离"模式已支撑 2 个引擎 + 后处理 pipeline + runtime-aware 池 dispatch（in-proc / multi-process 两套）等复杂需求，工程复杂度更低，第三个引擎接入再触发抽象不迟。

## 加新引擎的步骤
1. 在 `src/core/` 加 `<engine>_transcriber.py`，提供 `get_<engine>_transcriber()` 单例工厂（或 pool wrapper，参考 `qwen3_pool_transcriber.py`）
2. 在 `src/core/transcriber_dispatch.py` 的 `resolve_transcriber()` 加 `if name == "<engine>": ...` 分支
3. 加 unit test 到 `tests/unit/test_transcriber_dispatch.py`
4. 跑 parity 确认 FunASR + Qwen3 既有路径无回归

## 加新 diarize backend 的步骤
1. 在 `src/core/qwen3/diarize_*.py` 加新文件，暴露 `run_diarization_<backend>(audio_path, ..., num_speakers, cluster_threshold, ...) -> list[dict]`，输出 schema 跟 sherpa `run_diarization` 一致（`[{"start", "end", "speaker"}, ...]`）
2. 在 `src/core/qwen3/diarize.py:run_diarization_dispatched` 加 `if backend == "<name>": from ... import run_diarization_<backend>; return run_diarization_<backend>(...)` 分支
3. 在 `src/core/runtime.py` 的 `RuntimeEnvironment.recommend_diarize_backend()` 实现里返回新 backend 名（如果新 backend 是某 runtime 默认）
4. 单测 mock ORT session 验证 backend 行为（参考 `tests/unit/test_diarize_ort_backend.py`），integration 加 parity 测试（`tests/integration/test_diarize_ort_parity.py`）

# Config 体系（2026-05-22 治理后）

`src/core/config.py` 是单一 source of truth, Pydantic schema + 4 层优先级 + runtime-aware sentinel.

## 优先级链

```
defaults < config.json < FUNASR_PROFILE < FUNASR_* env
```

每层只填上层留空的字段; 显式覆盖永远胜出. 启动日志会列出"FUNASR_PROFILE=X applied. 覆盖字段(N): ..."防止"我明明 config.json 写了 X 怎么变 Y"的惊讶感.

PROFILE / auto sentinel / vendor / fail-fast / 切引擎手册：`docs/开发/Config体系.md`。部署切引擎也见 `docs/部署.md`。

## 加 config 字段的步骤

1. `Qwen3Config` (或对应 BaseModel) 加字段 + 默认值 + docstring
2. `_apply_env_overrides` 加一行 `_override_if_set(...)` 注入 env
3. 如有"运行时感知"需求 → 默认值用 `"auto"` sentinel + 在 `_resolve_auto_sentinels` model_validator 加分支
4. 如有平台/环境 profile 差异 → 加到 `PROFILES[<profile_name>]`
5. unit test 覆盖默认值 / env override / profile / "auto" 解析 (mock detect_runtime)

# 任务队列与高负载

- **⚠️ codex 窟窿**: `create_task` 先写 `self.tasks`,`submit_task` 才查容量;队列满**必须 `self.tasks.pop(task_id)` 回滚**,否则被拒 PENDING 永不终态 → 永久泄漏。学习: [[见 task_create_before_queue_check_leak]]。
详见 `docs/开发/任务队列与可观测性.md`、`docs/开发/2026-06-16-高负载队列机制-止血修复计划.md`、`docs/开发/2026-06-16-异步轮询契约-设计定案与落地计划.md`、`docs/开发/2026-06-17-P4小硬化-A1缓存sizecap-A2孤儿文件-落地计划.md`。

# 可观测性

⚠️ **铁律：见 `Upgrade: websocket` 头一律 `return None` 放行**（不管路径）——否则客户端连 `ws://host:port/`（根路径）会被 `/` 的 HTML 状态页拦成 HTTP 200，握手失败"无法连接到服务器"（生产事故 2026-06-17）。HTTP 端点只服务**非升级**请求；ws 握手永远穿透。`test_http_endpoints_live.py` 连根路径 `/` 钉死。
详见 `docs/开发/任务队列与可观测性.md`、`docs/开发/2026-06-16-可观测性仪表盘与测试加固-设计定案与落地计划.md`、`docs/开发/2026-06-17-根路径ws握手被HTML状态页劫持-事故retro.md`。

# DB 缓存

**⚠️ 时间戳比较铁律**:DB 用 SQLite `CURRENT_TIMESTAMP`(空格分隔),cutoff 必须 `strftime("%Y-%m-%d %H:%M:%S")` 对齐,**禁用 `isoformat()`**(`T` 分隔字符串比较 `'T'>' '` 误删 cutoff 同日晚于 cutoff 时刻的行,旧 regression 已修)。
详见 `docs/开发/任务队列与可观测性.md`、`docs/开发/2026-06-17-P4小硬化-A1缓存sizecap-A2孤儿文件-落地计划.md`。

# 部署约定

本项目仅在 macOS Apple Silicon 上运行（依赖 MPS GPU 加速）。**不要调用全局 `docker-deploy` skill**。
prod/dev 隔离与启停见 `docs/部署.md`。
