# 新 session 上下文 — Config 体系治理 (Mac + CUDA 双平台)

> **本文档是开新 session 用的 priming prompt**. 新 session 启动时把这份内容一次性给 Claude, 让它快速进入上下文.

## 一句话目标

把 config 体系治理到"Mac 和 CUDA 双平台切环境一句 env 搞定, runtime 抽象真落进 config, vendor 字段进 Pydantic, 启动时 engine-runtime 不兼容立即 fail-fast". 实现 **方案 A** (3 个改动, ~170 行新代码), 严格 TDD, 4 commit. 完成时间: 一晚 (1-2 小时).

## 上下文 — 为什么要治理

上几个 sprint (CLAUDE.md + `docs/开发/gpu加速/2026-05-22-ORT-CUDA-diarize-backend.md` + `docs/开发/gpu加速/2026-05-23-CUDA并发突破.md`) 把 Qwen3 在 CUDA 上跑通 + 单进程 in-proc pool 拿下, 现在两个引擎 (FunASR + Qwen3) × 两个平台 (Mac + Linux CUDA) 都接入了, 但 config 体系还没跟上:

- 切平台要手工拼 5+ 个 `FUNASR_QWEN3_*` env, 漏一个就 silent fallback
- `Qwen3Config.num_threads / provider` 是 hardcode 默认, **runtime.recommend_** *根本没接进 config*
- `FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS` 这个 env 在 `vendor/qwen_asr_gguf/inference/encoder.py` 里直接 `os.getenv()` 读, 绕过 Pydantic — config validation 看不到, print_config 不显示, unit test 没覆盖
- engine + runtime 不兼容时启动 OK, 第一个 task 才挂 — lazy fail-fast

## 八个具体痛点 (按重要性排序)

| # | 痛点 | 现网证据 | 本次治理 |
|---|---|---|---|
| 1 | vendor 字段绕过 Pydantic | `FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS` 在 `src/core/vendor/qwen_asr_gguf/inference/encoder.py` 直接读 | ✅ A2 |
| 2 | runtime-specific 字段表达成 hardcode | `Qwen3Config.num_threads = 4` (CUDA 上 vCPU 不同应不同) / `provider = "cpu"` | ✅ A2 |
| 3 | 跨段字段位置混乱 | `qwen3_pool_size` 在 `TranscriptionConfig` 段不在 `Qwen3Config` | ⏸ 留 B |
| 4 | 环境/平台 profile 不成体系 | `.env` 只有 dev 模板, 没 `cuda_dev/mac_prod/cuda_prod` | ✅ A1 |
| 5 | engine ↔ runtime 兼容性 lazy check | `default_engine=qwen3` + 没装 onnxruntime-gpu, 启动 OK, 第一 task 才挂 | ✅ A3 |
| 6 | env override mapping 手写 40+ 行 | `_apply_env_overrides()` 每字段手写 `_override_if_set` | ⏸ 留 C (pydantic-settings) |
| 7 | per-engine config 大块塞 config.py | `Qwen3Config` 70+ 行 (后处理 pipeline 4+5+4 字段全堆里) | ⏸ 留 B (接第 3 引擎再做) |
| 8 | 缺 auto-downgrade 自适应 | CUDA 挂了只能手动 `FUNASR_RUNTIME=cpu` escape | ❌ 不做 (prod alert 比 silent fallback 好) |

## 必读先验材料 (按顺序)

1. **`docs/开发/gpu加速/2026-05-23-CUDA并发突破.md`** — 最近 sprint 落档, 含 runtime-aware pool dispatch (cuda → InProcPool, 其他 → multi-process). **必读, 是本治理的上下文**.
2. **`CLAUDE.md`** — ASR 引擎架构 + Runtime + Diarize backend 抽象 + Qwen3 池 dispatch 章节. 必读.
3. **`src/core/config.py`** — 当前 config 全貌 (531 行). 重点看:
   - `class Qwen3Config` (L51-121) — 70+ 字段, 后处理 pipeline 全堆这里
   - `_apply_env_overrides()` (L262-369) — 40+ 个手写 `_override_if_set`
   - `_validate_config()` (L388-444) — 现有 validation, 没 engine-runtime check
4. **`src/core/runtime.py`** — `RuntimeEnvironment` Protocol + `MacRuntime/CudaRuntime/CpuRuntime` 实现. `recommend_num_threads()` 跟 `recommend_diarize_backend()` 已经写好, **但没注入 Qwen3Config**.
5. **`src/core/vendor/qwen_asr_gguf/inference/encoder.py`** — 内部直接 `os.getenv("FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS")` 读. 需要改成从 config 拿.
6. **`.env`** — 当前 dev 模板, 看 Qwen3 ANE Phase 3 那段 (L49-69) 实际 env 用法, profile 应基于这套.
7. **`config.json`** — 跟 `.env` 配合, 看注释字段 `_comment_*` 体系.

## 推荐方案: A (最小改动, 3-commit, TDD 严格)

### 改动 A1: `FUNASR_PROFILE` 套餐 env

**新增**: `config.py` 加 `_apply_profile_defaults(config_data)`, 在 `_apply_env_overrides` **之前** apply, 让用户的显式 env 仍能再覆盖.

支持的 profile (4 个):
- `mac_prod` — port 8767, engine=qwen3, pool=3, encoder=coreml_ane_full, backend_units=CPU_AND_GPU
- `mac_dev`  — port 8867, engine=qwen3, pool=2, encoder=coreml_ane_full, backend_units=CPU_AND_GPU, log=DEBUG
- `cuda_prod` — engine=qwen3, pool=2, encoder=cuda
- `cuda_dev`  — engine=qwen3, pool=2, encoder=cuda, log=DEBUG, port=8867

实现要点:
- profile 字段 deep-merge 进 config_data, 用 `setdefault` 避免覆盖已有
- 未知 profile name → warn + ignore (不挂)
- 单元测试: 每个 profile + 1 个 env 覆盖 profile 字段的 case + 1 个未知 profile case

### 改动 A2: Qwen3Config 关键字段改 "auto" + runtime 注入 + vendor 字段入 Pydantic

**改字段**:
```python
class Qwen3Config(BaseModel):
    num_threads: int | str = "auto"        # auto → runtime.recommend_num_threads()
    provider: str = "auto"                 # auto → runtime 选 (mac/cpu 都 cpu, cuda 也 cpu 因 sherpa diarize 不用了)
    backend_mlpackage_units: str = ""      # 新增, 从 vendor 移上来; "" 表示不设
```

**改 transcriber** (`qwen3_transcriber.py`):
- `get_qwen3_transcriber()` 内, 把 `num_threads="auto"` 解析成 `detect_runtime().recommend_num_threads()`
- `provider="auto"` → 同上规则
- 单例工厂 + InProc pool factory (`qwen3_inproc_pool.py:_default_transcriber_factory`) 都要改

**改 vendor**:
- `src/core/vendor/qwen_asr_gguf/inference/encoder.py` 删 `os.environ.get("FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS")`
- 改成函数参数 / 从 build_engine_config 传进来
- `src/core/qwen3/asr.py:build_engine_config` 加 `backend_mlpackage_units` 参数, 从 config 读

**回归测试**: vendor 改动是最大风险点, 必须跑 Mac integration parity (跟 production .env 一致, ANE phase 3) 验证.

### 改动 A3: startup engine ↔ runtime 兼容性 check

**改 `_validate_config()`**:
```python
def _validate_engine_runtime(config, errors):
    from src.core.runtime import detect_runtime
    runtime = detect_runtime()
    engine = config.transcription.default_engine
    if engine == "qwen3":
        backend = runtime.recommend_diarize_backend()
        if backend == "ort_cuda":
            try:
                runtime.validate()  # onnxruntime CUDA EP check
            except RuntimeError as e:
                errors.append(
                    f"default_engine=qwen3 + runtime={runtime.name} 需要 onnxruntime-gpu, "
                    f"但 {e}. 用 FUNASR_RUNTIME=cpu 降级或修依赖."
                )
```

单元测试: monkeypatch detect_runtime + validate 抛错, assert config load 时 errors 含明确字串.

## 自主推进准则

跟上次 (`Qwen3-CUDA并发突破-新session-prompt.md`) 一致.

### 什么时候 **不要** 中途询问
- ✅ 改 config / vendor / 写 test, 直接做
- ✅ 跑 pytest / git commit, 直接做
- ✅ commit message 仿现有 `git log --oneline -10` 风格
- ✅ profile 名命名风格, 自己定 (mac_prod / mac_dev / cuda_prod / cuda_dev 已建议)

### 什么时候 **必须** 停下来
- ❌ vendor 改动破坏 Mac integration parity 且改不回去
- ❌ 改完 pre-existing 7 个 ANE_FULL fail 测试变得 > 7 (说明引入回归)
- ❌ profile 设计上有歧义需要用户拍板 (比如多 profile 同名怎么处理)

## TDD 流程 (严格执行, "先红再绿再 commit")

跟上次完全一致.

### 推荐 commit 拆分 (4 个)

1. `test+feat(config/profile): FUNASR_PROFILE 套餐预填 4 profile` — A1, 含 4 profile 数据 + deep-merge + unit test (5-6 case)
2. `test+feat(config/runtime-inject): Qwen3Config num_threads/provider 字段 "auto" sentinel + runtime 注入` — A2 上半部分, vendor 不动, 仅 config + transcriber + inproc_pool factory 改
3. `test+feat(qwen3/vendor): backend_mlpackage_units 从 vendor env 提升到 Qwen3Config` — A2 下半部分, vendor encoder + build_engine_config 改, Mac integration parity 跑
4. `test+feat(config/validate): startup engine-runtime 兼容性 fail-fast` — A3
5. `docs(claude.md): 落档 FUNASR_PROFILE + auto sentinel + startup check`

## Acceptance Criteria

| # | 指标 | 怎么验 |
|---|---|---|
| 1 | `FUNASR_PROFILE=cuda_dev` 一行替代之前 5+ env | dev box ssh 测一次, log 显示 "FUNASR_PROFILE=cuda_dev applied" + 单 task 跑通 |
| 2 | `Qwen3Config.num_threads="auto"` 在 mac 上 = 4, 在 cuda dev box (8vCPU) 上 = 4, 在 4vCPU 容器上 = 2 | unit test 用 monkeypatch + 远端 integration |
| 3 | `FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS` env 删除, Qwen3Config 拿 | grep vendor/ 确认 `os.getenv` 没了; Mac integration parity 跑过 |
| 4 | `default_engine=qwen3` + 假装 CUDA 不可用 → 启动 fail-fast | unit test monkeypatch + 实测 Mac 上跑 |
| 5 | Mac unit 测试 365 pass, 7 pre-existing fail 不变 (零回归) | `venv/bin/python -m pytest tests/unit/` |
| 6 | Mac integration parity 通过 (含 ANE phase 3 路径) | `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/test_qwen3_backend_coreml_ane_parity.py` |
| 7 | 远端 cuda integration 通过 (走 dispatch + InProcPool) | `ssh <user>@<dev-host> ... bash scripts/_remote_pool_dispatch_integration.py` |
| 8 | 落档 + CLAUDE.md 更新 | docs/开发/ + CLAUDE.md diff |

## 不在范围内 (避免 scope creep)

- ❌ 方案 B (per-engine config 模块化 + pool config 拆段) — 接第 3 引擎再做
- ❌ 方案 C (pydantic-settings + YAML profile + auto-downgrade) — 5+ 部署目标才划算
- ❌ Mac unit 7 个 pre-existing ANE_FULL fail 修复 — 跟本 sprint 无关 (是其他 sprint 遗留)
- ❌ 改 `qwen3_pool_size` 位置 (从 TranscriptionConfig 挪到 Qwen3Config) — 痛点 3, 留 B
- ❌ 重命名 `max_concurrent_tasks` → `funasr_pool_size` — 改动面太大, 留 B

## 起手第一步

```bash
# 1. 看一遍现有 config + runtime
sed -n '51,121p' src/core/config.py    # Qwen3Config 全貌
sed -n '262,369p' src/core/config.py   # _apply_env_overrides
cat src/core/runtime.py                # 看 recommend_* 已有逻辑

# 2. 看 vendor 直接读 env 的痕迹
grep -rn "FUNASR_QWEN3" src/core/vendor/

# 3. 起 commit 1 — profile 套餐, TDD 红
#    新建 tests/unit/test_config_profile.py 描述 4 个 profile + deep-merge + 未知 profile + env 覆盖 profile
#    跑 pytest 看红 (ModuleNotFoundError 或 AttributeError)
#    实现 config.py 的 _apply_profile_defaults + PROFILES dict
#    跑 pytest 看绿
#    commit

# 4. 同样节奏推 commit 2/3/4/5
```

## 落档 / commit 约定

跟上次一致.

- **独立维护, 不开 PR** (`feedback_no_pr_workflow` memory)
- **不主动 push**, git commit 后等用户指示
- Mac 是 production, vendor 改动必须跑 Mac integration parity (`tests/integration/test_qwen3_backend_coreml_ane_parity.py`)
- 部署不走 docker (`project_funasr_no_docker` memory)
- 远端 dev box CUDA: `ssh <user>@<dev-host>`, 详见 `scripts/_remote_*.sh` 头部 LD_LIBRARY_PATH 配置

## 远端 cuda 验证 (最后一步)

完成所有改动后, 远端走一次 integration 看 `FUNASR_PROFILE=cuda_dev` 一句话替代之前 5 个 env:

```bash
# 同步 src + scripts
rsync -av --exclude='__pycache__' src/ scripts/ \
  <user>@<dev-host>:~/Dev/projects/funasr_spk_server/

# 远端跑
ssh <user>@<dev-host> 'cd ~/Dev/projects/funasr_spk_server && \
  unset FUNASR_QWEN3_ASR_ENCODER_PROVIDER FUNASR_QWEN3_POOL_SIZE FUNASR_DEFAULT_ENGINE && \
  export FUNASR_PROFILE=cuda_dev && \
  NV=venv/lib/python3.12/site-packages/nvidia; TRT=venv/lib/python3.12/site-packages/tensorrt_libs; \
  export LD_LIBRARY_PATH="${TRT}:${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/nvjitlink/lib:${NV}/cusparse/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cuda_cupti/lib" && \
  venv/bin/python scripts/_remote_pool_dispatch_integration.py'
```

预期: 输出含 `FUNASR_PROFILE=cuda_dev applied` + `pool class = Qwen3InProcPool` + `TOTAL_WALL≈142s` + 2 个 TASK_OK.

---

**Good luck. 一晚 (1-2 小时) 拿下方案 A 全部 4 commit + 落档**.

---

## 评审后更新 (2026-05-22 plan-eng-review)

经过工程评审, 5 个决议修正原 plan, 实际开工版以下面为准.

### 决议清单

| # | 决议 | 影响 |
|---|---|---|
| D1 | "auto" sentinel 用 **Pydantic model_validator** 一次性解析, 不在 factory 入口解 | Qwen3Config.num_threads 字段类型保持 `int`, 下游 5 处 (transcriber:58/237/399/575, inproc_pool:117) 零改动 |
| D2 | profile **覆盖 config.json**, 优先级 `defaults < config.json < profile < env` | A1 用 dict-merge 不是 setdefault, 启动日志列出"profile 覆盖了哪些字段" |
| D3 | A3 仅检查 default_engine 充分 (per-request engine ≠ default 已被 dispatch 在 transcriber_dispatch.py:57 拒) | plan 原 A3 不动 |
| D4 | ENCODER_TIMING (encoder.py:385) 一并提升 | Qwen3Config 加 `encoder_timing_enabled: bool = False` + .env override |
| D5 | qwen3_transcriber.py:58 `_build_default_diarize_engine` hardcode `num_threads=4` 顺手修接 cfg | A2 cluster_merge embedding 路径多 2 行 + 1 test, 跟 D1 一致 |

### Pre-existing bug 顺手修 (在 A2 commit 内)

- `qwen3_transcriber.py:237-238` 默认 `num_threads=8, provider="cpu"` 跟 `Qwen3Config.num_threads=4` 不一致, 改成 `num_threads=4` (跟 config 一致) 或直接走 model_validator 解析后的值

### 测试覆盖矩阵 (22 case + Mac parity + 远端 cuda integration)

```
A1 profile (commit 1, 6 case)
  ├── profile 覆盖 config.json 字段 (新优先级)
  ├── env 仍能覆盖 profile
  ├── 4 个 profile defaults 正确 × 4
  ├── 未知 profile name → warn + 不挂
  └── 启动日志列出 "profile=X applied, 覆盖了 [字段列表]"

A2 上 — Pydantic model_validator (commit 2, 7 case)
  ├── num_threads="auto" + mock MacRuntime → 4
  ├── num_threads="auto" + mock CudaRuntime(4vCPU) → 2
  ├── num_threads="auto" + mock CudaRuntime(8vCPU) → 4
  ├── num_threads=2 (显式 int) → 2 (不被覆盖)
  ├── provider="auto" + mock Mac/Cuda → "cpu"
  ├── [回归] qwen3_transcriber.py:58 _build_default_diarize_engine 用 cfg.num_threads
  └── [回归] qwen3_transcriber.py:237 默认值跟 config 一致

A2 下 — vendor (commit 3, 5 case + Mac parity)
  ├── build_engine_config 把 backend_mlpackage_units 传到 vendor encoder
  ├── build_engine_config 把 encoder_timing_enabled 传到 vendor encoder
  ├── backend_mlpackage_units Literal 约束 (CPU_AND_NE / CPU_AND_GPU / ALL)
  ├── encoder_timing_enabled 字段 default + env override
  ├── vendor encoder.py 不再 os.getenv (grep 验证)
  └── [Mac parity] FUNASR_RUN_INTEGRATION=1 跑 test_qwen3_backend_coreml_ane_parity.py

A3 engine-runtime check (commit 4, 2 case)
  ├── default_engine=qwen3 + mock CudaRuntime.validate() raise → config errors 含 "onnxruntime-gpu"
  └── default_engine=funasr + 同条件 → 不挂

远端 cuda integration (验收, 不入 commit)
  └── ssh ... FUNASR_PROFILE=cuda_dev → log 含 "profile applied" + InProcPool + 2 TASK_OK
```

### Commit 拆分 (更新)

1. `test+feat(config/profile): FUNASR_PROFILE 套餐预填 4 profile, profile 覆盖 config.json + env 覆盖 profile` — A1, 6 case
2. `test+feat(config/runtime-inject): Qwen3Config num_threads/provider "auto" sentinel via Pydantic validator, 顺手修 :58 hardcode 4 + :237 num_threads=8 默认` — A2 上, 7 case
3. `test+feat(qwen3/vendor): backend_mlpackage_units + encoder_timing_enabled 从 vendor env 提升到 Qwen3Config + Literal 约束` — A2 下, 5 case + Mac parity
4. `test+feat(config/validate): startup engine-runtime 兼容性 fail-fast` — A3, 2 case
5. `docs(claude.md): 落档 FUNASR_PROFILE + Pydantic auto sentinel + startup check`
