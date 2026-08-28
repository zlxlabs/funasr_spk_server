# Config 体系与切引擎 / 部署操作（CLAUDE.md 规则摘录）

> 下文自仓根 `CLAUDE.md`（基线 `f0c9e13`）按原标题、原正文搬出，本卡不改写。部署操作的现行手册以 `docs/部署.md` 为准；本文件保留规则文件原文字以便抽样对账。
>
> 核销表：`docs/开发/2026-08-28-CLAUDE规则瘦身核销表.md`

# Config 体系（2026-05-22 治理后）

`src/core/config.py` 是单一 source of truth, Pydantic schema + 4 层优先级 + runtime-aware sentinel.

## 优先级链

```
defaults < config.json < FUNASR_PROFILE < FUNASR_* env
```

每层只填上层留空的字段; 显式覆盖永远胜出. 启动日志会列出"FUNASR_PROFILE=X applied. 覆盖字段(N): ..."防止"我明明 config.json 写了 X 怎么变 Y"的惊讶感.

## FUNASR_PROFILE 套餐 env

一行 env 切平台 / 切环境, 不再手工拼 5+ `FUNASR_*` env:

| profile | port | engine | qwen3_pool | encoder | log |
|---|---|---|---|---|---|
| `mac_prod` | 8767 | **funasr** | 1 | coreml_ane_full | INFO |
| `mac_dev` | 8867 | **funasr** | 1 | coreml_ane_full | DEBUG |
| `cuda_prod` | (默认 8767) | qwen3 | 1 | cuda | INFO |
| `cuda_dev` | 8867 | qwen3 | 1 | cuda | DEBUG |

**引擎按硬件分**（2026-06-16 拍板）：**Mac → funasr**（速度快，大内存如 64G 并发拉得开，用 `FUNASR_MAX_CONCURRENT_TASKS` 调，实测可 3 进程）；**CUDA → qwen3**（准确度更高，GPU 算力补速度；3060 12G 显存只够 1 进程）。Mac 想用 qwen3（追准确度）走 `FUNASR_DEFAULT_ENGINE=qwen3` 临时切——mac profile 已保留 qwen3 的 encoder/pool 配置即用。

pool 全 profile 默认 1（2026-06-10 拍板）：3060 12GB 实测 pool=2 + word_align 双 MMS CUDA session 撞显存 OOM（fallback 不挂但词级时间戳静默丢失）。并发需求用 `FUNASR_QWEN3_POOL_SIZE` env 按机器显存/内存显式开。

用法: `FUNASR_PROFILE=cuda_dev venv/bin/python run_server.py`. 未知 profile name → warn + ignore, 不挂. 加新 profile 改 `src/core/config.py` 的 `PROFILES` dict.

## "auto" sentinel 字段 (Pydantic model_validator 解析)

`Qwen3Config.num_threads` / `provider` 默认 `"auto"`, 在 model load 时一次性解析:

```python
num_threads="auto" → detect_runtime().recommend_num_threads()
  · MacRuntime  → 4 (PoC: t=4 比 t=8 wall -11.5%)
  · CudaRuntime → 2/4 按 vCPU (≤4=2, ≥5=4)

provider="auto" → "cpu"  # sherpa embedding extractor 跨 runtime 都 cpu
```

字段类型 `int | str`, 解析后保证 int, 下游 5 个消费点 (transcriber:58/237/399/575, inproc_pool:117) 永远拿到具体 int. 显式 int / 显式 provider 字符串不被覆盖.

## vendor 字段进 Pydantic

vendor 不再 `os.environ.get(...)`, 全走 Qwen3Config:

- `backend_mlpackage_units: Literal["CPU_AND_NE", "CPU_AND_GPU", "ALL"] = "CPU_AND_NE"` — Phase 3 backend mlpackage compute_units
- `encoder_timing_enabled: bool = False` — encoder 打印 mel/fe/be 耗时 (排查性能用)

env override 仍可用 (`FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS` / `FUNASR_QWEN3_ENCODER_TIMING`), 但走 `_override_if_set` 标准路径, Pydantic 看到, `print_config` 显示.

## startup engine-runtime fail-fast

`Config._validate_engine_runtime`: `default_engine=qwen3` + cuda runtime + ORT CUDA EP 缺 → `sys.exit(1)`, 报 "用 FUNASR_RUNTIME=cpu 降级或修依赖". 不再 lazy 等第一个 task 才挂.

per-request `engine ≠ default_engine` 已被 `transcriber_dispatch.py:57` 拒, 所以 startup 仅查 default_engine 充分.

## 切换设备 / 切引擎操作手册

不同部署目标 × 不同引擎的常见组合, 推荐姿势:

### A. Mac + FunASR (主路径, 速度快 / 并发好)

mac_prod/mac_dev profile 默认就是 funasr, 直接起:

```bash
# prod (PM2 守护)
FUNASR_PROFILE=mac_prod pm2 start ecosystem.config.cjs

# dev (前台直跑, 看日志)
FUNASR_PROFILE=mac_dev venv/bin/python run_server.py
```

大内存(如 64G)拉并发: 加 `FUNASR_MAX_CONCURRENT_TASKS=3`(实测可 3 进程).

### B. Linux CUDA + Qwen3 (远端 dev box / 未来 prod)

cuda profile 默认 qwen3(准确度更高, GPU 算力补速度):

```bash
export FUNASR_PROFILE=cuda_dev   # 或 cuda_prod
# LD_LIBRARY_PATH 配 CUDA libs (远端启动脚本里设, 详见 scripts/_remote_*.sh)
venv/bin/python run_server.py
```

注: 3060 12G 显存只够 `qwen3_pool_size=1`.

### C. Mac + Qwen3 (追准确度, 按需切)

Mac 默认 funasr, 想用 qwen3 高准确度走 env 覆盖(mac profile 已留 qwen3 encoder/pool 配置即用, 需先 `scripts/download_qwen3_models.sh` 拉模型):

```bash
# 干净法 (走 config.json 默认 port/log)
FUNASR_DEFAULT_ENGINE=qwen3 venv/bin/python run_server.py

# 套餐 + 覆盖法 (沿用 mac profile 的 port/log/encoder, 只改引擎)
FUNASR_PROFILE=mac_dev FUNASR_DEFAULT_ENGINE=qwen3 venv/bin/python run_server.py
```

注: FunASR 不支持 CUDA, Linux 上自动走 CPU (MPS 仅 Mac).

### "改哪里" 决策树

```
要长期保留这个配置吗?
├── 是 → 改文件
│       ├── 单机日常用法         → 本机 .env 写 FUNASR_PROFILE=xxx
│       ├── 多机同环境共用       → config.json (但 profile 通常已够用)
│       └── 加一个新部署目标     → src/core/config.py 的 PROFILES dict
└── 否 (一次性 / 调试) → 命令行 export FUNASR_xxx=yyy
```

### 最常见 3 种用法

1. **dev 机切环境** (最常用) — 本机 .env 一行:
   ```
   FUNASR_PROFILE=mac_dev   # 或 cuda_dev
   ```
2. **临时换引擎调试** (Mac 默认 funasr, 临时切 qwen3 追准确度) — 命令行 cover:
   ```bash
   FUNASR_PROFILE=mac_dev FUNASR_DEFAULT_ENGINE=qwen3 venv/bin/python run_server.py
   ```
3. **加新部署目标** (新机器 / 新硬件配置) — 改 `src/core/config.py:PROFILES`:
   ```python
   PROFILES = {
       ...
       "cuda_l40_prod": {  # 例: L40 GPU, pool 4
           "transcription": {"default_engine": "qwen3", "qwen3_pool_size": 4},
           "qwen3": {"asr_encoder_provider": "cuda"},
       },
   }
   ```

### 启动后怎么验证配置生效

启动日志会打印 (`_apply_profile_defaults` 输出):
```
FUNASR_PROFILE=cuda_dev applied. 覆盖字段 (5):
  server.port('(默认)'→8867),
  transcription.default_engine('(默认)'→'qwen3'),
  ...
```

如果发现某字段没被 profile 覆盖, 100% 是 `.env` 或 shell env 里设了更高优先级的 `FUNASR_*`. grep `.env` 找污染源 (清完再起服务).

# 部署约定（macOS only）

本项目仅在 macOS Apple Silicon 上运行（依赖 MPS GPU 加速）。**不要调用全局 `docker-deploy` skill**。

prod/dev 物理隔离：

| 环境 | 目录 | 端口 | 守护 |
|---|---|---|---|
| prod | `~/Production/funasr_spk_server/` | 8767 | **PM2** (`funasr-server`) |
| dev | `~/Dev/projects/250729_funasr_spk_server/funasr_spk_server/` | 8867 | **不挂 PM2**（前台直跑） |

## prod 部署
```bash
cd ~/Production/funasr_spk_server
git pull origin main
venv/bin/pip install -r requirements.txt   # 仅 requirements 有变化时
pm2 restart funasr-server                  # 启动数据库自动迁移
```

## dev 运行
默认前台直跑，方便看日志和调试：
```bash
venv/bin/python run_server.py
```
仅在需要长跑调试时才用 PM2：`pm2 start ecosystem.config.cjs`，用完 `pm2 delete funasr-server-dev && pm2 save`。

详细部署文档：`docs/部署.md`
