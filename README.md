# FunASR SPK Server — 多引擎语音转录 + 说话人识别服务

多引擎（FunASR / Qwen3）音视频转录服务器，支持说话人识别（diarization）、词级时间戳，提供 JSON 和 SRT 两种输出格式。**引擎与各项转录选项均通过 WebSocket `upload_request` 字段按部署 / 请求指定**。

> 项目代号 `funasr_spk_server` 源于初代单引擎（FunASR）实现，现已演进为多引擎架构——**代号保留，不代表仅支持 FunASR**。

🍎 **当前生产目标是 macOS Apple Silicon**（`mac_prod` 默认 FunASR）；Linux CUDA 也是支持路径，使用 `cuda_prod` / `cuda_dev` 与 Qwen3。

## 功能特性

- ✅ 多种音视频格式（wav / mp3 / mp4 / m4a / flac / webm 等）
- ✅ 自动说话人识别与分离 + 精确到毫秒的时间戳
- ✅ 双输出格式：JSON（合并说话人）& SRT（原始分割）
- ✅ **双 ASR 引擎，按硬件选**（FunASR / Qwen3，见[引擎选型](#引擎选型按硬件)）
- ✅ 词级时间戳（qwen3，per-request 开关）
- ✅ 结构化 `terms` 术语提示：FunASR 作为 hotword，Qwen3 作为服务端控制的 context；空列表零影响
- ✅ `/capabilities` 能力探针与 `scripts/doctor.py --json` 只读部署诊断
- ✅ WebSocket 实时通信 + 任务队列 + 高负载准入控制
- ✅ 异步轮询契约（批量 `task_status_batch`，根治高负载 300s 超时）
- ✅ 按 `(file_hash, engine)` 的智能缓存，相同文件秒回
- ✅ 可观测性：同端口 `/health`（存活探针）+ `/metrics`（Prometheus 指标）
- ✅ 企微机器人通知 + JWT 认证

## 文档导航

README 只讲「是什么 + 怎么快速跑起来」，详细内容在 `docs/`（一处权威，避免重复）：

| 想做什么 | 看这里 |
|---|---|
| 📂 文档总索引 | [docs/README.md](docs/README.md) |
| 🔌 接入客户端（WebSocket 协议 / 上传 / 输出格式 / 异步轮询 / 故障排除） | [docs/使用/客户端交互指南.md](docs/使用/客户端交互指南.md) |
| 🚀 部署（Mac 生产 PM2 / Linux CUDA / doctor 验收） | [docs/部署.md](docs/部署.md) |
| 🧩 服务端协议（状态机 / 并发控制 / 消息规格） | [docs/开发/Server-Client 交互协议.md](docs/开发/Server-Client%20交互协议.md) |
| ⚙️ 架构 / 配置体系 / 引擎与池 / 加新引擎 | [CLAUDE.md](CLAUDE.md) |
| 🔧 配置项全集 | `.env.example`（env 权威）+ `config.json` |

## 架构概览

```
WebSocket Handler ──▶ Task Manager ──▶ ASR Engine (dispatch)
  (协议/上传)          (队列/并发)        FunASR / Qwen3
       │                   │                   │
   Auth & 校验         Database(缓存)      File Manager
```

- **Transcriber Dispatch**（`src/core/transcriber_dispatch.py`）：全局唯一引擎模式，启动时由 `transcription.default_engine` 锁定，`upload_request.engine` 跨引擎请求立即 reject。
- **引擎实现**：FunASR（`funasr_transcriber.py`，Paraformer-zh + cam++）/ Qwen3（`qwen3_pool_transcriber.py`，GGUF/ONNX + diarize，runtime-aware 池）。
- 架构、配置体系、池 dispatch、后处理 pipeline 等权威细节见 [CLAUDE.md](CLAUDE.md)。

## 快速开始

### 环境要求

- Mac 生产：macOS 13+（Apple Silicon）；Linux CUDA：支持 CUDA runtime 与 Qwen3
- Python 3.12 + FFmpeg（`brew install ffmpeg`）
- 8GB+ 内存（推荐 16GB）；首次运行自动下载模型（~2GB）

### 安装与运行

```bash
git clone <repository_url> && cd funasr_spk_server
python3.12 -m venv venv && venv/bin/pip install -r requirements.txt

cp .env.example .env          # 按需改 webhook / 认证密钥 / 端口
venv/bin/python run_server.py # 默认 funasr，监听 ws://0.0.0.0:8767
```

- **切 Qwen3 引擎**（追准确度）：先 `bash scripts/download_qwen3_models.sh` 拉 ~2.1GB 模型，再 `FUNASR_DEFAULT_ENGINE=qwen3 venv/bin/python run_server.py`。
- **生产部署**（PM2 守护 / prod-dev 隔离 / 更新流程）：见 [docs/部署.md](docs/部署.md)。

### 接入客户端

最小流程：`connect → upload_request → upload_data → 等 task_complete`；批量场景用 `task_status_batch` 异步轮询。协议入口、字段表、Python 基础上传示例、输出格式、错误处理见 **[客户端交互指南](docs/使用/客户端交互指南.md)**；使用 `terms` 时还必须按指南先完成 capabilities/connected 预检。

`upload_request.data.terms` 是可选的结构化术语列表。服务端会规范化并校验；空列表不改变识别行为。有效术语不会读取普通缓存，以免把带术语结果误当成普通结果；响应 metadata 会回显 `terms_count` 与引擎相关的 `context_applied`。FunASR hotword、Qwen3 context、缓存与错误契约见[客户端指南](docs/使用/客户端交互指南.md)和[服务端协议](docs/开发/Server-Client%20交互协议.md)。

上传前可用 `GET /capabilities` 确认当前引擎是否声明 `features.terms=true`；WebSocket `connected` 消息中的 `capabilities` 应与之完全一致，缺失或 `capability_id` 不一致时客户端应 fail-closed，不发送术语请求。

## 引擎选型（按硬件）

部署时由 profile / `FUNASR_DEFAULT_ENGINE` 锁定一个引擎，运行时全局唯一：

| 引擎 | 推荐硬件 | 准确度 | 速度 | 并发模型 | 默认所在 profile |
|------|----------|--------|------|----------|------------------|
| **`funasr`**<br>(Paraformer-zh + cam++) | **Mac（大内存如 64G）** | 高 | **快**（~0.1 RTF, MPS） | `max_concurrent_tasks` 多进程，内存越大并发越高（64G 实测 3 进程） | `mac_prod` / `mac_dev` + 裸 config 兜底 |
| **`qwen3`**<br>(1.7B GGUF/ONNX + diarize) | **CUDA / 强 GPU** | **更高** | 中（~0.118 RTF M1 Max；提速需更强 GPU） | `qwen3_pool_size`（默认 1；RTX 3060 12G 显存上限 1 进程） | `cuda_prod` / `cuda_dev` |

> 用 `FUNASR_PROFILE=mac_prod|mac_dev|cuda_prod|cuda_dev` 一行切平台/环境。profile 与配置优先级（`defaults < config.json < FUNASR_PROFILE < FUNASR_* env`）详见 [CLAUDE.md](CLAUDE.md) Config 体系章节。

## 可观测性

服务在 WebSocket 同端口暴露两个只读 HTTP 端点（零额外端口）：

```bash
curl http://<host>:<port>/health     # 存活探针: 200 healthy / 503 degraded (JSON)
curl http://<host>:<port>/capabilities # 当前 engine/runtime/features/capability_id
curl http://<host>:<port>/metrics    # Prometheus 文本: 队列深度/在途/错误率/缓存命中/EMA/VRAM
# 浏览器打开 http://<host>:<port>/  →  极简状态页 (自带, 每 3s 自刷, 零外部依赖)
```

- `/`（状态页）浏览器直接打开即可；绑 `0.0.0.0` 设了 token 时用 `http://host:端口/?token=xxx`。
- `/health` 裸放（只是死活）；客户端可在打转录前预检。
- `/metrics` 默认仅在显式绑 LAN/loopback 时裸放；`server.host=0.0.0.0` 时必须设 `FUNASR_METRICS_TOKEN`（否则拒绝，防全网段暴露），访问带 `?token=` 或 `Authorization`。
- `scripts/doctor.py --json` 可从任意当前工作目录运行，严格只读检查配置、provider、模型工件和目录可用性：退出码 `0`=正常、`1`=警告、`2`=错误。它不创建目录、不写配置、不启动服务；详见[部署指南](docs/部署.md)。
- 总开关 `FUNASR_METRICS_ENABLED`（默认 on）。设计与指标清单见 [docs/开发/2026-06-16-可观测性仪表盘与测试加固-设计定案与落地计划.md](docs/开发/2026-06-16-可观测性仪表盘与测试加固-设计定案与落地计划.md)。

## 测试

```bash
# 单元测试（毫秒级，默认）
venv/bin/python -m pytest tests/unit/

# 端到端真实模型测试（FunASR parity + Qwen3/FunASR ws e2e，需加载模型）
FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/
```

测试约定（三层目录 / TDD / parity）见 [CLAUDE.md](CLAUDE.md) 测试约定章节。

## License

本项目以 **MIT** 协议开源，见 [LICENSE](LICENSE)。

集成的第三方源码(`src/core/vendor/qwen_asr_gguf/`,来自 [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline),MIT)、依赖库与**模型权重**的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

> ⚠️ 模型权重许可证与本仓库代码相互独立:其中 word_align 用的 **MMS-300M 为 CC-BY-NC-4.0(禁商用)**。本项目定位个人/非商业自用;商用前请逐一复核各模型卡条款。
