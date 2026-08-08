# 文档索引

本目录文档分三类：**使用** / **开发** / **背景资料**。

## 📘 使用（给 API 调用方看）

| 文档 | 主题 |
|---|---|
| [客户端交互指南](使用/客户端交互指南.md) | WebSocket 协议、`terms` 术语、能力协商、单文件/分片上传、**异步轮询契约（批量 `task_status_batch`，根治 300s 超时）**、Python 客户端完整示例 |
| [部署指南](部署.md) | Mac `mac_prod`/FunASR 生产 (PM2)、Linux CUDA 支持路径、`doctor --json` 验收 |

## 🛠 开发（给改源码的人看）

> **架构现状权威来源**：项目根 `CLAUDE.md` ASR 引擎章节（FunASR + Qwen3 双引擎、dispatch 路由、加新引擎步骤）

| 文档 | 主题 |
|---|---|
| [Server-Client 交互协议](开发/Server-Client%20交互协议.md) | 服务端视角：能力契约、`terms`、任务状态机、并发控制、错误重试、`engine` 字段路由、异步轮询契约（`task_status_batch` 消息规格） |
| [WebSocket大文件传输最佳实践](开发/WebSocket大文件传输最佳实践.md) | 大文件分片上传方案 + 客户端实现参考 |
| [兼容性开发文档](开发/FunASR音频转文本服务器兼容性开发文档.md) | FunASR 生产路径详解（Qwen3 路径见 CLAUDE.md） |
| [可观测性仪表盘设计](开发/2026-06-16-可观测性仪表盘与测试加固-设计定案与落地计划.md) | `/health` + `/metrics`(Prometheus) + `/` 状态页：架构、指标清单、A5 安全、测试加固（CEO+Eng+codex 三重审稿） |
| [gpu加速/](开发/gpu加速/) | MPS 加速实施记录 + 长音频补丁方案（3 篇按日期） |
| [archive/](开发/archive/) | 历史 PR/PoC 决策档案：ASR 引擎抽象计划、Qwen3 Mac 加速 PoC、PR4 长音频校对稿对比等 |

## 📚 背景资料

| 文档 | 主题 |
|---|---|
| [项目起源](项目起源.md) | 2025-07 初始设计稿（含「已演进项」说明，原跨平台需求实际收敛为 mac-only + MPS） |
| [VAD并发问题解决方案](funasr相关/VAD并发问题解决方案.md) | FunASR Python 版 VAD 并发不安全的历史问题及解决路径（pool 模式由来） |

---

## 文档维护原则

- **使用文档**：API/部署方式变化时立即同步
- **开发文档**：架构调整时跟进；历史 ADR（架构决策记录）不删除，按日期归档
- **背景资料**：FunASR 上游内容不复制到本仓库（保持 modelscope/huggingface 官方为权威）

## 历史清理记录（2026-05-15）

第一轮 — docs/ 整理：
- 删除：`README_WINDOWS.md`（项目 mac-only）、`DEPLOYMENT.md`（被 `部署.md` 替代）
- 删除：`funasr相关/funasr readme.md` + `non-streaming.md`（FunASR 上游冗余内容）
- 删除：`funasr python 并发错误问题/funasr-onnx-offline-vad.cpp`（无关 C++ 片段）
- 移动：`使用/server-client-interaction.md` → `开发/Server-Client 交互协议.md`（受众是服务端开发者）
- 新建：`部署.md` + `README.md` 索引

第二轮 — 项目根目录整理：
- 移动：`项目设计.md` → `docs/项目起源.md`（顶部加「已演进项」说明）
- 修订：`setup_mac.sh` Python 版本对齐 3.11 + 指向 docs/部署.md

第三轮 — mac-only 收敛，删除 Windows / Docker 残留：
- 删除：`Dockerfile` / `docker-compose.yml` / `docker/`（整个目录）
- 删除：`manage.bat` / `start.bat` / `start_background.bat`（根级 Windows wrapper）
- 删除：`windows_scripts/`（整个目录）
- 简化：`README.md` / `CLAUDE.md` / `docs/部署.md` 移除「为什么不能 Docker」长解释
- src/ 内尚有少量 Windows 平台分支代码（main.py / file_based_process_pool.py / utils/platform_utils.py），留待后续 PR 清理（见 TODOS.md）

## 历史清理记录（2026-05-16）

第四轮 — `docs/开发/` 文档清理，PR1-4 + Qwen3 Mac 加速工程化全部落地后整理：
- 删除（已完成且无再生价值的 session prompt）：
  - `集成-Qwen3-Diarize-引擎-新session-prompt.md`（路径已废，工程实际走 spike→工程化）
  - `集成-Qwen3-多Worker池-新session-prompt.md`（多 worker pool 已全部落地）
  - `PR4-工程化-新session-prompt.md`（short-segment guard + cluster merge 全部落地）
  - `Qwen3-Mac硬件加速-工程化-新session-prompt.md`（Phase 1+2 工程化 8 commit 全落地）
- 归档到 `开发/archive/`（PoC 决策档案 + 实测数据）：
  - `重构计划-ASR引擎抽象.md`（ABC 抽象方案决策日志，全部落地后判定过度设计）
  - `PR4-长音频质量与并发性能-新session-prompt.md`（PR4 问题清单）
  - `Qwen3-Mac硬件加速-PoC-新session-prompt.md`（5 方向 PoC 路线）
  - `PR4-Qwen3长音频POC验证结果.md`（分段参数 + RTF 基准）
  - `PR4-149min校对稿对比分析.md`（长音频校对稿实测数据）
- 打补丁：
  - `Server-Client 交互协议.md` 加 §2.1 `engine` 字段路由 + 缓存隔离说明
  - `FunASR音频转文本服务器兼容性开发文档.md` 开头加适用范围声明（FunASR 生产路径 + Qwen3 旁路）
- `CLAUDE.md` ASR 引擎章节重写：反映双引擎并行接入现状 + ABC 抽象决策（"过度设计"判定）
- 修复 broken link：`docs/部署.md` / `docs/项目起源.md` / `README.md`（2 处）/ `docs/README.md` 中指向旧"重构计划-ASR引擎抽象.md"的链接,统一改成指向 `CLAUDE.md` ASR 引擎章节
