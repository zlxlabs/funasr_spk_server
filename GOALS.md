# 项目里程碑路线图

## 项目目标

- **目标**：让 VTA 生成的结构化 ASR 术语在服务端安全、可审计地进入各引擎，同时保持空值请求的旧行为。
- **完成定义**：I1 协议/FunASR、I2 Qwen context、I4 doctor 均独立合并，并在各阶段拿到单测、CI 与真实入口证据；I3 仅向 VideoTranscriptAPI 创建交接 issue，不修改下游仓库。
- **当前激活里程碑**：I2

## 里程碑路线图

### I1：terms 协议与 FunASR 生产路径

- **状态**：已完成
- **预期产出**：请求 terms 规范化、分片/单文件共享 effective request、错误映射、FunASR hotword、最小 capabilities。
- **当前范围**：本仓协议、FunASR、capabilities；有效 terms 绕过普通缓存；不做 Qwen context 和 doctor。
- **关键决策**：raw 先限额，再 NFKC/trim/空白折叠/稳定去重；验证只由 FileUploadRequest 调用，任务只复制 effective terms；超限 fail-fast 为 invalid_terms。
- **已知阻塞**：无。I1 已由 PR #4 合并至 `faf2fdb`；仓库没有 workflow/status checks，因此没有获得 CI 证据，作为既有平台缺口记录，不伪称 CI 通过。
- **推进前必须拿到的证据**：
  - [x] 协议与回归 unit 全绿；环境：本地 venv；命令：`FUNASR_NOTIFICATION_ENABLED=false venv/bin/python -m pytest tests/unit/test_common_terms_protocol.py tests/unit/test_transcribe_options.py tests/unit/test_websocket_handler_engine.py tests/unit/test_terms_cache_bypass.py`（35 passed）。
  - [x] 单文件空/非空/无效 terms 上传行为；环境：本地 dev server；真实入口：WebSocket `upload_request → upload_data`，确认 `error.invalid_terms` 与 task_complete；I1 入口回归合计 104 passed。
  - [x] 分片 authority 与 queue_full 重试；环境：本地 dev server；真实入口：WebSocket `upload_request → upload_chunk → finalize_upload`，确认不重传且 terms 不变；真实 ephemeral socket probe 确认文件仅保存一次、session terms 保持、最终 `upload_complete`。
  - [x] FunASR 参数链与 capability_id；环境：本地 dev server；真实入口：HTTP `/capabilities` 与 WebSocket `connected`，确认 capability_id 一致；I1 live/入口回归合计 109 passed。
- **完成条件**：I1 required unit/入口测试通过，日志无术语原文，空 terms parity 成立，并可基于主干部署；CI 证据由平台提供时纳入验收，本仓当前无 workflow/status checks，不能声称 CI 通过。

### I2：Qwen3 context 穿透

- **状态**：进行中
- **预期产出**：有效 terms 从 TranscribeOptions 穿透两套 pool/旧 worker 到 Qwen vendor；非空使用固定 context 模板，空值传 `None`。
- **当前范围**：只消费 I1 已验证 terms；覆盖 file/inproc、旧任务、diarize/word_align parity；继续统一 cache bypass。
- **关键决策**：服务端构造 `You are a helpful assistant.\nKnown terms:\n` 模板；不接受任意 prompt，不做 context fingerprint/cache。
- **已知阻塞**：无代码阻塞；真实 Qwen 模型入口证据尚未取得，需在具备模型/运行时的本地 dev server 补齐，不能用 unit 结果替代。
- **推进前必须拿到的证据**：
  - [x] 2×2×2 context unit 矩阵全绿；环境：本地 venv；命令：`FUNASR_NOTIFICATION_ENABLED=false venv/bin/python -m pytest tests/unit/test_qwen3_terms_context.py tests/unit/test_diarize_options_propagation.py tests/unit/test_transcribe_options.py tests/unit/test_result_metadata.py`（73 passed）。相关 Qwen/pool/worker/capability 整文件回归 119 passed；terms cache 回归 9 passed。
  - [ ] Qwen 实际请求入口保持空/非空 parity；环境：本地 dev server；真实入口：WebSocket 上传并轮询 task status，确认 JSON/SRT 的 speaker/words 不回归。

### I4：只读 doctor 运维诊断

- **状态**：未开始
- **预期产出**：`scripts/doctor.py --json` 任意 cwd 输出单一 JSON，报告 provider/artifact/fallback 与退出码，严格无副作用。
- **当前范围**：只读配置与工件诊断；不加载模型、不联网、不下载、不创建目录、不改变 capabilities schema。
- **关键决策**：退出码固定 0/1/2；FunASR 动态缓存为 unknown/deferred；可选 word-align 缺失仅 WARN。
- **已知阻塞**：等待 I2 合并与路线审计；解除条件：I2 context 与 capability 语义通过入口证据。
- **推进前必须拿到的证据**：
  - [ ] doctor unit/subprocess 矩阵全绿；环境：本地 venv；命令：`venv/bin/python -m pytest tests/unit/test_doctor.py tests/unit/test_http_capabilities.py`
  - [ ] 多 cwd、provider/artifact 与副作用行为；环境：本地临时目录；真实入口：从仓库根和任意 cwd 执行 `venv/bin/python scripts/doctor.py --json`，确认 stdout 仅一份 JSON。

## I3 跨仓边界

I1 协议合并且 schema 稳定后，仅在 `VideoTranscriptAPI` 创建 issue #57，交接 capability_id、fail-closed gate、KeyInfo/术语库来源和验收标准；不修改、测试或部署下游代码，且不作为本仓 I2/I4 gate。

## 路线图审计

- **审计日期 / 增量**：2026-08-07 / PR #4，合并提交 `faf2fdb`
- **里程碑真完成了吗？**：是。I1 两轮 review 均 0 finding；I1 unit/入口证据记录为 104、109、35 passed；真实 HTTP/WS capability_id 一致，真实 ephemeral WS 已验证 queue_full→finalize retry 不重传且 terms 保持。仓库无 workflow/status checks，未获得 CI 证据，不伪称 CI 通过。
- **下一个目标还是对的吗？**：是。I2 只消费 I1 已验证 terms，补齐 Qwen context 穿透后才提升 Qwen capability。
- **有没有漏掉的里程碑？**：无；I3 已由 VideoTranscriptAPI issue #57 承接，保持跨仓 issue 边界。
- **新证据是否改变了工作顺序？**：没有；I1 合并与审计完成后激活 I2，I4 仍后置。
- **done 的定义还成立吗？**：是；各阶段仍需本地测试、真实入口和可追溯证据，CI 缺口单独标注，不以本地结果替代。
- **审计结论**：保持 I1 → I2 → I4，I3 仅 issue #57，不进入本仓实现路线。
