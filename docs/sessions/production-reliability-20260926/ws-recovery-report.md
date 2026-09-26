# B 卡：WebSocket 断线恢复与诊断

- 任务：`funasr_spk_server-20260926-B`
- 分支：`feat/funasr-ws-recovery-0926`
- 基线：`ec0b892035ff1def7e1168254b5b89622c5e7bae`
- 初始实现提交：`692e9d43a8edd6adccf55980883ff1111af1174e`
- 最终业务冻结 SHA：`e602f796a8b4d51c8681ede567f8b9422b52f40f`

## 结果

- 收到上传 ack 后，客户端可保留 `task_id`；重连使用既有 `task_status_batch` 查询。JSON 终态从 `items[].result` 取回，SRT 从 `items[].srt_content` 取回；miss/expired 按现有协议凭原 `file_hash` 和选项重新提交。
- 异常日志写入稳定的 `connection_ref` 哈希、可用的 `task_id`、处理阶段、消息类型、异常类和 close code；不写异常正文或远端 close reason。异常连接用 WARNING 保持可见，消息处理和通知失败仍清理对应连接，单个坏连接不会阻断其他连接。
- `examples/transcribe_media.py` 未改动，也没有证据说明现有外部客户端已采用重连流程。

## 验证

- 定向命令（使用主仓 venv、当前工作树 cwd）：`FUNASR_NOTIFICATION_ENABLED=false /home/zlx/projects/personal/funasr_spk_server/venv/bin/python -m pytest -q tests/unit/test_i1_websocket_protocol_live.py tests/unit/test_websocket_connection_diagnostics.py tests/unit/test_websocket_task_status_batch.py tests/unit/test_ws_terminal_failure_message.py`：**37 passed**。
- 日志测试先让进度、完成、失败通知和消息处理抛出带敏感标记的异常；旧实现因此有 5 项失败（含 close 日志缺少异常类/close code 的断言）。旧实现没有回显远端 close reason；新测试继续用敏感 reason 锁定此约束，并按真实 formatter 输出检查关联字段、异常类和 close code。
- 临时 fault injection 将 batch 完成结果清空：JSON 与 SRT 两个恢复用例都按预期失败；模拟断开时取消任务：两个用例都因 `cancel_task` 被调用而失败。注入文件已删除，工作树代码未被突变。
- `git diff --check` 通过。
- 全量命令：`FUNASR_NOTIFICATION_ENABLED=false FUNASR_DATA_DIR=/tmp/funasr-ws-recovery-tests/data FUNASR_LOG_DIR=/tmp/funasr-ws-recovery-tests/logs TMPDIR=/tmp/funasr-ws-recovery-tests /home/zlx/projects/personal/funasr_spk_server/venv/bin/python -m pytest -q tests/unit`：**1110 passed, 6 skipped, 1 failed, 10 errors**。失败/错误集中在 Qwen3 Mac CoreML/vendor 动态库路径；主脑报告A卡基线也有相同平台类问题，本卡未改相关代码。

失败与错误的完整 nodeid：

```text
tests/unit/test_config_qwen3_asr_encoder_provider.py::TestBuildEngineConfigReadsConfigField::test_auto_on_macos_resolves_to_coreml_ane_fe
tests/unit/test_qwen3_encoder_coreml_ane_full.py::TestCoremlAneFullOnMacos::test_macos_fe_coreml_be_mlpackage
tests/unit/test_qwen3_encoder_coreml_ane_full.py::TestCoremlAneFullFallback::test_linux_fallback_cpu_full
tests/unit/test_qwen3_encoder_coreml_ane_full.py::TestCoremlAneFullFallback::test_macos_mlpackage_missing_fallback_to_ane_fe
tests/unit/test_qwen3_encoder_coreml_ane_full.py::TestCoremlAneFullFallback::test_macos_no_coreml_ep_fallback_cpu
tests/unit/test_qwen3_encoder_coreml_ane_full.py::TestExistingBranchesUnchanged::test_coreml_ane_fe_unchanged
tests/unit/test_qwen3_encoder_coreml_ane_full.py::TestExistingBranchesUnchanged::test_cpu_unchanged
tests/unit/test_qwen3_encoder_provider.py::TestCoremlAneFeOnMacos::test_macos_uses_coreml_fe_and_cpu_be
tests/unit/test_qwen3_encoder_provider.py::TestCoremlAneFeFallback::test_linux_fallback_cpu_only
tests/unit/test_qwen3_encoder_provider.py::TestCoremlAneFeFallback::test_macos_no_coreml_ep_fallback_cpu
tests/unit/test_qwen3_encoder_provider.py::TestExistingBranchesUnchanged::test_default_cpu_still_works
```

## 边界与待办

- 恢复测试使用真实 loopback WebSocket server、真实帧序列化和 handler，但任务管理器与模型执行由 fake manager/测试结果模拟；它证明连接断开后 handler 不发取消、不重投，且可从同一任务对象序列化取回 JSON/SRT。它没有验证真实模型或生产 worker 的持续执行。
- 默认终态保留时间与数量上限由配置定义为 3600 秒、500 条；环境变量可覆盖，达到数量上限时旧终态可能提前移除。
- 本仓没有 GitHub Actions workflow，不报告 CI 结果。未连接或操作生产环境。

## OCR 续修

- OCR 状态为 `reviewed`；独立复核为 `partial`（13 项候选中 12 项已复核、1 项超时）。本轮只修复主脑采纳的两项诊断缺陷：消息处理中遇到连接关闭时直接交外层分类；重连任务日志按严格 UUID 格式保留标识，不再依赖已清理的连接映射。批量查询仅在 `task_ids` 恰有一个字符串时关联该 ID；列表形状与非 UUID 文本记录为 `task_id=None`。
- 新增测试在修复前有 2 项失败（合法 UUID 被丢弃、连接关闭误入 `message_handler`）；修复后，受影响四个测试文件 **40 passed**。日志断言检查实际 formatter 输出，并确认敏感关闭原因、任意 ID 文本不入日志，连接关闭只分类一次且不发送 `message_error`。
- 这轮只复核了上述异常诊断路径；原恢复测试仍使用 fake TaskManager 与模拟模型结果，不代表真实生产 worker 或模型任务持续执行已验证。

## 最终验证与审查

- Linux 最新定向结果：上述四个 WebSocket/task status 测试文件 **40 passed**。Mac 冻结组合为 A `4e49d35` + B `e602f79`；Mac 上恢复与诊断两个测试文件 **17 passed, 1 warning in 0.58s**，日志见主脑白名单回读的 `/tmp/funasr-result-contract-4e49d35/logs/B-targeted-unit.log`。
- Mac 组合验证中，真实 FunASR 服务子进程的 JSON、SRT、缓存命中 3 项，以及 3 项模型 parity 测试通过。Mac 原仓 integration 42 项为 18 passed、7 skipped、4 failed、13 errors；17 个 Qwen 失败节点与基线 `ec0b892` 同环境的 17/17 完全一致，基线为 4 failed、13 errors（123.53s）。
- 首次合并运行 43 项时额外有 1 个临时 probe 失败：之前的 Qwen 测试更改全局 engine，导致 `resolve_funasr` 拒绝；该 probe 单独与 3 个 golden 测试运行共 4 passed，确认是测试间状态污染，不是业务改动。
- Linux 全量 unit 的 11 个失败/错误节点与 base、A H0 保存的基线节点集合逐项一致；这只说明已测失败子集匹配基线，不代表全量 unit 全绿。独立静态审查结论为 P1/P2/P3 均 0；R2 运行时审查结论为 `failure-visibility: clean`，审查证据分别见 `reviews/B-r1-verdict.md` 与 `reviews/B-r2-verdict.md`。
- Mac 真实服务 E2E 没有断线步骤，断线恢复用例仍使用 fake TaskManager 和模拟模型结果；两类证据不能合并成真实生产 worker 断线后持续执行已验证。本仓没有 GitHub Actions workflow，也未部署或操作生产。
