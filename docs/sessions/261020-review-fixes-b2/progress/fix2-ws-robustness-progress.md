## 2026-10-21：先写回归测试并确认原实现变红
- 当前阶段：implementing，缺陷复现已完成。
- 本段结论：新建 `tests/unit/test_ws_robustness.py`；原实现的首次窄测为 6 failed、3 passed，失败覆盖发送无超时、越界/长度错误写盘计数、finalize 在事件循环算 MD5、按创建时间误扫活动会话。
- 关键决策与已否决方案：websockets 12.0 的 `send()` 文档不建议取消发送，写缓冲区满时应 `close()`；因此超时后关闭连接并清理关联，再把 `TimeoutError` 上抛。未加配置字段。
- 下一步唯一动作：实现发送限时、分片校验、off-loop finalize 与活动时间 sweep。

## 2026-10-21：修复实现与窄测转绿
- 当前阶段：implementing，代码修复和针对性验证完成。
- 本段结论：发送限制为 10 秒并记录 `WS-SEND-TIMEOUT`；MD5 与流式落盘复制均移出事件循环，MD5 不匹配清理会话；分片拒绝越界/类型/长度错误；活动时间覆盖创建、分片、finalize、queue_full，sweep 跳过 finalizing。窄测 11 passed，相关连接/分片/上传生命周期测试另 24 passed（合计 35 passed）。
- 关键决策与已否决方案：不把文件读成整段 bytes，也不新增抽象或配置；使用现有流式 MD5 和 `shutil.copyfile` 在 executor 中复制，保留 queue_full 后可重试所需临时文件。
- 下一步唯一动作：跑 scope 检查，提交并推送卡分支。

## 2026-10-21：卡分支提交与验收结果
- 当前阶段：实现已提交并推送；完整验收未全绿。
- 本段结论：提交 `fdc7007a178d17d986ff9c161285c78bd5fca08e` 已由 `git ls-remote` 核实；scope 检查通过；窄测 11 passed。全量 unit 为 1171 passed、6 skipped、2 failed、10 errors，其中 1 条旧 finalize 测试仍断言调用已移除的 `save_uploaded_file`，另有 Linux/macOS 原生依赖红；`scripts/run_checks.sh` 因本 worktree 无 `venv/bin/python` 未启动。
- 关键决策与已否决方案：遵守 Scope-Globs，不改卡外旧测试来适配流式文件复制；基线 timeout，继承红未能判定；不在 Linux 伪报 FunASR parity。
- 下一步唯一动作：由主脑验收旧测试断言是否需另卡修正，并在 macOS 补跑 parity。

## 2026-10-21：避免 close 握手再次阻塞 worker
- 当前阶段：实现补强完成，准备提交。
- 本段结论：发送超时后不等待最长 60 秒的 WebSocket close 握手；改为立即异步发起 close，并同步清理连接关联、上抛超时。测试以阻塞的 send 和阻塞的 close 验证 `_send_message` 仍在 0.1 秒内失败。窄测 11 passed，连接诊断与终态通知测试 20 passed。
- 关键决策与已否决方案：close 继续由 websockets 12 的协议超时机制收尾，但不让 task worker 等待它；不做无界后台清理抽象。
- 下一步唯一动作：提交并推送该补强，复核远端分支与工作树状态。

## 2026-10-21：收尾确认
- 当前阶段：实现已提交并推送；验收因完整套件红未能全绿。
- 本段结论：最终实现提交 `3f5242f8c9fd0421622d4a35e5dc1aa099f4235a` 在远端分支已核实；scope 检查通过，窄测 11 passed，阻塞 close 的诊断/终态相关测试 20 passed，工作树干净。
- 关键决策与已否决方案：完整 unit 的 1 条旧测试断言 `save_uploaded_file` 调用，和流式复制修法冲突且该测试文件不在 Scope-Globs；不越界修改。基线 timeout，继承红仍未能判定；Linux 不运行 macOS parity。
- 下一步唯一动作：主脑验收已推送卡分支，并决定旧测试适配与 macOS parity 的后续安排。

## 2026-10-21：最终全量 unit 复核
- 当前阶段：实现已推送，验收阻塞项已归档。
- 本段结论：最终代码全量 unit 复跑为 1171 passed、6 skipped、3 failed、10 errors；除旧 `save_uploaded_file` mock 断言与 macOS 原生依赖失败外，还出现 1 条未改动的进程池工作槽分配时序失败。窄测仍为 11 passed。
- 关键决策与已否决方案：进程池测试只在全量并发运行中失败，不修改卡外文件；本卡基线 timeout，不能把任何平台/时序红宣称为继承红。
- 下一步唯一动作：主脑基于报告决定是否另派旧测试更新，并在 macOS 跑 FunASR parity。
