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
