## 2026-10-21：回归复现与修复

- 当前阶段：实现完成；目标测试与 HTTP/WebSocket 可观测性测试通过，待提交和推送。
- 本段结论：真实 `websockets.serve` + `ws_handler.handle_connection` 测试在修复前因缺少 `path` 收到 1011；改为单参数包装器并显式传入 `websocket.path` 后，真实 handler 发送 `connected`，既有连接限流用例仍通过。全量脚本在 Linux 环境中有 1 个 macOS 专用配置测试失败、10 个 vendor 动态库 setup 错误；通知配置未显式关闭时另有 8 个收集错误。
- 关键决策与已否决方案：按锁定修法保留 handler 签名，不修改 `src/api/websocket_handler.py`；不引入 fallback、抽象或协议改动。主干基线不可用，以上全量失败的继承状态未能判定。
- 下一步唯一动作：提交三个 Scope-Globs 文件，随后运行卡片范围检查并推送本卡分支。
