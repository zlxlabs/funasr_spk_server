## 2026-10-21：回归复现与修复

- 当前阶段：实现完成；目标测试与 HTTP/WebSocket 可观测性测试通过，待提交和推送。
- 本段结论：真实 `websockets.serve` + `ws_handler.handle_connection` 测试在修复前因缺少 `path` 收到 1011；改为单参数包装器并显式传入 `websocket.path` 后，真实 handler 发送 `connected`，既有连接限流用例仍通过。全量脚本在 Linux 环境中有 1 个 macOS 专用配置测试失败、10 个 vendor 动态库 setup 错误；通知配置未显式关闭时另有 8 个收集错误。
- 关键决策与已否决方案：按锁定修法保留 handler 签名，不修改 `src/api/websocket_handler.py`；不引入 fallback、抽象或协议改动。主干基线不可用，以上全量失败的继承状态未能判定。
- 下一步唯一动作：提交三个 Scope-Globs 文件，随后运行卡片范围检查并推送本卡分支。

## 2026-10-21：验证与交付

- 当前阶段：代码已提交并推送至 `card/fix3-conn-wrapper-path`，等待主脑验收。
- 本段结论：窄测 2/2 通过，`test_http_endpoints_live.py` 3/3 通过；开启通知配置开关后全量单测为 1206 passed、6 skipped、1 failed、10 errors，失败集中于 Linux 无 macOS vendor 动态库/配置测试；Scope-Globs 检查通过，远端分支 SHA 为 `1f09d0129e8dce6435bc60cf10d00058daddc7a1`。主干基线不可用，继承红状态未能判定。
- 关键决策与已否决方案：Linux/macOS 专用测试失败不在本卡范围，不通过修改配置或 vendor 代码绕过；目标连接路径由真实 WebSocket 测试验证。
- 下一步唯一动作：由主脑验收已推送提交。
