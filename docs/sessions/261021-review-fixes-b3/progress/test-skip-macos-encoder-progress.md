# macOS 专用 Qwen3 encoder 单测跳过进度

- 当前阶段：实现与验证完成，待提交与推送。
- 本段结论：Linux 基线 11 条目标失败已逐项复现；三处配置测试改用预加载真实 schema、mock llama 原生模块，保持平台决策测试执行；两个 CoreML encoder 测试模块仅在非 darwin 显式 skip。目标测试 10 passed、10 skipped；unit 全量 1207 passed、16 skipped；`scripts/run_checks.sh` 在继承 `FUNASR_NOTIFICATION_ENABLED=false` 后通过。
- 关键决策与已否决方案：配置字段的 auto 平台断言不能 skip，因为它不需要真实 CoreML，改为 mock 原生 llama 导入并继续运行真实配置 schema；CoreML encoder 模块因 import 会进入真实平台原生依赖，只按 `sys.platform != "darwin"` 跳过；不改 `src/`，不扩展到其它测试。
- 下一步唯一动作：执行 scope 检查并提交、推送本卡限定文件。
