# macOS 专用 Qwen3 encoder 单测跳过进度

- 当前阶段：实现与验证完成，待提交与推送。
- 本段结论：Linux 基线 11 条目标失败已逐项复现；三处配置测试改用预加载真实 schema、mock llama 原生模块，保持平台决策测试执行；两个 CoreML encoder 测试模块仅在非 darwin 显式 skip。目标测试 10 passed、10 skipped；unit 全量 1207 passed、16 skipped；`scripts/run_checks.sh` 在继承 `FUNASR_NOTIFICATION_ENABLED=false` 后通过。
- 关键决策与已否决方案：配置字段的 auto 平台断言不能 skip，因为它不需要真实 CoreML，改为 mock 原生 llama 导入并继续运行真实配置 schema；CoreML encoder 模块因 import 会进入真实平台原生依赖，只按 `sys.platform != "darwin"` 跳过；不改 `src/`，不扩展到其它测试。
- 下一步唯一动作：执行 scope 检查并提交、推送本卡限定文件。

## 主脑验收反馈第 1 轮后的修正
- 当前阶段：已移除两处 encoder 测试模块级 skip，等待最终验证、scope 检查与提交推送。
- 本段结论：在 tests/unit/conftest.py 共用 fixture，仅非 darwin 时用 sys.modules mock 原生 inference.llama 导入；保留两个 encoder 模块的真实 QwenAudioEncoder 测试体与 ONNX mock。Linux 窄测 20 条全部通过、无 skip。
- 关键决策与已否决方案：保留第一条已验收的 config 测试处理；不再整模块 skip，因为这会抑制 Linux fallback/CPU 分支验证；不改 src/，不加新测试文件。
- 下一步唯一动作：跑卡面指定全量 -rs 与验收脚本，随后 scope 检查、commit+push 并报告。
