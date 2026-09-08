# Pool recovery 验证进度

## 冻结状态

- 功能基准为 `a24d79332761f92c330d388123aef85390e67234`。
- `11f4695` 仅为删空行/注释凑预算，已用 revert `69cabd3ae6e51de09e37154f32e49f9c22c8ec45` 撤回。
- 当前分支为 `feat/funasr-pool-recovery-0909`，revert 已推送 PR #11；未部署、未标 ready。

## 全量 unit 结果

- 命令：`FUNASR_NOTIFICATION_ENABLED=false /home/zlx/projects/personal/funasr_spk_server/venv/bin/python -m pytest -q tests/unit`。
- 结果（a24）：`1101 passed, 6 skipped, 10 errors, 1 failed`。
- 精确失败：`tests/unit/test_config_qwen3_asr_encoder_provider.py::TestBuildEngineConfigReadsConfigField::test_auto_on_macos_resolves_to_coreml_ane_fe`，`OSError`，Linux 加载 Darwin `libggml.dylib` 报 `invalid ELF header`。
- 6 个 full encoder error：`TestCoremlAneFullOnMacos::test_macos_fe_coreml_be_mlpackage`、`TestCoremlAneFullFallback::test_linux_fallback_cpu_full`、`TestCoremlAneFullFallback::test_macos_mlpackage_missing_fallback_to_ane_fe`、`TestCoremlAneFullFallback::test_macos_no_coreml_ep_fallback_cpu`、`TestExistingBranchesUnchanged::test_coreml_ane_fe_unchanged`、`TestExistingBranchesUnchanged::test_cpu_unchanged`；均 setup `OSError`，Linux 缺 `bin/libggml.so`。
- 4 个 provider error：`TestCoremlAneFeOnMacos::test_macos_uses_coreml_fe_and_cpu_be`、`TestCoremlAneFeFallback::test_linux_fallback_cpu_only`、`TestCoremlAneFeFallback::test_macos_no_coreml_ep_fallback_cpu`、`TestExistingBranchesUnchanged::test_default_cpu_still_works`；均 setup `ImportError: cannot import name 'inference'`。
- 当前精确复跑输出：`/tmp/funasr-pool-current-unit-failures.log`；共 20 项，`9 passed, 10 errors, 1 failed`，错误类型与上述一致。

## base 对照与反向红验

- 在 base `43e8e94` 临时隔离 worktree 用同一 Linux venv 跑上述 3 个完整对应文件：同为 `9 passed, 10 errors, 1 failed`。
- base 对照输出：`/tmp/funasr-pool-base-unit-failures.log`。
- base 的 6 个 full encoder error 同为缺 `bin/libggml.so` 的 `OSError`；4 个 provider error 同为 `ImportError`；唯一 failure 同为 Darwin dylib `invalid ELF header` 的 `OSError`。
- 仅拷贝新增 `tests/unit/test_server_shutdown.py` 到 base 运行，得到预期 `1 failed`，`AssertionError`：旧代码事件顺序以 `task_stop` 先于 `ws_close`；输出：`/tmp/funasr-pool-base-shutdown-red.log`。
- 两个临时 worktree 均已移除，未改 base 或其他会话。

## 预算与剩余验证

- 按 base 到当前 revert 后 HEAD 的 numstat：源码新增 575、测试/fixture 新增 742、design 文档 53、progress 文档 34；整批新增 1404、删除 291，低于整批 1800。
- 5 次完整定向组此前每次 61 passed；功能收口后的生命周期/worker/server 定向复跑 43 passed。
- Linux 环境限制了 CoreML/llama vendor 测试；Mac 真机 worker/长音频验证由后续卡执行。
