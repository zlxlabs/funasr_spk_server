# fix4-gpu-mem 进度

## TDD 与实现
- 当前阶段：实现完成；窄测通过，全量检查已执行。
- 本段结论：新增 `process_vram_mib(pid)` 公共探针与 sidecar `running_pid` 只读属性；Qwen3 预检仅使用公开接口。删除配置测试模块中的重复 fixture，四个 fixture 使用点所在测试全部通过。全量 unit 检查有一个既存测试失败：`tests/unit/test_qwen3_transcriber_b3_fixes.py::test_preflight_counts_free_vram_after_excluding_live_sidecar_usage` 仍 patch 被本卡移除的私有导入；该文件不在本卡 Scope-Globs，未修改。
- 关键决策与已否决方案：PID 未出现在成功查询结果时占用量返回 `0`；探针失败/超时返回 `None` 并记录 `GPU-MEM-PID-PROBE-FAILED`。遵守范围锁定，不为兼容过期测试在转写器中保留私有导入或访问 `_proc`。
- 下一步唯一动作：运行范围检查，提交并推送当前卡分支，写入最终验证报告。

## 续派：更新旧预检测试
- 当前阶段：修复完成；全量 unit 检查通过。
- 本段结论：`test_preflight_counts_free_vram_after_excluding_live_sidecar_usage` 改用 `client.running_pid` 与 `process_vram_mib` 打桩，保留“加回存活 sidecar 占用后再判阈值”的行为断言；测试 helper 不再设置私有 `_proc`。`bash scripts/run_checks.sh` 结果为 1226 passed、6 skipped。
- 关键决策与已否决方案：不放宽、不删除原预检行为用例；修订测试边界到已公开接口，不改生产实现。
- 下一步唯一动作：提交并推送续派测试修订，更新最终报告并核验远端分支。
