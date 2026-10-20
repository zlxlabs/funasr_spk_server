## 2026-10-20 · cluster_merge off-loop 实现与验证

- **当前阶段**：实现完成，等待范围核验并提交。
- **本段结论**：`Qwen3DiarizeTranscriber.transcribe()` 通过 `run_in_executor` 执行 cluster_merge；公开入口测试确认重计算在线程池运行、期间事件循环仍推进，并锁定失败日志标记。新测试窄验证通过；全量 `run_checks.sh` 为 1153 passed、6 skipped、1 failed、10 errors，失败集中在 Linux 上的 macOS/CoreML/llama 动态库用例；派发时基线不可用，继承红无法判定。
- **关键决策与已否决方案**：保留原参数、结果与现有 try/except 降级语义，仅以 executor 的 lambda 承载关键字参数；不新增 helper、配置或重试。已用只改断言谓词的最小注入验证测试转红并恢复原断言。
- **下一步唯一动作**：对最终变更执行卡面 Scope-Globs 核验，然后显式路径暂存并提交、推送本卡分支。
