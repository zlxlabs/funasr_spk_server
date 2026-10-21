## 进度一：实现与定向测试

- 当前阶段：实现中，行为与相邻进程池测试已通过。
- 本段结论：公开入口测试以真实 fake worker 子进程复现成功推理被 reap 的 `OSError` 覆盖；修复后成功结果仍返回、错误日志包含稳定标记与任务/worker/异常信息。失败 result 仍释放 lease 后抛错；已删除无调用者的 `_is_worker_alive`。
- 关键决策与已否决方案：只在成功 result 分支捕获 lease 回收异常，保留 `_worker_tasks` 中的 lease 归属供 `cleanup()` 回收；不重试、不调整超时或 worker 生命周期。失败 result 分支保留原有 fail-fast 语义。
- 下一步唯一动作：对成功结果返回判据做单行反向注入，确认新增测试以断言失败。
