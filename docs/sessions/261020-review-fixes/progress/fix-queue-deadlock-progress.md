# fix-queue-deadlock 进度存档

卡: 可重试失败重入有界队列导致 worker 死锁（src/core/task_manager.py）
基线: 7b8bfaacd4b242522864cd271fb63b14e166ff11
分支: card/fix-queue-deadlock

## 段 1 — 2026-10-20 红验（复现缺陷）

- 当前阶段: implement（红）
- 本段结论: 新增 `tests/unit/test_task_manager_retry_no_deadlock.py`，用真实 TaskManager（只 mock 转写器）复现死锁：`max_queue_size=2` 填满队列、2 个 worker 各撞一次可重试错误后，旧实现的 A/B 两个任务 5s 内都停在 `PENDING`（worker 卡在 `await task_queue.put`），服务停转。红验成立。
- 关键决策与已否决方案: 判据不用 sleep 赌时序 —— 用 gate 让两个 worker 先卡在转写里，测试再把队列填满（死锁条件确定）。锁死两条不变式：① 5s 内全部任务到终态；② 任务 id 从未 put 回有界队列（`_SpyQueue` 记录 put 调用）。已否决（卡面锁定，不重提）：无界队列、put_nowait 失败即 FAILED。
- 下一步唯一动作: 去掉可重试失败分支的阻塞 put，改为当前 worker 内原地重跑。

## 段 2 — 2026-10-20 绿验（修完 + 全量 unit）

- 当前阶段: implement（绿）
- 本段结论: `_process_task` 改为「单次尝试 + 返回是否需重跑」，重试循环上移到 `_worker`（当前 worker 内原地重跑，不再 `task_queue.put`）。新增测试 5 条全绿；`tests/unit/` 全量 1128 passed / 1 failed / 10 errors，与基线（1123 passed / 1 failed / 10 errors）同源：那 1 failed + 10 errors 是 macOS dylib 在 Linux 上的继承红（qwen3 coreml 编码器），非本卡引入。
- 关键决策与已否决方案: ① 重试循环放 `_worker` 而非 `_process_task` 内部 —— 既有 `tests/unit/test_worker_error_classification.py` 断言「可重试失败后任务状态为 PENDING（重试重入队）」，该文件在本卡 Scope 外不能改，把重试执行放在 `_process_task` 内会让它变终态 FAILED 而变红；放在 `_worker` 里则单次尝试后仍是 PENDING，语义与旧实现一致。② 重试日志升级为 WARNING 并带 `TASK-RETRY-INLINE` 标记（可 grep 的降级信号），保留原「任务重试 {id}: 第{n}次」文案。③ 保留 `retry_count` 上限、funasr MODEL_ERROR 的 `_try_reset_model` + `sleep(2)`、状态/进度通知语义不变。已否决：`_process_task` 内自递归重跑（栈式重试语义不透明）、无界队列、put_nowait 失败即 FAILED。
- 下一步唯一动作: 主脑验收（Mac 上补跑 `bash scripts/run_checks.sh --parity`，本机 Linux 跑不了 integration）。
