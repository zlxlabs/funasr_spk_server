<!-- delegate-outcome: succeeded -->

# R2 审查结论：FAIL

## 结论摘要

固定审查范围为 `43e8e949d8a860cdc9ab0b58e50a20f279d90861..a24d79332761f92c330d388123aef85390e67234`，并按设计说明检查了目标提交依赖的完整多进程池恢复、原子结果发布、任务管理器和服务收停路径。结论为 **FAIL**：发现 1 个 P1、1 个 P2，未发现 P3。

P1 是真实用户取消没有传递到实际的池调用协程：任务记录会进入 `CANCELLED`，但 worker 进程和池调用仍继续运行，迟到结果还可以把任务改回 `COMPLETED` 并发送完成通知。P2 是池资源回收失败被任务管理器按普通可重试引擎错误重新入队，而失败回收留下的 busy lease 又不会让池获取快速失败，重试可能永久等待同一 slot。

## 审查范围与判定依据

- 基线（H0）：`43e8e949d8a860cdc9ab0b58e50a20f279d90861`
- 目标（H1）：`a24d79332761f92c330d388123aef85390e67234`
- 设计依据：`docs/sessions/funasr-pool-recovery-20260909/design.md`
- 重点不变式：每个调用的 process/task/audio/result 归属；success/error/timeout/crash/cancel/shutdown 六条状态轴；取消和超时必须回收自有资源；发布后的结果必须是完整原子文件；worker 崩溃不在池内重试；cleanup 期间不得复活池。
- 目标提交本身只规范 worker 发布日志；因此本轮没有把审查限制成两行 diff，而是核对了该提交交付所依赖的完整池生命周期和调用方边界。

## Findings

### P1-1：用户取消不取消实际池调用，迟到结果可覆盖取消终态

**位置**

- `src/api/websocket_handler.py:493-505` 的取消入口只调用 `task_manager.cancel_task(task_id)`。
- `src/core/task_manager.py:600-618` 的 `cancel_task()` 只把任务记录改为 `CANCELLED`、记录错误并删除任务文件；没有保存或取消正在执行的 `_process_task()` 协程句柄。
- `src/core/file_based_process_pool.py:678-685` 只有在 `generate_with_pool()` 自身收到 `asyncio.CancelledError` 时才会调用 `_release_worker_lease(..., force=True)`，从而终止 process 并清理归属文件。
- `src/core/task_manager.py:787-802` 在转录返回后无 `CANCELLED` 守卫，直接把任务改为 `COMPLETED`、发送完成通知并执行完成路径。

**影响**

取消接口返回成功只代表内存中的任务状态被改写，不代表实际推理停止。长音频取消后仍占用池 slot、继续消耗模型和 GPU/CPU 资源；结果返回后还会覆盖用户已经看到的取消终态并发送错误的完成通知。取消时删除源文件也与仍在运行的转录存在竞态，具体表现取决于 transcriber 是否已经复制或打开该文件。

**证据**

1. 使用 H0 的真实 fake subprocess 路径调用 `TaskManager.cancel_task()`：返回 `True`，任务状态从 `PROCESSING` 变为 `CANCELLED`，但取消后 worker PID 仍存活；只有直接取消处理协程后，池的 `CancelledError` 路径才执行回收。
2. 使用 fake transcriber 让处理协程在取消后迟到返回：`cancel_task()` 后释放调用，任务最终被改为 `COMPLETED`，并产生 1 次完成通知。
3. H0 的取消测试在基线 worktree 上复跑时，真实进程存活断言失败；该测试不是恒真断言。

**与设计的冲突**

这违反设计说明第 22-24 行关于调用取消时退休原 process、清理自有资源和唯一调用终态的要求，也违反第 35-38 行 cancel 轴和第 53 行“取消后不接受迟到结果”的预期。

**最小修复方向**

在 `TaskManager` 已有的每任务处理边界保存在途 `_process_task()` 协程句柄；用户取消时取消该句柄，使 `generate_with_pool()` 的既有 `CancelledError` 回收路径真正执行。正常结果、失败和通知提交前增加针对 `CANCELLED` 的终态守卫，禁止迟到结果覆盖取消状态。源文件删除要与这条取消/池复制窗口协调，不能在仍可能被调用方读取时先删除。不要通过新增 fallback、池内重试或另一层抽象绕过生命周期问题。

### P2-1：回收失败后的 lease 被保留，普通重试会无限等待 slot

**位置**

- `src/core/file_based_process_pool.py:520-549` 的 `_finish_worker_lease_release()` 在 `_reap_worker_process()` 失败时保留 `_worker_tasks[worker_id]`，这是为了避免尚未证明退出的 process 被误删或复用。
- `src/core/file_based_process_pool.py:471-500` 的 `_acquire_worker_lease()` 没有等待上限；已有 busy lease 时循环调用 `_ensure_workers_alive()` 并持续 sleep。
- `src/core/file_based_process_pool.py:451-453` 的 health 判定会跳过仍在 `_worker_tasks` 中的 busy worker，因此不会把该 lease 当作可修复的空闲退出 slot。
- `src/core/task_manager.py:810-835` 把普通 `RuntimeError` 分类为可重试的 engine error，增加 `retry_count` 后重新放入队列。

**影响**

当进程确实无法被回收时，第一次处理可能被重新排队；下一次处理在同一池实例上看不到可用 slot，既不会得到明确终态，也不会触发普通的有限次任务重试，而是长时间/无限等待。未删除仍可能使用的目录这一安全选择是合理的，但上层不能把它伪装成可安全重试的普通引擎错误。

**证据与置信度**

注入 H0 的 `_reap_worker_process` 失败后，第一次 `_process_task()` 将状态分类为 `engine_error`，`retry_count=1`、状态回到 `PENDING`、队列仍有 1 项且 PID 仍存活；第二次处理等待约 0.15 秒后仍停在 `PROCESSING`，原因是 `_worker_tasks` 保留 lease，`_acquire_worker_lease()` 没有上界。恢复 reap 后 cleanup 可以成功。该证据是受控的回收失败注入，尚未在生产环境制造真实 OS reap failure，因此定为 P2 而非 P1。

**与设计的冲突**

设计说明第 10 行明确排除池内无限重试，第 24 行要求崩溃/调用失败交由 `TaskManager` 的既有有界策略处理；当前错误分类与 busy lease 的组合使这个边界失效，也不满足第 53 行每种池调用失败都有唯一终态的预期。

**最小修复方向**

让“资源尚未成功回收/lease 不可用”保持为可见的资源生命周期错误，不走普通引擎错误的重试入队；或由上层把它显式终态化，同时保留未证明归属安全的目录和 process，交给后续明确的 cleanup/reap 路径处理。不要为此增加池内无限重试或无边界 acquire 等待。

## 已验证通过的边界

以下定向检查在 H0 实现上通过，不能抵消上述调用方缺口：

- `tests/unit/test_file_based_pool_lifecycle.py` 与 `tests/unit/test_server_shutdown.py`：24 passed。
- `tests/unit/test_qwen3_worker_process.py`、`tests/unit/test_funasr_hotword_propagation.py`、`tests/unit/test_qwen3_pool_transcriber.py`：37 passed。
- 真实 fake subprocess 的完整 argv/task bytes、atomic task/result/error 发布、PID/slot/task 归属校验、exit-after-publish 复查、正常 timeout/cancel/crash 的 direct pool 回收、并行 cleanup、shutdown 顺序均有定向绿证。
- server stop 的顺序测试通过：先关闭 websocket，再停止 task manager，再 cleanup transcriber/pool，最后等待 websocket 关闭。
- watchdog 只把任务状态改为 `TIMED_OUT`，不立即中断 MPS；这是设计说明第 47 行和现有可观测性契约声明的边界，本轮不另列为新 finding。

## 测试与工具说明

本轮没有修改源码或测试。未运行完整 pytest；项目给出的 Linux vendor 环境已有已知的 `9 passed / 10 errors / 1 failed`，本轮使用与问题直接相关的定向单测和受控探针。OCR review 工具因输入长度限制只返回 `status=partial`、`coverage=partial`，且部分 finding 读取的是当前 base 而非 H0，因此其输出仅作辅助信号，未作为“通过”或额外 finding 的依据。

## 最终建议

在 P1-1 修复并补上“用户取消后真实 PID 已退出、owned artifacts 已回收、迟到结果不会改回完成”的跨边界测试前，不应接受本次恢复方案。P2-1 需要同步修复错误分类/终态与资源回收契约，并覆盖注入 reap failure 后不会无限等待的测试；修复后重新跑本轮定向测试，再按设计说明要求在真实 macOS venv 执行完整 parity 和长音频验收。
