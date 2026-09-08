# DESIGN-note：Mac 多进程池恢复与结果发布收口

## 目标

长音频在 Mac MPS 上由池内互不干扰的 worker 完成；文件池调用超时、worker 崩溃或服务收停后可判定调用终态并回收本次资源，客户端只读到完整结果。

## 非目标

- 不更换 FunASR/Qwen3 引擎、依赖、模型、600 秒最小超时或业务缓存字段。
- 不引入新的池抽象、fallback、配置层或池内无限重试；重试仍由 `TaskManager` 既有有界策略决定。
- 不删除或清理已有共享 Darwin/MPSGraph 缓存；不把一次 20 次目录元数据对照外推成 600 秒超时的全部因果解释。
- 不改变 CUDA/in-proc 路径和非池化 FunASR 语义；Darwin suffix 只用于 Mac FunASR worker，Qwen3 保持既有 CoreML/Metal 用户缓存位置。

## 为什么不是分区 / 删除 / 约定

- **分区**：任务文件目录已按引擎分开，但 Mac FunASR worker 的 Darwin Foundation/MPSGraph 仍会落到用户共享目录；只有每个 FunASR worker 的 Darwin User/Temp/Cache suffix 才能隔离进程级缓存和 cleanup 所有权。Qwen3 不改变既有用户缓存位置。
- **删除**：不能删除池或超时/崩溃恢复；这些是长音频服务可用性的必要路径。不可证明归属的共享缓存清理必须删除，避免误伤其他进程。
- **约定**：文件名约定不能阻止健康巡检与派发同时替换同一 slot，也不能证明 `.pkl` 已写完；锁内状态迁移、进程生命周期和临时文件原子发布必须由运行时保证。

## 方案要点与已否决方案

- **要点**：`FileBasedProcessPool` 为每个 worker 记录 slot、process 与当前 task；Darwin 私有目录归属只保存在对应 `Popen`，lease 和 cleanup 从该 process 读取，避免重复同步。分配只选择空闲且 ready 的 slot，并在同一管理锁内完成占用。替换时先确认原 process 退出（`terminate → wait → 必要时 kill → wait`），再发布新 process；仅清理该 process 创建的 task/audio/result（含 Qwen3 被 kill 前可能留下的 `.converted.wav`）和 FunASR Darwin 私有三根目录。初始化、cleanup、health、generate 共用生命周期锁，进行中的 cleanup 禁止新任务复活池，cleanup 完成后显式 `initialize()` 仍可重启。
- **要点**：父进程把任务 JSON 通过显式临时路径写入并 `os.replace` 为 `.task`；FunASR `worker_process.py` 与 Qwen3 `qwen3_worker_process.py` 都把 result/error 写到同目录临时 `.pkl`（FunASR 的 JSON transport 也原子写 `.result`）后 `os.replace`，反序列化后校验 `task_id` 与 worker PID/slot 归属再接受，读取后按所有权删除。
- **要点**：池调用超时、取消和 worker 崩溃都先退休原 process、清理自有资源、记录唯一调用失败；崩溃不在池内重提或重置 deadline，交由 `TaskManager` 既有有界策略。health 只修复无任务的退出 slot，busy 不抢占。
- **已否决**：继续使用共享 `/var/folders/.../T/com.apple.MetalPerformanceShadersGraph` 并在 cleanup 中 `rmtree`；实测目录存在约 462 MB 共享状态，无法归属。只改 `TMPDIR` 也否决，实测 Foundation 路径不随其改变。
- **已否决**：health 锁外检查后再 spawn、轮询轮询分配；已有时序会让 health 与 generate 对同一 slot 重复 replace，且 D 可撞忙 A 而漏掉空闲 1。
- **已否决**：结果直接写最终 `.pkl`、父进程轮询 `exists()` 即读取；写入过程可被观察到，不能满足跨进程完整性。JSON 结果模式仍保留，与 pickle 一样原子发布。

## 关键不变式

下表中的源码位置和测试名是实现卡的锁定点；`[实测]` 是本轮生产/探针证据，所有测试名称均已落地。

| 轴 | success | error | timeout | crash | cancel | shutdown |
|---|---|---|---|---|---|---|
| 进程归属 | `generate_with_pool` 校验 task/PID；`test_real_subprocess_receives_argv_and_complete_task_bytes` | producer 错误 payload 带 task/PID；`test_funasr_worker_publishes_json_error_atomically` | 旧 PID reap 后才清理；`test_timeout_reaps_process_before_cleaning_owned_files` | 原进程退出且不池内重试；`test_worker_crash_is_reported_without_pool_retry_or_deadline_reset` | cancel 退休原进程；`test_cancel_reaps_process_and_cleans_owned_files` | cleanup 置关闭闸且不复活；`test_generate_does_not_revive_pool_after_explicit_cleanup`、`test_cleanup_waits_for_generate_release_before_deleting_run_dir` |
| task/audio/result | atomic task/result 发布并清理；`test_real_subprocess_receives_argv_and_complete_task_bytes` | JSON error 原子发布并回读 payload；`test_funasr_worker_publishes_json_error_atomically` | 只清本次资源；`test_timeout_reaps_process_before_cleaning_owned_files` | run 目录按实际路径核对；`test_worker_crash_is_reported_without_pool_retry_or_deadline_reset` | cancel 清理 owned artifacts；`test_cancel_reaps_process_and_cleans_owned_files` | reap 窗口与失败保留 run 目录；`test_cleanup_waits_for_generate_release_before_deleting_run_dir`、`test_cleanup_reap_failure_is_visible_and_preserves_run_dir` |
| Darwin 资源 | suffix 子进程 Foundation/confstr 三根路径切换，私有 Temp 可放 MPSGraph；`test_darwin_worker_environment_owns_private_cache_lifecycle` | 同一 process 目录归属；producer error 测试锁 payload | timeout 仅清本 worker 目录；`test_failed_task_slot_recovers_for_next_request[timeout]` | crash 后 slot 可补齐；`test_failed_task_slot_recovers_for_next_request[crash]` | cancel 后 slot 可补齐；`test_failed_task_slot_recovers_for_next_request[cancel]` | shutdown 回收本池 run 目录；`test_cleanup_reaps_three_workers_in_parallel_and_removes_run_dir`、`test_cleanup_reap_failure_is_visible_and_preserves_run_dir` |
| 结果边界 | FunASR/Qwen3 pickle `os.replace`；`test_funasr_worker_publishes_result_with_replace`、`test_success_result_is_published_with_replace` | FunASR JSON `.result` `os.replace`，发布前最终路径不存在；`test_funasr_worker_publishes_json_error_atomically` | 父进程只读已发布文件；`test_timeout_reaps_process_before_cleaning_owned_files` | 退出后复查 result；`test_exit_after_publish_rechecks_result_before_reporting_crash` | 取消后不接受迟到结果；`test_cancel_reaps_process_and_cleans_owned_files` | cleanup 关闭接纳；`test_generate_does_not_revive_pool_after_explicit_cleanup`、`test_server_stop_closes_websocket_before_pool_cleanup` |

并发状态轴覆盖：三并发 A/B/C 占 slot 后 B 先完成、D 取空闲 slot（`test_out_of_order_completion_assigns_new_task_to_free_slot`）；多个 dead slot 并行补齐（`test_missing_workers_are_replenished_in_parallel`）；已有 ready slot 不等其他 dead slot（`test_ready_worker_is_not_blocked_by_missing_slot`）；health 不抢 busy（`test_health_replenishes_dead_slot_without_touching_busy_worker`）；正常结果后台补 worker 且不阻塞交付（`test_success_delivery_does_not_wait_for_replenish_ready`）；cleanup 与 release 同步且 reap 失败可重试（`test_cleanup_waits_for_generate_release_before_deleting_run_dir`、`test_cleanup_reap_failure_is_visible_and_preserves_run_dir`）；启动失败清理 run 目录（`test_spawn_failure_cleans_run_directory`）。

## 待验证前提

1. [实测] 生产 Mac venv 中字符串形式的三个 `os.confstr` 名称均为 `ValueError: unrecognized configuration name`；已采用数值常量 65536/65537/65538。此前真实 venv 子进程已验证 suffix 后 Foundation/confstr 三根路径切换，私有 Temp 有 `com.apple.MetalPerformanceShadersGraph`。
2. [推断] 真实 FunASR 与 Qwen3 worker 都能在对应 task 目录使用同一 atomic publish helper，且现有 pickle payload 字节语义保持不变；用 producer fixture 捕获实际写入 bytes 验证。
3. [推断] `Popen` 的 PID、slot 与任务 UUID 足以拒绝迟到结果；用 hung subprocess fixture 先终止旧 PID，再注入同名旧结果验证拒绝。
4. [实测] 生产曾有 78 分钟任务首轮 `Ran out of input`，随后既有重试成功（2038 段/5 speaker，215.35s），证明父进程确实可能读到半写 pickle；长音频修复验证仍由 Mac 实际 venv/WS 卡执行，不调整阈值。本卡不把 TaskManager 看门狗的 `TIMED_OUT` 状态解释为立即中断 MPS。

## 验收路径

1. 入口：Mac 生产 WebSocket 上传真实长音频，`force_refresh=true`；同时读取 health/capabilities，记录当前 worker PID/PPID。
2. 步骤：先跑定向 5 组池/worker 单测并重复 5 次，再跑 unit 全量；Mac 后续卡设置 `FUNASR_RUN_INTEGRATION=1` 用实际 venv 跑相关 integration；最后以生产 PM2 实际 Python 做 ≥12 分钟 force_refresh，对照调用终态、task_id、结果 bytes、worker PID 和私有目录。
3. 预期：并发乱序无串台；success/error 只读原子完整 JSON 或 pickle；pool timeout/crash/cancel/shutdown 都有唯一调用终态且不重置 deadline；旧进程确认退出后才清自有目录；共享缓存中的预置哨兵仍在且不被扫描清理；health 不抢 busy，shutdown 后不复活池。
