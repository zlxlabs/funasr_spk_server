# R1 审查结论

审查对象固定为 `43e8e949d8a860cdc9ab0b58e50a20f279d90861..a24d79332761f92c330d388123aef85390e67234`，规格为 H0 中的 `design.md`。结论：**FAIL**（P1=1，P2=1，P3=0）。P1 未修复前不应合并；本文件只记录审查，不修改实现。

## 读取范围与证据

- 已整读固定设计规格、`file_based_process_pool.py`、`atomic_result_publish.py`、FunASR/Qwen3 worker、两个 transcriber wrapper、`main.py`，以及实际调用方 `task_manager.py`、`websocket_handler.py`。
- 已审改动测试与真实 producer fixture：`test_file_based_pool_lifecycle.py`、`test_server_shutdown.py`、`test_funasr_hotword_propagation.py`、`test_qwen3_worker_process.py`、Qwen3 pool 测试/集成测试。
- H0 临时隔离树定向测试：42 passed；Qwen3 并发派发集成测试：4 passed。完整 `tests/unit`：1101 passed、6 skipped；1 failed + 10 errors 均为 Linux 加载 macOS/动态库专属 Qwen3 vendor 测试，未涉及本次池路径。
- base 临时树红验 `test_funasr_worker_publishes_result_with_replace` 按预期失败（旧实现 `os.replace` 调用 0 次），证明新增 producer 断言确实锁定了本次改动。
- OCR 前置扫描返回 `status=reviewed`、`coverage=complete`；其复核器部分读取了基线工作树，未未经 H0 核验采纳其意见。

## 通过项

- lease 在管理锁内绑定空闲存活 slot；busy slot 不被 health 抢占，三并发乱序和空闲 slot 复用测试通过。
- task、pickle、JSON result 和 ready marker 均由 producer 原子发布；父进程只读取最终路径，并校验 task_id 与 worker PID。正常结果先交付，补 worker 在后台进行。
- timeout/cancel/crash 路径先 reap 原 PID，再清理本 lease 的 task/audio/result/converted wav；cleanup 等待 reap，失败保留 run 目录并对外抛错。
- Darwin 数值 confstr suffix 只注入 FunASR worker；Qwen3 继续保留既有支持格式跳过 ffmpeg 的路径，且池目录物理隔离。

## Findings

### P1 — 停机关闭 WebSocket 后仍可接纳新任务

- **固定 SHA 位置**：`src/main.py:112-117`；调用方 `src/core/task_manager.py:228-260`。
- **触发序列**：PM2 发送 SIGTERM，`FunASRServer.stop()` 调用 `self.server.close()` 后立即 `await task_manager.stop()`。当前 websockets 12.0 的 `Server.close()` 先调度异步 close task，既有连接在 close task 完成前仍可处理消息；`task_manager.stop()` 已将 `is_running=False`、取消所有队列 worker，但 `submit_task()` 完全不检查 `is_running`。活动连接此时完成一次 upload 后仍能成功 `put_nowait(task_id)`。
- **证据**：在 H0 临时树直接令 `TaskManager.is_running=False` 后调用 `submit_task()`，实际输出 `queue_size=1 status=pending`；本地 websockets 12.0 源码显示 `close()` 只创建 `_close` task，关闭既有连接需后续 await。
- **违反不变式**：规格“关闭准入后停 TaskManager 再回收池”；停机不得接纳新任务且任务应有可判定终态。
- **业务影响**：该任务进入已无 worker 的队列，永久停在 PENDING，上传文件也因仍被 live task 引用而不能被孤儿清理器回收；客户端收到上传成功却永远拿不到结果，属于内网真实活动 WebSocket 停机竞态下的静默失败。
- **最小修复方向**：复用现有 `TaskManager.is_running`，在 `submit_task` 的队列锁内二次检查并在拒绝时回滚 task、清理刚落地文件；同步覆盖 cache-hit/队列 put 的竞态。仅把 `server.close()` 调用提前或等待关闭不足以代替准入 guard。

### P2 — pool cleanup 失败后 FunASR wrapper 仍伪报已初始化

- **固定 SHA 位置**：`src/core/funasr_transcriber.py:131-136`；池失败传播在 `src/core/file_based_process_pool.py:786-805`。
- **触发序列**：已初始化池执行 `FunASRTranscriber.cleanup()`；池的 reap 或 owned-path 删除失败，`_cleanup_owned_resources()` 已将池置为关闭并保留 run 目录后抛错，但 wrapper 在 `await self.model_pool.cleanup()` 之后才设置 `self.is_initialized=False`，因此异常时仍为 True。
- **证据**：H0 失败保留测试锁定 reap 异常；同样条件的直接探针输出 `is_initialized_after_failed_cleanup=True`。
- **违反不变式**：cleanup 失败应可见且保留仍可能使用的资源，同时 wrapper 的生命周期状态必须与池状态一致；否则不能依靠显式 `initialize()` 重新进入合法状态。
- **业务影响**：后续 `transcribe()` 跳过 wrapper 初始化，直接命中已关闭池；显式 `initialize()` 又被 wrapper 的早返回挡住。资源失败本身仍会抛出且不是已知正常 Mac 路径，故降为 P2。
- **最小修复方向**：在 cleanup 的 `finally` 中清除 wrapper `is_initialized`，并补 wrapper 层 cleanup 失败后状态/重试测试；保留池自身 fail-closed 的“未完成 cleanup 不允许 initialize”判定。

## 结论

原子结果完整性、真实 producer payload、三 worker lease/reap 及 Darwin 隔离满足规格；但 P1 停机准入竞态直接破坏停机语义，当前 verdict 为 **FAIL**。本轮没有 Mac FunASR 模型集成测试，也没有把目录元数据耗时外推为长音频根因。
