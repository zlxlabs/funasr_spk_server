# DESIGN-note：全仓审查缺陷修复（第二批）

来源：2026-10-20 `/code-review max src/ scripts/`（基线 7b8bfaa）与同日 PR #17 文档审查；第一批见 `docs/sessions/261020-review-fixes/design.md`（PR #24）。基线 5e1abc6。

## 目标

修完后：内容相同的文件被多个任务共用时，一个任务被拒不会删掉另一个任务的音频；命中缓存的上传不再留下永不结束的任务；
分片上传在传数据前就校验引擎；已取消 / 已超时的任务不会被改成完成或被重跑；长音频说话人聚类合并不再卡住所有连接；
读不出时长的音频不再被误判「太短」永久失败；不读消息的客户端不会卡死 worker；恶意分片不能造出超大文件冻结服务；
慢速大文件上传不会因开始时间超 30 分钟被清掉；客户端文档与服务端行为一致。

## 非目标

- `queue_full` 消息类型不改：客户端指南已把它定为带 `error` 兼容字段的独立类型（指南 377/382 行），审查该条判为设计内。
- `task.started_at` 早于拿到池资源导致看门狗误判超时（task_manager.py:670）：本批只保证终态不被覆盖，计时口径另议。
- 不清理已写进缓存的历史错误结果；不改 config 字段。

## 为什么不是分区 / 删除 / 约定

- **分区**：按文件分区。`websocket_handler.py` 一个文件承载 6 个缺陷，拆两张卡串行（W1 → W2），其余 task_manager / qwen3_transcriber / file_utils 各一张并行；文档卡依赖 W1 行为，排在 W1 之后。
- **删除**：单文件上传「先 create_task 再查缓存」——把缓存查询提前到 create_task 之前，命中时根本不建任务，泄漏源头消失；不需要额外的清理机制。
- **约定**：终态不可逆是状态机约定：所有写状态的点先判「已是终态就不动」，不新增锁或状态。

## 方案要点与已否决方案

- **W1 上传生命周期**（`src/api/websocket_handler.py`）：
  1. 单文件与分片的缓存查询都提前到 create_task 之前，命中直接回 task_complete、不建任务；
  2. 分片上传在 `upload_request` 阶段就按与单文件相同的规则校验 engine（不一致立即 `upload_error`，不建 session）；
  3. queue_full / finalize 失败 / sweep 等删除上传文件的路径，删除前必须确认没有其他 live 任务（PENDING/PROCESSING）使用同一 file_hash 的文件——复用 `task_manager._maybe_delete_task_file`（被拒任务已从 tasks 回滚，不会把自己算进 live），不修改 task_manager.py。
  已否决：给 PENDING 加 TTL 清理（治标，泄漏源头仍在）。
- **T 终态不粘**（`src/core/task_manager.py`）：完成写 COMPLETED、可重试失败回 PENDING 前，若任务已是 CANCELLED / TIMED_OUT 则保持终态不动、不重跑、不重复计数；`cancel_task` 守卫补上 TIMED_OUT。
- **Q 聚类合并阻塞事件循环**（`src/core/qwen3_transcriber.py`）：`apply_cluster_centroid_merge_to_turns` 改走 `run_in_executor`，与相邻 ASR / 对齐 / 静音检测一致。
- **F 时长探测**（`src/utils/file_utils.py`、`src/core/funasr_transcriber.py`）：`get_audio_duration` 不再吞异常返回 0.0；format 无 duration 时用音频流 duration，仍无则用 ffmpeg 解码统计时长；探测本身失败抛出明确错误，调用方不再把它报成「音频时长过短」。
- **W2 连接健壮性**（`src/api/websocket_handler.py`，依赖 W1 合并）：`_send_message` 加发送超时，超时视为该连接失效并清理关联，不阻塞 worker；`upload_chunk` 校验 chunk_index 范围与分片长度，越界拒收且不计入 received；finalize 的 MD5 改为在 executor 中流式计算，去掉整文件读入内存；sweep 按最近活动时间而不是创建时间判定过期，正在 finalize 的会话不清扫。
- **D 文档**（客户端指南、README、docs/README、服务端协议文档、AGENTS.md 引擎优先级段，依赖 W1 合并）：修正 PR #17 审查列出的 7 处文档问题，以 W1 后的真实行为为准。

## 关键不变式

1. [实测] 被拒 / 失败 / 过期的上传不会删除仍被 live 任务使用的同 hash 文件 —— W1 测试锁死。
2. [实测] 命中缓存的 upload_request 不在 task_manager.tasks 中留下任务 —— W1 测试锁死。
3. [实测] engine 不一致的分片 upload_request 在传任何分片前被拒 —— W1 测试锁死。
4. [实测] CANCELLED / TIMED_OUT 一经写入不再变化 —— T 测试锁死。
5. [实测] cluster_merge 不在事件循环线程执行 —— Q 测试锁死。
6. [实测] 时长探测失败不再表现为「音频时长过短」 —— F 测试锁死。
7. [实测] 对端不读消息时 worker 的发送在超时内返回 —— W2 测试锁死。

## 待验证前提

1. [推断] `_maybe_delete_task_file` 对已从 tasks 回滚的任务对象调用时 live 判定正确 —— W1 执行器核实。
2. [推断] 当前 websockets 版本的 send 超时取消不会破坏连接内部状态 —— W2 执行器核实。

## 验收路径

1. 入口：`bash scripts/run_checks.sh`（unit 全量）+ 各卡新增测试；修复前代码上新测试必须失败。
2. W1 / T / F 触及 FunASR 路径（websocket_handler / task_manager / funasr_transcriber），合并后在 Mac 补跑 `bash scripts/run_checks.sh --parity`。
3. 预期：unit 全绿（Linux 上加载 macOS 专用库导致的 11 条存量失败除外）。
