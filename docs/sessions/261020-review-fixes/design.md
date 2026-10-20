# DESIGN-note：全仓审查 P1 与静默出错缺陷修复（第一批）

来源：2026-10-20 `/code-review max src/ scripts/`，基线 7b8bfaa。

## 目标

修完后：队列满载时服务不再因重试卡死；Qwen3 说话人分离出错不再让模型被两个线程同时使用；
告警脚本发送失败会大声报错并在下轮重发；Qwen3 不再静默丢转写文本或丢右声道说话人；
FunASR 热词不再静默失效、不再对客户端给的字符串发 HTTP 请求。

## 非目标

- 审查报告中其余问题（共享上传文件误删、终态不粘、PENDING 泄漏、send 无超时、事件循环阻塞、
  分片越界、sweep 误杀、queue_full 消息类型等）留下一批，本批不碰 `src/api/websocket_handler.py`。
- 不清理已经写进缓存的错误结果（空 segments 等）；如需清理另开卡。
- 不改协议、不改 config 字段。

## 为什么不是分区 / 删除 / 约定

- **分区**：五个缺陷分属五组互不重叠的文件，按文件分区拆成 5 张并行卡，天然无写冲突。
- **删除**：重试入队那一步可以删掉——worker 已持有任务，直接在本 worker 内重跑，不需要再进有界队列；
  热词「字符串拼接」那层可以删掉，改为把词表写成 FunASR 原生支持的 .txt 文件。
- **约定**：告警「先发后记」、并发线程「先收齐再抛错」都是顺序约定，不新增运行时机制。

## 方案要点与已否决方案

- **卡 1 队列死锁**（`src/core/task_manager.py`）：可重试失败不再 `await task_queue.put`，
  改为在当前 worker 内重跑（保留 retry_count 上限与 funasr 模型重置逻辑）。
  已否决：无界队列（破坏容量闸语义）；put_nowait 失败即 FAILED（高负载下误杀合法重试）。
- **卡 2 孤儿 ASR 线程**（`src/core/qwen3_transcriber.py`）：并行 ASR+diarize 任一失败时，
  必须等另一路线程跑完再把异常抛出，保证 engine 归还池时没有线程在用它。
  已否决：取消 future（executor 线程无法真正取消）。
- **卡 3 告警脚本**（`scripts/metrics_alert.py`、`src/utils/notification.py`）：先发送、成功的才写入
  已告警状态；发送失败或通知未启用时退出码非 0；新增 `--selftest`。
  `send_custom_notification` 返回成功与否（新增消费者：metrics_alert；main.py 现有调用忽略返回值不受影响）。
- **卡 4 Qwen3 静默丢内容**（`src/core/qwen3/merge.py`、`src/core/qwen3/audio_io.py`）：
  diarize 无 turn 但 ASR 有文本时，全部文本归同一个说话人返回，并打 WARNING；
  soundfile 读多声道时取各声道平均混为单声道，与 librosa mono=True 路径一致。
  已否决：diarize 无 turn 时把任务判失败（ASR 结果是对的，丢弃有效转写对用户更差）。
- **卡 5 FunASR 热词**（`src/core/funasr_transcriber.py`、`src/core/result_projection.py`）：
  terms 写入临时 .txt（每行一个词）并把路径作为 hotword 传入，任务结束删除；
  `context_applied` 只在热词真正交给模型时为 true。
  已否决：对 terms 做 http/.txt 字符串过滤（打补丁、仍会拆开含空格的术语）。

## 关键不变式

1. [实测] 可重试失败路径不再对有界 task_queue 做阻塞 put —— `task_manager.py` 重试分支；卡 1 新增并发满队列测试锁死。
2. [实测] engine 归还池时没有任何 executor 线程仍在使用它 —— `qwen3_transcriber.py` 并行分支；卡 2 测试锁死。
3. [实测] 告警状态只记录已成功送达的告警 —— `metrics_alert.py`；卡 3 测试锁死。
4. [实测] ASR 有文本时 segments 不为空 —— `merge.py` 两个 merge 函数；卡 4 测试锁死。
5. [实测] FunASR 收到的 hotword 是存在的 .txt 路径，每个术语原样一行 —— `funasr_transcriber.py`；卡 5 测试锁死。

## 待验证前提

1. [推断] funasr 1.2.7 的 .txt 热词文件按行读取、保留行内空格 —— 卡 5 执行器对照 venv 内 wheel 源码核实。
2. [推断] Qwen3 ASR 侧对多声道是做平均混音 —— 卡 4 执行器核实 ASR 读音频路径。

## 验收路径

1. 入口：`bash scripts/run_checks.sh`（unit 全量）+ 每卡新增测试。
2. FunASR 路径改动（卡 1、卡 5）在 Mac 上补跑 `bash scripts/run_checks.sh --parity`。
3. 预期：unit 全绿；各卡新增测试在回退修复时变红。
