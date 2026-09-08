# R1 主脑分诊 [codex]

审查对象：`43e8e949d8a860cdc9ab0b58e50a20f279d90861..a24d79332761f92c330d388123aef85390e67234`。独立报告保留原始 FAIL，不覆写审查者意见；以下是结合真实部署与既有契约后的处置。

| 意见 | 工具标注 | 本仓判定 | 真实触发与后果 | 处置 |
|---|---|---|---|---|
| 停机期间已建立连接仍可能提交任务 | P1 | P2，存量边界 | 真实 Mac 的 websockets=12.0，PM2 kill_timeout=10000ms，异步关闭短窗口具有可能性；报告只测了手工停止状态的 submit，未复现完整 WebSocket 停机。任务与队列仅在内存，PM2 退出后旧 PENDING 不会恢复；“永久 PENDING / 永久 live 引用”不成立。 | 接受不修，保留为上传准入改进项；不把本批池回收扩成任务持久化或停机上传协议改造。 |
| FunASR wrapper cleanup 抛错后初始化标记仍为 True | P2 | P2 | 池 reap/删除失败时可触发；异常明确传播、资源保留，PM2 停机最终退出。不是正常长音频路径的静默错误。 | 接受不修；若以后支持同进程 cleanup 失败后的服务重用，再整理 wrapper 重试契约。 |

## 新增核对证据

- `src/core/task_manager.py:118` 创建内存 `asyncio.Queue`，`:196-218` 把任务写进内存 dict；数据库仅缓存已完成转录，没有 PENDING 恢复表。
- 既有 `docs/开发/2026-06-16-任务系统设计与异步规划-新session交接.md:43-45,69` 明示状态易失、重启丢失在途/排队任务；`docs/开发/2026-06-16-异步轮询契约-设计定案与落地计划.md:137` 同样写明“重启（队列/在途全丢）”。不能把既有且明确的重启语义算成本批新引入的永久泄漏。
- base `src/main.py:109-116` 已在关闭 WebSocket 前停止 TaskManager；H0 仅提前发出 close 并新增池回收，没有新引入这个上传窗口。
- 本批 design 的准入不变式落在 `pool.generate/cleanup`：cleanup 后不复活池、旧 PID reap 后才清资源。它没有声明重启保留任务或用户 API cancel 立即终止子进程。
- 真实 Mac 消费环境核验：生产 PM2 cwd 为 `/Users/zhanglixing/Production/funasr_spk_server`，入口 `run_server.py`，解释器为该仓 `venv/bin/python`，websockets 12.0，kill_timeout 10000ms。

本轮确认新增 P1 为 0。后续仍需另一角度的独立审查以及 Mac 候选版本集成、真实长音频和资源清理证据，不能只凭本分诊宣布交付完成。
