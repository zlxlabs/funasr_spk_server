## 现场核查

- 当前阶段：verifying；工作树基线为 `43e8e94`，仅本卡三个产物允许写入。
- 本段结论：生产健康端点 HTTP 200/healthy；端点未暴露 `active/queued`，已在证据中保留 null，不伪造为零。
- 关键决策：按远端实际路径使用 `temp/samples/4-person-example.m4a`，不改生产文件或共享缓存。
- 下一步唯一动作：运行同一 60 秒 16kHz 单声道音频的默认目录 A→suffix B。

## 实测

- 当前阶段：verifying；真实生产 `src/core/worker_process.py` 两例均成功退出，均返回 1 段结果。
- 本段结论：A 总耗时 13.274s，B 总耗时 24.136s；B 未显示提速，ready 分别为 8.786s/8.480s。
- 关键决策：不反序、不扩大音频或次数；结果仅解释 60 秒样本，不能外推原始长任务。
- 下一步唯一动作：确认 B 私有 MPSGraph 条目的 PID、回收状态与无残留进程。

## 收尾

- 当前阶段：verifying 已收尾；B 私有目录前 10 条中 9 条 `mpsgraph-2016-...` 与 worker PID 2016 匹配。
- 本段结论：三根私有 Darwin 目录均已删除，worker 子进程均已退出，生产健康端点仍为 200/healthy。
- 关键决策：原始 worker stdout/stderr 只留远端自有临时根，证据仅保留结构化白名单字段。
- 下一步唯一动作：向主脑回传两个 commit 与结构化证据摘要。
