# DESIGN-note：全仓审查缺陷修复（第三批）

来源：2026-10-20 `/code-review max src/ scripts/`（基线 7b8bfaa）「超出 15 条上限」中验证成立的中低优先级问题；前两批见 `docs/sessions/261020-review-fixes/design.md`、`docs/sessions/261020-review-fixes-b2/design.md`。基线 e8736b6。

## 目标

修完后：启动失败时进程以非零退出、不再推送「正常关闭」；`max_connections` 真正限制并发连接数；环境变量写错时启动即报错；
Qwen3 分段后处理不再产出跨几分钟的巨段、乱序段、超长切片、假说话人，也不丢零时长的词；
Mac 上 m4a 等转码格式的缓存能正常命中、回给客户端的 hash 正确；worker 预热失败不再假装就绪；主进程崩溃后不留孤儿 worker；
显存预检不把 sidecar 自己的占用算进去；language 带空白不会污染缓存；告警脚本状态写入原子化、默认配置下指标规则不再静默失效；
安装 git hooks 不再覆盖全局 hooksPath；任务完成通知不再占用 worker 几十秒；GBK 元数据的音频能做静音检测。

## 非目标

- `task.started_at` 早于拿到池资源（task_manager.py:670）：第二批已定另议，本批不动。
- 审查只到「可能成立」的条目（asr.py token 预算、file_utils 'wb' 原地重写、回收 worker 丢结果、FunASR 空列表重试、Mac 词级对齐超时公式）与低级别条目（SRT 毫秒截断、缓存过期时区、/metrics 不存在的键、死代码）。
- `src/api/websocket_handler.py:884` 分片 finalize 查缓存漏传 output_format：在主干 e8736b6 上已按 `session["output_format"]` 查询，不再成立。
- `_upload_engine_error` 与 create_task 的 engine 判定重复：不在本批抽取。
- 不新增 config 字段；不清理历史缓存。

## 为什么不是分区 / 删除 / 约定

- **分区**：按文件分 6 张互不重叠的卡，全部并行，无产物依赖。
- **删除**：`install-git-hooks.sh` 写本仓 `core.hooksPath` 的做法在已设全局 hooksPath 的机器上只会把全局闸停掉；改为检测到全局 hooksPath 时不覆盖。
- **约定**：每张卡第一步在最新主干上写测试复现缺陷；复现不出的条目在报告中写明，不改代码。

## 方案要点

| 卡 | 文件 | 条目 |
|---|---|---|
| S 启动与配置 | `src/main.py`、`src/core/config.py` | 启动失败退出码非零且不发「正常关闭」；`max_connections` 不再误传给 `max_queue`，改为真实计数限制，超限以 1013 关闭并打 WS-CONN-LIMIT；环境变量转换失败启动即抛错，不再只记日志 |
| P 分段后处理 | `src/core/qwen3/postprocess.py`、`merge.py`、`segment_split.py`、`diarize_ort.py` | postprocess:86 巨段、merge:457 乱序、segment_split:55 超 max_dur、diarize_ort:523 num_speakers≤0 假说话人、merge:69 丢零时长词、merge:62 词挂段 O(词×段)、无 chunks 时段时长为 0 |
| W 进程池 | `src/core/qwen3_worker_process.py`、`src/core/file_based_process_pool.py` | worker:115 用转码后 wav 算 file_hash、worker:177 预热失败仍发 ready、pool:631 孤立代理字符杀热 worker 且误报、worker 不检测父进程死亡、pool:409 run-* 目录崩溃后泄漏、pool:499 等租约每 10ms 两条 INFO |
| Q Qwen3 转写器 | `src/core/qwen3_transcriber.py`、`src/core/qwen3_inproc_pool.py` | 显存预检计入 sidecar 自身占用、language 带空白污染缓存、inproc_pool:146 工厂漏传 sidecar 与预检参数 |
| A 运维脚本 | `scripts/metrics_alert.py`、`scripts/install-git-hooks.sh` | 状态文件原子写入、探测地址写死、默认配置下 /metrics 403 致指标规则静默失效、hooks 安装覆盖全局 hooksPath |
| N 通知与静音检测 | `src/utils/notification.py`、`src/core/task_manager.py`（仅通知调用点）、`src/utils/silence_detect.py` | webhook 重试在 worker 内同步执行约 33 秒、ffmpeg 输出严格 UTF-8 解码遇 GBK 失败 |

已否决：为 /metrics 403 直接关闭鉴权（扩大暴露面）；为 hooks 冲突改写全局 hook 文件（跨仓写入）。

## 关键不变式

1. [实测] 启动失败进程退出码非零 —— S 测试锁死。
2. [实测] 第 max_connections+1 个连接被拒 —— S 测试锁死。
3. [实测] 后处理输出按 start 有序、每段不超过 max_dur、零时长词保留 —— P 测试锁死。
4. [实测] 转码路径的 file_hash 等于原始上传文件的 hash —— W 测试锁死。
5. [实测] 预热失败的 worker 不发 ready —— W 测试锁死。
6. [实测] language 带空白与规范写法产生同一缓存 tag 且对齐行为一致 —— Q 测试锁死。
7. [实测] 告警脚本在 /metrics 拒绝访问时非零退出或发出告警，不静默跳过 —— A 测试锁死。
8. [实测] 任务完成通知不阻塞 worker 推进下一个任务 —— N 测试锁死。

## 待验证前提

1. [推断] websockets 当前版本 `serve` 无内建最大连接数参数 —— S 执行器核实。
2. [推断] /metrics 403 来自鉴权或来源地址限制 —— A 执行器核实原因再定修法。
3. [推断] 全局 hooksPath 下的 pre-push 是否支持委托到仓内 hook —— A 执行器只读核实，不改全局文件。

## 验收路径

1. 入口：`bash scripts/run_checks.sh`（unit 全量）+ 各卡新增测试；修复前代码上新测试必须失败。
2. N 卡触及 `task_manager.py`（FunASR 路径），P / W / Q 触及 Qwen3 输出，合并后在 Mac 补跑 `bash scripts/run_checks.sh --parity`。
3. 预期：unit 全绿（Linux 上加载 macOS 专用库导致的 11 条存量失败除外）。
