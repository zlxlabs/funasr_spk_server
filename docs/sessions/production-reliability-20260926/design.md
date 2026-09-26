# DESIGN-note：转写结果可判错、断线可取回

## 目标

用户得到可信的转写完成状态：损坏的 FunASR 结果明确失败；WebSocket 断开后可重新查询原任务并取回终态结果。

## 非目标

不升级模型或依赖，不调整心跳/任务超时，不增加重试、fallback、新查询协议或数据库 schema；不部署或重启生产；不改写既有缓存。

## 为什么不是分区 / 删除 / 约定

- **分区**：JSON 与 SRT 是同一模型结果的两个出口；各自判形状会再次产生语义差异。断线后的任务也仍由同一任务标识查询。
- **删除**：不能删除 JSON/SRT 任一出口，也不能让客户端断线时放弃已提交任务；现有批量状态查询应继续可用。
- **约定**：约定调用方检查结果或自行重连无法阻止服务端把损坏结果记为成功；边界必须由解析和状态查询路径执行并测试。

## 方案要点与已否决方案

- **要点**：A 在 `src/core/funasr_transcriber.py` 由 JSON/SRT 共用私有结构校验；非空原文配缺失/空 `sentence_info`、未知顶层结构显式失败，异常不返回成功 payload，因而不会到达新鲜结果的成功缓存写入；仍沿用任务管理器现有错误分类与重试策略，不承诺立即终态。明确空 `text` 可缺少 `sentence_info` 或带空数组；`text` 缺失时仅显式 `sentence_info: []` 保留为空成功。`text` 与 `sentence_info` 同时缺失属于未知结构并失败。这些只表示模型空输出，不推断音频是否静音。判形状与空结果诊断只写 task id 与 `empty_result`/`empty_text`/`invalid_shape` 等客观状态，不写原文、文件名或 URL。
- **要点**：B 独立验证 WebSocket 真断开后，客户端用现有 `task_status_batch` 查询同一 task id 并读到终态；仅增加安全的连接诊断字段以定位异常，不延长超时。
- **已否决**：加长心跳/任务超时不能让已完成结果重新送达；新增持久化或恢复协议会重复现有 batch 查询能力；自动重试或格式猜测会掩盖解析错误。

## 关键不变式

1. [已实现并测试] `transcribe` 的 JSON 与新鲜 SRT 共享 `_validated_funasr_sentences`。基线红验中两个实际 `transcribe` 出口都未抛错；H0 定向测试 15 passed，覆盖非空 text 配缺失/空句子时抛错，以及合法空输出。新鲜解析失败不会返回成功 payload，因而不进入成功缓存写入；错误沿现有任务错误分类/重试处理，不保证立即终态。
2. [Mac 隔离实测] `silence_5s.wav` 的 raw 顶层为 list，首项是 dict，`text` 存在且为空，`sentence_info` 缺失；三条既有 FunASR golden parity 通过且前后哈希未变。该结构被作为模型合法空输出保留，不据此断言生产中两条空任务一定是静音。
3. [已实现并测试] `task_status_batch` 原已存在于 `src/api/websocket_handler.py`。B 的真实断连重连入口测试在 Mac 候选通过；连接诊断不暴露识别正文、文件名或 URL。组合全 integration 中的 Qwen 失败另按基线归因，不影响 FunASR 结果契约验证。
4. [证据边界] FunASR 实测按现有 fixture 使用 lock 模式，未验证生产多 worker 池并发。历史缓存命中不经过本次新鲜解析校验；只读抽样中 540 条非空 text 缓存未发现缺/空 `sentence_info`，另有 5 条空 segments 缓存的 text 均明确为空。历史 cache SRT 路径留作 P2，不改数据库或既有缓存。

## 验收路径

1. 入口：本地 FunASR parser 单测；Mac 隔离候选中的 `FUNASR_RUN_INTEGRATION=1` parity 与真实 silence raw-shape probe；B 的 WebSocket 集成入口。
2. 步骤：验证非空 text + 缺失/空句子结构失败且 JSON/SRT 一致；验证 Mac silence 仍空成功；断开正在运行任务的连接，再连入并 `task_status_batch` 查询原 id。
3. 预期：损坏结果失败且无新鲜成功缓存；合法模型空输出为空完成；重连后读到原任务真实终态。仓库没有 CI workflow，报告只列实际本地与隔离候选验证，不宣称 CI 通过。
