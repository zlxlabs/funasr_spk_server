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

- **要点**：A 在 `src/core/funasr_transcriber.py` 由 JSON/SRT 共用私有结构校验；非空原文配缺失/空 `sentence_info`、未知顶层结构显式失败，异常不返回成功 payload，因而不会到达新鲜结果的成功缓存写入；仍沿用任务管理器现有错误分类与重试策略，不承诺立即终态。明确空原文且无句子、或显式空 `sentence_info`（原文缺失/为空）保留为空成功；这只表示模型空输出，不推断音频是否静音。`text` 与 `sentence_info` 同时缺失才是未知结构。判形状与空结果诊断只写 task id 与 `empty_result`/`empty_text`/`invalid_shape` 等客观状态，不写原文、文件名或 URL。
- **要点**：B 独立验证 WebSocket 真断开后，客户端用现有 `task_status_batch` 查询同一 task id 并读到终态；仅增加安全的连接诊断字段以定位异常，不延长超时。
- **已否决**：加长心跳/任务超时不能让已完成结果重新送达；新增持久化或恢复协议会重复现有 batch 查询能力；自动重试或格式猜测会掩盖解析错误。

## 关键不变式

1. [实测现状] `transcribe` 的 JSON 与新鲜 SRT 分别调用 `_parse_and_merge_segments`、`_generate_srt_from_raw_result`；目前两处各自接受缺 `sentence_info` 并输出空。`tests/unit/test_funasr_parse_sentence_level.py` 既有用例已锁定显式空 `sentence_info` 可输出空；本卡新增 JSON/SRT 坏形状与合法空输出用例锁定共同校验。
2. [实测现状] `tests/fixtures/audio/silence_5s.wav` 和 `tests/fixtures/golden/silence_5s.golden.json` 存在，golden 只记录空 segments/speakers，不含原始模型结构。[待验证] Mac parity 实测合法空输出 raw shape 后再锁定可接受形态。
3. [实测现状] `task_status_batch` 已存在于 `src/api/websocket_handler.py`，单测在 `tests/unit/test_websocket_task_status_batch.py`；B 新增真实断连、重连、原 task 取终态的入口验证。
4. [待新增测试] 非法结果使 `transcribe` 抛错而不返回缓存 payload；现有 `src/core/task_manager.py` 仅在成功获得返回值后调用缓存写入，错误仍沿既有错误分类/重试策略处理，不保证立即终态。判形状日志只含 task id 与结构状态，不含识别正文或音频路径。

## 待验证前提

1. [推断] Mac 生产同类环境对 `silence_5s.wav` 返回的原始结构能区分合法空输出与损坏；在 Mac 隔离候选目录运行 parity 前先报命令和副作用边界并获主脑确认。
2. [推断] 现有 batch 查询在真实客户端重连后仍可凭原 task id 返回终态；由 B 的 WebSocket 入口测试验证。

## 验收路径

1. 入口：本地 FunASR parser 单测；Mac 隔离环境 `FUNASR_RUN_INTEGRATION=1` parity；B 的 WebSocket 集成入口。
2. 步骤：验证非空 text + 缺失/空句子结构失败且 JSON/SRT 一致；验证 Mac silence 仍空成功；断开正在运行任务的连接，再连入并 `task_status_batch` 查询原 id。
3. 预期：损坏结果失败且无成功缓存；合法模型空输出为空完成；重连后读到原任务真实终态。只报告本地实际可运行的验证，不宣称 CI（仓库无 workflow）。
