# A-r1 独立审查结论

- 冻结范围：`ec0b892035ff1def7e1168254b5b89622c5e7bae..4e49d35070b49f678ea9b4692d7a39c37c49da40`
- 风险等级：internal（任务卡指定；仓级未另行声明）
- 判定：无有效 finding
- Finding 分级：P1 0、P2 0、P3 0；未发现违反本卡不变式的问题。
failure-visibility: clean

## 核心核验

1. JSON 与 SRT 都调用 `_validated_funasr_sentences`：SRT 在 `funasr_transcriber.py:319` 校验，JSON 在 `:346` 校验；SRT 返回缓存载荷前还会解析 segments（`:342`）。非空 `text` 配缺失或空 `sentence_info` 在 `:407-416` 抛 `ValueError`；空结果列表、非对象顶层、首项非对象也明确失败（`:391-405`）。
2. 兼容空形态保留：无 `text` 的显式空 `sentence_info`、空 `text` 且缺少句子，以及空 `text` 配显式空列表均返回空句子（`:407-421`）。未见时间戳猜测或历史缓存改动。
3. 结构错误和合法空结果日志只带 task id 与状态（`:387-389`, `:418-420`）；未输出原文、文件名或 URL。单测检查 task id 可见且识别文字与音频名不在日志参数中（`test_funasr_parse_sentence_level.py:123-166`）。
4. 测试覆盖了 helper 的正负形态与两种格式（`test_funasr_parse_sentence_level.py:95-121`），并通过实际 `transcribe()` 对 JSON、SRT 检查非空原文缺句时抛错及日志安全（`:123-166`）。任务管理器先等待 `transcribe()` 返回再写缓存（`task_manager.py:683-689`, `:717-720`, `:743-747`），因此该异常不会走新鲜成功缓存写入；之后沿现有错误分类/重试路径处理（`:810-845`），不代表保证立即终态。

## 验证与证据边界

- 按卡面命令运行 `git diff --check ec0b892035ff1def7e1168254b5b89622c5e7bae..4e49d35070b49f678ea9b4692d7a39c37c49da40`，退出码 0。
- OCR 前置扫描状态为 `reviewed`（MiniMax-M3），返回 5 条候选意见；均未采纳：格式风格建议不影响契约；保留 `ValueError` 类型不是本卡要求，现有 `classify_error` 对该异常走既有字符串兜底并归为 `ENGINE_ERROR`；sentence_info 元素类型校验超出本卡要求的顶层结构边界。OCR 的内置复核按执行 worktree 的 base 内容反驳了 H0 新代码，故只把候选意见当线索，逐条按冻结 SHA 复核。
- 未运行 pytest 或 Mac FunASR 模型；结论基于冻结 SHA 的 diff、规格、源码和测试静态核对。测试没有单独以裸 dict 作为有效模型结果贯穿 `transcribe()`；源码仍显式接受 dict 顶层（`:395-396`），目前未据此确认功能缺陷。
- 冻结 diff 为 186 行新增、57 行删除，超过卡面 Diff-Lines-Hard=150；卡面机器预算却记录 base=head、diff=0。该预算元数据不一致作为派发记录事项，不构成结果契约 finding。
