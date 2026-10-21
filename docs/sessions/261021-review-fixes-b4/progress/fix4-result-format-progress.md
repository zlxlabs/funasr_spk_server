## 进度三：实现与验证

- 当前阶段：目标行为已修复，提交收尾中。
- 本段结论：4 个新增定向测试先复现旧行为失败，修复后窄测试 7 项通过；逐项反向注入断言均以 AssertionError 转红。全量单测 1231 通过、6 跳过、1 失败：既有 `test_funasr_parse_sentence_level.py::test_unknown_or_missing_text_shape_fails[result0]` 仍要求顶层 `[]` 抛错，与本卡 R4 新契约相反；该文件不在 Scope-Globs 内，未越界修改。基线作业超时，继承红未能判定。
- 关键决策与已否决方案：顶层空列表按合法空结果处理并单独发出 `FUNASR-EMPTY-RESULT` WARNING；毫秒统一先 round 总毫秒再 divmod；缓存 cutoff 按 UTC 生成；terms 错误使用专属 marker。按范围约束保留旧测试文件不动，也不触碰 result_projection / qwen3 文件。
- 下一步唯一动作：提交本卡允许路径的改动，执行提交范围检查并推送后核实远端。
