# fix-funasr-hotword 进度

## 2026-10-20 20:56 测试见红

- 当前阶段：implementing / TDD 红
- 本段结论：对照 PyPI funasr 1.2.7 wheel（本机 worktree venv 只有 AutoModel 桩）`funasr/models/seaco_paraformer/model.py` `generate_hotwords_list`：L545 本地 `.txt` 且 `os.path.exists` 才按行读；L549-556 `line.strip()` 后整行进 `hotword_str_list`（行内空格保留），`hw.split()` 只用于分词；L565 `startswith("http")` 会 `requests.get`；L605-606 以 `.txt` 结尾但文件不存在则 `hotword_list=None`。空格拼接会踩这三条。已新增 `tests/unit/test_funasr_hotword_file.py` 并改 propagation 断言，现实现仍 `" ".join(terms)`，测试应红。
- 关键决策与已否决方案：修法锁定为每任务唯一临时 `.txt` 路径作 hotword；已否决对 terms 做 http/.txt 字符串过滤（仍会拆开含空格术语）。
- 下一步唯一动作：改 `src/core/funasr_transcriber.py` 把 terms 写入临时 `.txt`，转写结束删除，失败日志带 `FUNASR-HOTWORD-FILE`。

## 2026-10-20 21:00 实现见绿

- 当前阶段：implementing / TDD 绿
- 本段结论：`funasr_transcriber.transcribe` 把非空 terms 写成每任务 `.txt`（UTF-8，每行原样）把路径作 hotword，成功/失败都在 finally 删除；写文件失败打 `FUNASR-HOTWORD-FILE` 后原样上抛（写文件在转录 try 外，避免被包成「转录失败」）。`context_applied` 仍只在成功出口且 terms 非空时为 true。窄测 17 passed。红验把 `lines == TRICKY_TERMS` 改成 `lines == []` 得到 `AssertionError: assert ['config.txt'...'hello world'] == []`。
- 关键决策与已否决方案：写文件放转录 try 外以 fail-fast 保留 OSError；已否决 http/.txt 字符串过滤。无 terms 仍传空串、不建文件。
- 下一步唯一动作：跑 `bash scripts/run_checks.sh` 与 scope check，小步 commit+push。
