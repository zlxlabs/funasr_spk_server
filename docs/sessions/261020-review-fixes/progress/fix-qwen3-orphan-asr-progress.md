# fix-qwen3-orphan-asr 进度

## 2026-10-20 红测

- 当前阶段：red
- 本段结论：新增 `tests/unit/test_qwen3_transcriber_parallel_failure.py`，从公开入口 `transcribe()` 复现 diarize 立即失败、ASR 仍 in-flight 的孤儿线程；对称覆盖 ASR 失败 / diarize 慢路径。本文件其它 `run_in_executor` 均为顺序 await，无同类并行点。
- 关键决策与已否决方案：测试只 mock `run_asr` / `run_diarization_dispatched` / `build_engine`（最外层模型调用）；用专用异常类钉死类型不被包装。已否决：取消 future（executor 线程无法真正取消，设计文档已否决）。
- 下一步唯一动作：跑窄测见红，再改 `asyncio.gather(..., return_exceptions=True)` 等两路结束再抛首个异常。

## 2026-10-20 红测见红

- 当前阶段：red-confirmed
- 本段结论：`FUNASR_NOTIFICATION_ENABLED=false venv/bin/python -m pytest tests/unit/test_qwen3_transcriber_parallel_failure.py -q --tb=short` 两测均红。`AssertionError: transcribe 抛出时 ASR executor 线程必须已经结束 / assert 1 == 0`（对称路径 diarize in-flight 同理）。异常类型已原样抛出，卡在「未等另一路结束」。
- 关键决策与已否决方案：本文件其它 `run_in_executor` 均为顺序 await，只修 gather 这一处。已否决取消 future。Linux 工作树无 `.env`，跑测需 `FUNASR_NOTIFICATION_ENABLED=false`（默认 enabled=True 无 webhook 会 sys.exit），不入库 .env。
- 下一步唯一动作：改 gather 为 `return_exceptions=True`，两路都结束后打 `QWEN3-PARALLEL-BRANCH-FAILED` 并原样抛首个异常。

## 2026-10-20 绿测

- 当前阶段：green
- 本段结论：`asyncio.gather(..., return_exceptions=True)` 收齐两路后打 `QWEN3-PARALLEL-BRANCH-FAILED` 并 `raise` 首个原异常。窄测 2 passed。未新增 helper/抽象。本文件无其它并行 `run_in_executor` 点。
- 关键决策与已否决方案：按卡面锁定写法，不取消 future、不包一层新异常类型。失败可见性字面量落在 error 日志。
- 下一步唯一动作：提交修复；用「只改判据那一行」做红验后还原；跑 `scripts/run_checks.sh` 与 scope check。
