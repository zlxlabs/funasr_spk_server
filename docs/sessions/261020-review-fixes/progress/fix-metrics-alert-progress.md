# fix-metrics-alert 进度存档

## 里程碑 1：失败测试已落盘（TDD 红）

- **当前阶段**：implementing — 先写测试见红。
- **本段结论**：新增 `tests/unit/test_metrics_alert_send.py`，从 `main()` 公开入口
  覆盖四条卡面行为：mock send 失败非 0 且 state 不记；mock 成功退出 0 且发送前不记
  state；通知未启用且有告警非 0；`--selftest` 失败非 0。四条都断言 `ALERT-SEND-FAILED`
  出现在 ERROR 日志（成功路径除外）。
- **关键决策与已否决方案**：测试只 mock `httpx.Client`（探测）和
  `send_custom_notification`（发送），不喂 `evaluate_alerts`。否决 subprocess 跑脚本：
  难 mock 发送层。否决给 `main` 加额外关键字参数：保持 argv 公开入口。
- **下一步唯一动作**：跑窄测确认转红，再改 `notification.py` / `metrics_alert.py` 见绿。

## 里程碑 2：实现见绿 + 红验

- **当前阶段**：implementing → verifying。
- **本段结论**：`send_custom_notification` 改为返回 bool（未启用/失败 False，成功 True，
  失败仍记日志）。`metrics_alert.main` 先发送，只把送达成功的 key 写入 state；
  `error_surge` 未送达不推进 `error_total_seen`。任一未送达打 `ALERT-SEND-FAILED` 并
  退出 1。新增 `--selftest`。窄测 4 passed。红验：把
  `test_send_failure_nonzero_and_state_not_recorded` 的 `assert rc != 0` 改成
  `assert rc == 0`，得到 `AssertionError: assert 1 == 0`，再只改回那一行。
- **关键决策与已否决方案**：未送达回退 prev_state，而不是「整轮不写 state」——同轮
  部分成功仍要记已送达的，避免成功的那条下轮重复。否决给 send 再包一层 try-except：
  沿用 notification 既有记日志返回 False，metrics_alert 侧 fail-loud 看返回值。
- **下一步唯一动作**：跑 `bash scripts/run_checks.sh`，scope check，commit+push。

## 里程碑 3：unit 全量与收工

- **当前阶段**：verifying → 收工。
- **本段结论**：窄测 passed。`run_checks.sh` 在本 Linux 工作树出现 CoreML/dylib 与
  file-based pool 时序失败，均不在本卡改动文件内；本卡新增 4 条测试全绿。无新红。
  主干基线派发时不可用，继承红标未能严格判定。
- **关键决策与已否决方案**：工作树无 `.env` 时 config 校验会 sys.exit；测试文件
  `setdefault("FUNASR_NOTIFICATION_ENABLED","false")` 仅在未设置时兜底，不覆盖
  已有 webhook 环境。否决改 conftest / run_checks.sh（越界）。
- **下一步唯一动作**：scope check 后 commit+push，写报告。
