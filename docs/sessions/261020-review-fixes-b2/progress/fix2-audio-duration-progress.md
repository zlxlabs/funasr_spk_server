# fix2-audio-duration 进度

## 2026-10-20：实现与验收
- 当前阶段：实现完成，准备提交与推送。
- 本段结论：`get_audio_duration` 已按容器时长、音频流时长、ffmpeg 解码时间顺序探测；探测失败会记录 `AUDIO-DURATION-PROBE-FAILED` 并抛出含原因的异常。新增测试覆盖真实无 Duration WebM、流时长、探测失败、解码失败与 FunASR 公开入口不误报「音频时长过短」。临时测试环境 Narrow 测试 5 项通过；验收脚本因 checkout 未提供 `venv/bin/python` 未能启动。
- 关键决策与已否决方案：保持 `funasr_transcriber.py` 现有时长阈值判断与错误分类，不额外改分类；探测异常原样包装为引擎失败，不回退成 0.0；无容器和流时长时按卡面要求运行 ffmpeg 解码统计。
- 下一步唯一动作：运行范围检查并提交、推送本卡改动，然后写入最终报告与远端核验结果。

## 2026-10-20：提交与推送
- 当前阶段：实现已提交并推送；Linux checkout 的项目验收环境缺失。
- 本段结论：提交 `8d61405` 已推送至 `origin/card/fix2-audio-duration`，远端 `git ls-remote` 核实 SHA 一致；范围检查通过。临时轻量环境中新增测试 5 项通过；卡面要求的 `venv/bin/python` 和 `bash scripts/run_checks.sh` 均因 checkout 中不存在 `venv/bin/python` 而无法启动。Linux 未运行 FunASR parity。
- 关键决策与已否决方案：未为绕过本机环境缺失而在仓库内创建或替换 venv；保留现有 FunASR 入口错误包装和分类，让探测异常作为普通引擎异常按现有逻辑可重试。
- 下一步唯一动作：在具备 FunASR 项目 venv 的 macOS 环境运行 Narrow-Verify、`bash scripts/run_checks.sh` 与 FunASR parity 并回报结果。
