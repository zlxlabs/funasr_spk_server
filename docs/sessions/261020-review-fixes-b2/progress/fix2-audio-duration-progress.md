# fix2-audio-duration 进度

## 2026-10-20：实现与验收
- 当前阶段：实现完成，准备提交与推送。
- 本段结论：`get_audio_duration` 已按容器时长、音频流时长、ffmpeg 解码时间顺序探测；探测失败会记录 `AUDIO-DURATION-PROBE-FAILED` 并抛出含原因的异常。新增测试覆盖真实无 Duration WebM、流时长、探测失败、解码失败与 FunASR 公开入口不误报「音频时长过短」。临时测试环境 Narrow 测试 5 项通过；验收脚本因 checkout 未提供 `venv/bin/python` 未能启动。
- 关键决策与已否决方案：保持 `funasr_transcriber.py` 现有时长阈值判断与错误分类，不额外改分类；探测异常原样包装为引擎失败，不回退成 0.0；无容器和流时长时按卡面要求运行 ffmpeg 解码统计。
- 下一步唯一动作：运行范围检查并提交、推送本卡改动，然后写入最终报告与远端核验结果。
