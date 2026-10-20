# 进度存档：fix-qwen3-silent-drop（卡 4：Qwen3 静默丢转写 / 丢右声道）

## 2026-10-20 段 1（dispatch dlg-20261020-124823-35540e）

**当前阶段**：implementing 完成，待主脑验收。

**本段结论**：
- 缺陷 (a) 修复：`merge_asr_and_diarize` / `merge_asr_chunks_and_diarize` 在 turns 为空但
  ASR 有文本时不再 `return []`——全文归 speaker 0 单段返回（chunks 版时间取各 chunk
  起止），打含 `DIARIZE-NO-TURNS` 标记的 WARNING；ASR 文本为空仍返回 `[]` 不打日志。
- 缺陷 (b) 修复：`audio_io.load_audio_mono_16k` soundfile 路径由 `audio[:, 0]` 改为
  `audio.mean(axis=1)` 平均混音。
- 核实（卡面待验证前提 2）：Qwen3 ASR 侧读音频**确为平均混音**——
  `src/core/vendor/qwen_asr_gguf/inference/audio.py` `load_audio_numpy` 用
  `audio.mean(axis=1)`、ffmpeg fallback 用 `-ac 1`。diarize 侧改平均混音后与 ASR
  对齐，方向正确。
- 下游高风险区确认：单说话人输入经 `relabel_segments_by_duration_desc`（唯一 speaker
  → 0，无变化）、`apply_short_segment_guard`（`drop_tiny_segments` 对无邻居单段走
  `out.append` 不丢文本；全 speaker=0 时 ABA 平滑无 A-B-A 模式可动、同 spk 合并只减
  碎片）均安全，不报错不丢文本。
- 测试：`tests/unit/test_qwen3_merge_empty_turns.py`（6 用例）+
  `tests/unit/test_qwen3_audio_io_downmix.py`（2 用例）先红后绿；红验有效性用最小注入
  （无 turn 分支插回 `return []` / 混音行改回 `[:, 0]`）三路注入均转红（AssertionError），
  精确还原后绿，未用 git stash / checkout。
- 全量 unit（`bash scripts/run_checks.sh`）：1129 passed, 6 skipped, 3 failed,
  10 errors。**失败清单与 base 7b8bfaa 干净 worktree 全量跑逐条一致（差集为 0，
  passed 差 8 恰为本卡新增用例）→ 全部继承红，零新红**。继承红根因：本机为 Linux，
  vendor `qwen_asr_gguf/inference/bin/` 只有 macOS dylib（dlopen 失败：libggml.so 不存在
  / invalid ELF header），波及 test_qwen3_encoder_* 与 test_config_qwen3_asr_encoder_provider
  真实 import vendor 的 10 error + 1 failed；test_file_based_pool_lifecycle 2 failed 为
  全量顺序下时序抖动（单跑复现不定，base 同样）。

**关键决策与已否决方案**：
- 按卡面锁定修法执行；「diarize 无 turn 判任务失败」已否决（主脑定案）。
- `merge_asr_and_diarize` 无 turn 段时间置 `start=0.0, end=0.0`：函数签名无时间输入，
  且 Scope 禁改 caller（`qwen3_transcriber.py:897` 不在本卡 scope）；经查
  `drop_tiny_segments` 对「微短且有文本但无邻居」的段走 `out.append(seg)` 保留文本，
  0 时长不会导致丢字，SRT 出口只按空文本跳段。
- 环境修复（非源码改动）：共享 venv 的 pytest 被漂移升级到 9.1.1/pytest-asyncio 1.4.0，
  与仓 pin（requirements.txt: pytest==7.4.4 / pytest-asyncio==0.23.3）不符，全仓测试
  fixture setup 即 AssertionError；已装回 pin 版本。worktree 无 `.env` 导致
  `config.py` import 时单例校验 `sys.exit(1)`（NotificationConfig.enabled 默认 True），
  按 .gitignore 约定在 worktree 根放非凭据 `.env`（仅 `FUNASR_NOTIFICATION_ENABLED=false`）。

**下一步唯一动作**：等待主脑验收本卡（报告已写 $DELEGATE_REPORT_PATH）。
