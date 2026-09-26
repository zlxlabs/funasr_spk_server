# 实现交接:qwen3 词级时间戳(MMS-300M CTC-FA,增量挂 words)

> 新 session 自主执行用。设计 + PoC + CEO/Eng 双评审都已完成,计划定稿。本文件自包含,照着 TDD 顺序从头干到尾即可。

## 执行方式(重要)
- **严格 TDD**:每步先写红测试 → 实现转绿 → 立即 commit。红→绿→commit 是最小单位,不要攒多个改动一次提交。
- **非必要不停下来提问**:计划已定死(见下方决策表),除非遇到跟计划冲突的硬阻塞,否则一路做到整个任务完成。
- **完成后更新文档**:任务做完及时更新 `docs/开发/2026-06-09-qwen3-词级时间戳-PoC计划.md`、`CLAUDE.md`(Qwen3 后处理 pipeline 一节加 word_align 层)、`spikes/qwen3_word_timestamp/SUMMARY.md`,确保文档跟代码同步。
- **本项目独立维护,不要建议开 PR**(记忆 `feedback_no_pr_workflow`)。直接 commit 到当前分支 `spike/qwen3-diarize-poc`。
- commit message 末尾带 HAPI 署名(`via [HAPI](https://hapi.run)` + `Co-Authored-By: HAPI <noreply@hapi.run>`)。

## 任务一句话
给 qwen3 引擎加**词级时间戳**:用 **MMS-300M CTC-FA**(`ctc-forced-aligner==1.0.2` deskpai ONNX fork)拿 `(audio, ASR文本)` 出每个词的时间,**增量挂进 `segment.words`**,不替换现有 merge。

## 背景(为什么这么做)
- ASR(Qwen3 GGUF)给整段文本 + 粗 40s chunk 窗,**无词级时间**。diarize 给 speaker turn。现有 `merge.py:216 merge_asr_chunks_and_diarize` 按字符比例把文本切到 turn(近似),silence_align 再吸附,段边界已 ~200-260ms 准。
- PoC(`spikes/qwen3_word_timestamp/SUMMARY.md`)测了 3 个对齐器,选 MMS:多语种(需求是中英以外还有其他语言)、纯 ONNX 契合现有 onnxruntime 栈、内置切块、中文精度≈Qwen aligner(AAS 58ms)、Mac CPU RTF 0.166。
- CEO + Eng 双评审 + codex 跨模型挑战定稿:**增量加 words 是低风险主价值,替换段边界是高风险边际改善(留 TODOS #14)**。

## 决策表(定死,照做)
| 项 | 决策 |
|---|---|
| 集成 | **增量**:新 `attach_words_to_segments(segments, words)` 把 MMS 词按时间窗挂进现有段。**不动 `merge_asr_chunks_and_diarize`** |
| silence_align | 保持开(段边界仍由它产出) |
| 语言来源 | **per-request 语言字段**(ISO 码 chi/eng/jpn/kor…)穿 schema→websocket→task_manager→transcriber;config 默认语言兜底 |
| 模型加载 | 自建 `onnxruntime.InferenceSession` 传 runtime-aware providers(Mac→CPUExecutionProvider / CUDA→CUDAExecutionProvider),调低层函数,**不用 deskpai AlignmentSingleton**(它运行时下载 + 写死 CPU)。模型预下到本地路径 |
| 依赖 | `ctc-forced-aligner==1.0.2` + `unidecode` 进 requirements。vendor 当逃生口(暂不做) |
| 暴露 | 嵌套 `segment.words: Optional[List[WordTimestamp]]`;`WordTimestamp(text:str, start:float, end:float, confidence:Optional[float])` 绝对秒;JSON-only(SRT 不带);默认 None 向后兼容 |
| fallback | **逐 window + stats**:某 chunk 对齐失败 → 该 chunk 段 words=None,段照常出(不崩);stats 记失败 window 数 + 原因 |
| 缓存 | **word_align 状态进缓存 key**(加字段是契约变化,key 现有 `(file_hash, engine)` 加一维) |

## 关键代码事实(挂钩点,带 file:line)
- `src/core/qwen3_transcriber.py:transcribe`(367-543):pipeline 顺序 = ASR → diarize(397)→ filter_spurious(413)→ cluster_merge(427)→ **merge_asr_chunks_and_diarize(441)** → short_guard(447)→ silence_align(462)→ relabel(489)→ TranscriptionSegment(535)。**word_align 挂在 silence_align 之后、relabel 之前**(在干净段上挂词)。
- `src/core/qwen3_transcriber.py:445 apply_short_segment_guard_to_segments`:把 Segment 转 dict(只 start/end/speaker/text)再转回 —— **必须改成透传 words,否则静默丢词(codex #6,critical 回归测试)**。
- `src/models/schemas.py:19 TranscriptionSegment`:加 `words` 字段 + 新 `WordTimestamp` model。
- `src/core/qwen3/merge.py`:`Segment` dataclass(start/end/speaker/text)加 words;新 `attach_words_to_segments` 放这里。
- `src/core/config.py:89 Qwen3Config`:加 word_align 5 件套(enabled / language / model_path / provider(auto) / batch_size),+ `_apply_env_overrides` 注 env,+ profile(可选)。
- `src/core/runtime.py`:`recommend_*` 加 word_align provider 推荐(Mac→CPU / CUDA→CUDA),或复用现有 provider 解析。
- diarize dispatch 模板:`src/core/qwen3/diarize.py:149 run_diarization_dispatched` + `diarize_ort.py`(自建 ORT session 的范例,照抄)。

## MMS / ctc-forced-aligner deskpai API(1.0.2,已装在 venv)
低层函数(不用 AlignmentSingleton):
```python
from ctc_forced_aligner import (
    generate_emissions, preprocess_text, get_alignments, get_spans, postprocess_results,
)
import onnxruntime, numpy
# 自建 session(providers 由 runtime 选)
session = onnxruntime.InferenceSession(local_model_path, providers=[...])
emissions, stride = generate_emissions(session, audio_waveform_numpy_1d_16k, batch_size=16)
tokens_starred, text_starred = preprocess_text(text, romanize=True, language="chi")  # ISO: chi/eng/jpn/kor
segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)  # tokenizer = ctc_forced_aligner.Tokenizer()
spans = get_spans(tokens_starred, segments, blank)
word_ts = postprocess_results(text_starred, spans, stride, scores)  # [{text,start,end,score}], 秒
```
- 模型 ONNX:`MODEL_URL = huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx`(~预下到本地,别运行时下)。
- 语言码:`preprocess_text` 里 `chi/jpn` 触发逐字切;`romanize=True` 需 `unidecode` + uroman(包内置)。中英混排用 `chi`(实测能吃英文,英文词被 char-split 罗马化)。
- 喂法:**按 ASR chunk 喂**(`asr_result.chunks` 每个 chunk 的 audio 切片 + chunk.text → 词时间 + offset chunk.start 拼接),不要喂整文件 monolith(trellis OOM)。
- 段文本保 ASR 原文(标点/格式),**不要用对齐器吐的归一化 token 拼**(codex #4)。
- 词→段归属:词按时间落在哪个段的 [start,end] 窗就挂哪个段;边界用最大重叠时长优先(codex #5)。
- 参考可跑脚本:`spikes/qwen3_word_timestamp/scripts/poc_mms_ctc.py`(完整调用流程)、`compare_chars.py`(精度基准复用)。

## TDD 实现顺序(每步红→绿→commit)
1. **WordTimestamp schema + TranscriptionSegment.words 字段** — unit:序列化、words=None 向后兼容。
2. **config word_align 5 件套** — unit:默认值 / env override / provider auto 解析(mock detect_runtime)。
3. **word_align wrapper**(`src/core/qwen3/word_align.py`)— unit:mock onnxruntime session,验证 happy / 模型加载失败 / 对齐异常 / 空文本;runtime provider 选择。
4. **attach_words_to_segments 纯函数**(merge.py)— unit:词挂段(时间窗 + 最大重叠)/ 空 words / 空 segments / 词不落任何段 / 跨段。
5. **per-request 语言字段** — 穿 schema→websocket→task_manager→transcriber + config 兜底;unit 覆盖透传。
6. **short-guard 透传 words**(critical)— 回归测试:guard 后 words 不丢。
7. **transcribe pipeline 挂钩** — word_align flag 开:silence_align 后挂词(逐 window fallback);flag 关:老路不变。进度条插更新(别卡 80%)。raw_result 加遥测。
8. **缓存 key 加 word_align 维**。
9. **integration**(`FUNASR_RUN_INTEGRATION=1`)— parity(flag 关字节不变,golden 不破)+ fallback(mock MMS 抛异常仍出段)+ 真 MMS 跑 `tests/fixtures/audio/podcast_2speakers_60s.wav` 出 words + 轻量精度基准(复用 compare_chars.py 思路,AAS 不退化超阈值)。
10. **requirements.txt** 加 `ctc-forced-aligner==1.0.2` + `unidecode`;模型预下脚本(进 `scripts/download_qwen3_models.sh` 或并列)。

## 测试约定(CLAUDE.md)
- 三层:`tests/unit/`(mock,毫秒)/ `tests/integration/`(真模型,默认 skip,需 `FUNASR_RUN_INTEGRATION=1`)/ `tests/manual/`(不收集)。
- 命令:`venv/bin/python -m pytest`(venv pytest shebang 漂移,走 `-m pytest`)。
- **改 Qwen3 路径必跑 parity**:`FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/`。通过=安全。

## 收尾(任务完成后)
- 更新 `CLAUDE.md` 的「Qwen3 后处理 pipeline」一节:加 word_align 层说明(挂在 silence_align 后、relabel 前;flag + env;逐 window fallback)。
- 更新 PoC 计划文档落地状态(标已实现)+ SUMMARY。
- diarize 开关议题现在可塌缩成纯布尔(时间戳恒有基线)—— 若要顺手做,见 `docs/开发/2026-06-09-qwen3-diarize开关API-设计讨论-新session-prompt.md`。
- CUDA provider 要在 3060 实测(`ssh <user>@<dev-host>`,记忆 `reference_cuda_box`):MMS ONNX 用 CUDAExecutionProvider + 跟 llama.cpp CUDA 共存验证。

## 相关记忆 / 文档
- 记忆:`project_word_timestamp_multilang`(多语种 + 最终决策)、`reference_qwen3_aligner_weights`(Qwen aligner 权重,本任务不用但留底)、`feedback_tdd_strict`、`feedback_no_pr_workflow`、`reference_cuda_box`。
- 文档:`docs/开发/2026-06-09-qwen3-词级时间戳-PoC计划.md`(落地计划定稿)、`spikes/qwen3_word_timestamp/SUMMARY.md`(PoC 数据 + 脚本)、`TODOS.md #14`(Phase 2 替换式 merge)。
