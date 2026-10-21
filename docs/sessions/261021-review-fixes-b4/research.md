# 第四批调研：审查残留条目在主干的复现与定位

基线核查：当前分支 `card/fix4-research` 的 HEAD 为 `3fd1b2a18ee2030dbfb97477529c5475304a990d`，与任务卡指定主干一致；以下均按该提交重新定位。原审查行号基准 `7b8bfaa` 不直接沿用。结论中的严重度按 personal 档：数据丢失、静默错误或崩溃为 P1，其余不高于 P2。

## 执行环境与复现边界

本机为 Linux、Python 3.12.3；没有 `venv/bin/python`、`loguru`、`psutil` 或 `nvidia-smi`，不是项目 Mac/CUDA 消费环境。`bash scripts/run_checks.sh` 实际输出：

```text
==> unit tests (hermetic, 毫秒级)
scripts/run_checks.sh: line 23: venv/bin/python: No such file or directory
```

尝试 `python -m pytest -q tests/unit/test_funasr_parse_sentence_level.py tests/unit/test_error_classification.py tests/unit/test_qwen3_worker_pool_fixes.py`，pytest 收集阶段报 `ModuleNotFoundError: No module named 'loguru'`；另一组 `tests/unit/test_http_endpoints.py tests/unit/test_http_endpoints_live.py tests/unit/test_database_cache_sizecap.py tests/unit/test_file_based_pool_lifecycle.py` 同样在收集阶段因缺 `loguru` 失败。未把环境依赖安装或伪造为测试通过。对可在 Linux 检验的纯逻辑，使用实际源码函数（必要时 AST 提取以绕过缺失的项目依赖）或注入受控依赖执行，并逐条贴出结果；需 Mac/CUDA 或真实模型的结论明确注明未实机验证。

### R1 — Qwen3 terms 没有上下文 token 预算

(a) `src/core/qwen3/asr.py:60-64` 只拼接 context；vendor 的 `src/core/vendor/qwen_asr_gguf/inference/schema.py:87-89` 指 ASR decoder `n_ctx=2048`，每秒音频/文字约 20 token、chunk 40 秒。`src/core/vendor/qwen_asr_gguf/inference/asr.py:299-305` 把当前音频与最多两段历史音频、历史文字和 context 合并后直接解码，无 terms 预算或截断检查。

(b) Linux 最小探针调用实际 `build_qwen_context`，使用 schema 可接受的 16 个 64 字中文术语（总计 1024 字）；该探针验证拼接保留尾项，不代表真实 GGUF tokenizer/推理行为：

```text
valid_shape_terms=16 term_chars=1024 context_chars=1081
context_tail_preserved=True
```
vendor `asr.py` 中历史音频记忆队列上限为两段，解码输入把历史片段与当前片段一起拼入，因此文本术语预算还要与实际音频 embedding 位置竞争 2048 上限。schema 的 token/秒只是估算，不足以替代实际 tokenizer/embedding 长度。无本机 Qwen GGUF/Metal 模型及 tokenizer 环境，未复现实际 decode 返回码、静默截断或 abort；也没有代码证据证明底层会自动截断。

(c) **成立**：没有预算/越界校验成立；“一定静默截断或 abort”未证实，风险机制成立但具体表现取决于 tokenizer 与推理库。

(d) 最小修法：在构造实际 embedding 后按模型 `n_ctx` 检查总输入位置；不静默删术语，超限时明确报输入错误。涉及 Qwen3 ASR/vendor 边界（优先在本项目调用侧实现）；需 Qwen3 单测/模型集成验证，不需 FunASR parity。

(e) **P2**：可能导致 Qwen3 识别失败或术语上下文效果不稳定；无证据表明会丢失上传音频数据。

### R2 — 内容寻址上传文件被原地截断重写

(a) `src/utils/file_utils.py:42-55` 按 MD5+扩展名生成共享路径，随后以 `'wb'` 打开原路径。分片上传还有同类路径：`src/api/websocket_handler.py:1005-1018` 以 `shutil.copyfile` 直接复制到 `{file_hash}{ext}`。

(b) 对实际 `save_uploaded_file` 做 Linux 并发探针：第一次上传完成后保留读句柄；第二次同内容、同扩展名上传在真实 `'wb'` 打开/截断后暂停写入，读取旧句柄。输出：

```text
shared_path=6ee1f1da89fce3f575e18c7bcb4ccdf6.wav
reader_bytes_during_second_wb=0
final_file_bytes=23
```
读者在第二次写入尚未完成时确实读到零字节；最后文件恢复完整不能挽回已读到的空副本。

(c) **成立**。

(d) 最小修法：先在上传目录同一文件系统写唯一临时文件，完成后 `os.replace` 原子发布；统一用于 `save_uploaded_file` 与分片 finalize 的 `copyfile` 路径，不能只修一个入口。涉及 `src/utils/file_utils.py`、`src/api/websocket_handler.py`（或两处调用同一个有第二消费者的原子发布 helper）及并发回归测试。影响两引擎上传；因改 `websocket_handler.py`，按项目约定需跑 FunASR parity。

(e) **P1**：并发任务可能读取到空/截断音频，属于输入数据丢失/静默错误。

### R3 — worker 回收失败掩盖已成功的推理结果

(a) `src/core/file_based_process_pool.py:680-698` 先读并校验 result，成功分支却先 `await _release_worker_lease(...)`，再返回结果。`_finish_worker_lease_release` 在 `:547-566` 中 reap 或删除资源失败会原样抛错；外层 `:716-723` 再次尝试 release，调用者收不到已读出的成功结果。

(b) Linux 以真实 `generate_with_pool` 方法执行合成成功 result，并注入 lease 回收 OSError；这是故障注入，不代表真实进程的回收失败频率：

```text
caller_exception=OSError: synthetic reap/cleanup failure
success_result_file_still_present=True
producer_success=True
```
producer 已发布成功结果，pool API 仍向调用者抛回收异常，而不是返回结果。实际清理若已删 result 文件，结果还会不可恢复。

(c) **成立**。

(d) 最小修法方向：把已验证的成功结果与资源回收状态解耦；回收失败仍须记录并保留 lease 归属交给既有 cleanup 重试，不能让已成功推理被转成任务失败。涉及 `src/core/file_based_process_pool.py` 与对应生命周期/故障注入单测。该池被 FunASR 与 Qwen3 共用，应跑池单测及 FunASR parity（并验证 Qwen3 池调用）。

(e) **P1**：成功结果丢失/任务被错误判失败，可能重复执行甚至最终丢失结果。

### R4 — FunASR 顶层空列表被判异常并进入可重试分类

(a) `src/core/funasr_transcriber.py:426-429` 对 `[]` 调 `invalid("empty result list")`，抛 `ValueError`。`src/core/task_manager.py:97-112` 将其分类成默认 `ENGINE_ERROR`；`:843-875` 对该 kind 在 `retry_times` 内原地重试（默认配置 2 次，`src/core/config.py:231`）。

(b) Linux AST 探针执行当前 validator 与当前 classifier 函数（纯 Python，未加载 FunASR 模型）：

```text
logged_invalid_shape=True
validator=ValueError: FunASR result invalid-shape: empty result list
classify_error=engine_error retryable=True
```
未在本机运行真实 FunASR 模型，不能断言空列表来自无语音还是推理异常。

(c) **成立**：对空列表的失败和可重试分类都可直接复现；“空列表应当是成功空转录”须由 FunASR 输出契约决定。

(d) 最小修法：先确认 FunASR 对无语音输出的约定；若 `[]` 是合法空转录，规范化为空 `sentences=[]` 并完成空结果；若它代表无效模型输出，则至少给该明确形状作不可重试输入/结果分类，避免无效重试。涉及 `funasr_transcriber.py`、`task_manager.py` 与单测；触及 FunASR 路径，必须跑 parity。

(e) **P2**：重复无效重试并最终误报失败；没有音频数据丢失证据。

### R5 — Mac Qwen3 开 word_align 时池超时是否偏紧

(a) 共享公式在 `src/core/file_based_process_pool.py:728-747`：`300 + duration*0.3` 秒，钳制 `[600, 3600]`，不接收 options/word_align。Mac/非 CUDA Qwen3 使用该 file-based pool；`docs/开发/ASR引擎架构.md:64` 与 `docs/部署.md:385-386` 记载 Mac CPU word_align 增量 RTF 约 `+0.166`。

(b) Linux 注入 3600 秒音频时长调用实际公式：

```text
platform=linux duration_s=3600 timeout_s=1380 formula=300+0.3*duration capped[600,3600]
word_align_option_is_not_an_input_to_calculate_timeout=True
```
本机无 Mac，不能在 Mac 上测实际整体 RTF/超时边界；公式复现只证明不按 word_align 加预算，文档增量 RTF 也不能单独证明完整转录必超过 1380 秒。

(c) **无法判定**：风险因素成立，但“生产 Mac 上公式实际偏紧/会超时”缺少同 workload 的实测数据。

(d) 若 Mac 长音频实测显示超时，最小修法是在 Qwen3 JSON 且 `options.word_align` 的情况下纳入实测增量（目前文档给 `+0.166 RTF`），保持 FunASR 和未开 word_align 的预算不变。涉及 file-based pool/Qwen3 选项传递与超时单测；池共用，建议跑 FunASR parity，并做 Mac Qwen3 长音频验证。

(e) **暂列 P2 风险**（尚非已确认缺陷）；只有长任务失败/重跑风险，没有已证数据丢失。

### L1 — SRT 毫秒格式化存在截断

(a) `src/core/task_manager.py:980-987` 用 `int((seconds % 1)*1000)`；`src/core/funasr_transcriber.py:541-549` 与 `src/core/database.py:655-662` 对毫秒值也先转秒再 `int`。对照：`src/core/result_projection.py:215-216` 及 `src/core/qwen3/merge.py:564` 使用 `round`。

(b) Linux AST 探针运行实际 task-manager 格式化函数：

```text
legacy_seconds_to_srt_time(1.2349)=00:00:01,234
source_formula=int((seconds % 1) * 1000) => truncates fractional millisecond
```
`cx references --name _convert_json_to_srt` 只列 `src/core/task_manager.py:963` 自身定义；该 legacy converter 无调用者。FunASR 上游通常提供整数毫秒，因此不能仅凭此探针证明当前每个用户出口都丢有效精度；浮点秒/小数毫秒输入则明确会截断。

(c) **成立**（格式化实现对非整数毫秒输入截断；活跃出口影响依具体输入形状）。

(d) 最小修法：先换算成整数总毫秒 `round(seconds*1000)`，用 `divmod` 生成时间字段以正确处理进位；原始毫秒路径也先 round 成整数。涉及 `task_manager.py`、`funasr_transcriber.py`、`database.py`（必要时统一投影 formatter）和精度/进位测试。改 FunASR SRT 路径须跑 parity。

(e) **P2**：最多为字幕时间戳亚毫秒量化误差；未达 P1。

### L2 — cache TTL cutoff 与 SQLite UTC 时间存在时区差

(a) `src/core/database.py:522-530` 已用正确的空格分隔 `strftime`（不是 `isoformat()`），但 cutoff 由本地 `datetime.now()` 生成；SQLite `CURRENT_TIMESTAMP` 是 UTC。格式一致不等于时区一致。

(b) Linux 标准库探针设置 `TZ=America/Los_Angeles`，对比本地时钟与 SQLite：

```text
local_now=2026-10-20 22:38:06
sqlite_current_timestamp_utc=2026-10-21 05:38:06
local_minus_utc_hours=-7
```
用本地值做 cutoff 与 UTC 文本比较会在该时区造成 7 小时 TTL 偏差（此示例中延迟删除）；其他时区方向/时长随偏移变化。这里保留空格格式，未使用 `isoformat()`。

(c) **成立**：当前比较有时区语义错位；现有注释只解决字符串分隔符问题。

(d) 最小修法：用 UTC 计算 cutoff（`datetime.now(timezone.utc)`，移除 tzinfo 后按 SQLite 格式序列化），保持 SQLite `CURRENT_TIMESTAMP` 空格格式。影响 `database.py` 和 UTC 边界测试；数据库缓存跨两个引擎，改此文件按项目约定跑 FunASR parity。

(e) **P2**：缓存过期最多按部署时区偏移早/晚若干小时，不是数据损坏。

### L3 — /metrics 读取不存在的键

(a) `src/api/http_endpoints.py:76-105` 对 task snapshot 的必需键均由 `src/core/task_manager.py:1056-1069` 提供；可选 counters 使用 `.get`。cache stats 的 `total_cached/hits/misses` 通过 `if key in cache_stats` 检查后才读。`src/core/database.py:587-592` 实际只提供 `total_count/today_count/cache_size_mb/projected_serves`。

(b) Linux AST 探针用实际 `render_prometheus` 函数和 `get_cache_stats` 形状执行：

```text
render_completed=True cache_lines=['funasr_cache_projected_serves 3']
missing_cache_gauges=cache_total_cached,cache_hits,cache_misses (membership-guarded, no KeyError)
```
没有 KeyError/不存在键访问；但总缓存数、命中、未命中等预期 gauge 会因为键名不匹配而被省略。

(c) **不成立**（“读取不存在的键”不复现）；另有 cache gauge 未对齐的可观测性缺口。

(d) 原指控无需修；若要补缓存 gauge，先明确现有数据库统计能提供哪些量，再把 renderer 名称/producer 字段对齐并单测。改 `http_endpoints.py` 即可；不涉及 FunASR parity。不得把无来源的 hits/misses 伪造为 0。

(e) **P2（仅旁述的 gauge 缺失）**；原始指控不适用严重度。

### L4 — 顺带确认的无引用函数

(a) 限定检查上述模块范围，不做全仓死代码扫描。`src/core/task_manager.py:963` 的 `_convert_json_to_srt` 与 `src/core/file_based_process_pool.py:753` 的 `_is_worker_alive`。

(b) `cx references --name _convert_json_to_srt` 输出仅有 `src/core/task_manager.py:963`（refs=1，自身定义）；`cx references --name _is_worker_alive` 输出仅有 `src/core/file_based_process_pool.py:753`（refs=1，自身定义）。`cx` 的 refs 计入定义行，两个函数均没有调用者。

(c) **成立**：在当前索引可见的 Python 引用中没有消费者；动态字符串调用未逐一证明不存在。

(d) 若后续清理，直接删除并移除相关无用调用链，不加替代抽象；`task_manager.py` 变更后按项目约定跑 FunASR parity，pool 侧跑 lifecycle 单测。只读调研，本次不删。

(e) **P2**：死代码维护负担，不是运行时故障。

### P1 — 显存 sidecar 预检越过 gpu_mem 与 client 私有边界

(a) `src/core/qwen3_transcriber.py:698-716` 直接读 `client._proc`、调用 `subprocess.run(nvidia-smi)` 并 import/call `gpu_mem._resolve_device_index`（`gpu_mem.py:36-48`）。公开探针 `gpu_mem.free_vram_mib` 在 `:51-81` 已封装设备解析、5 秒超时、失败返回 `None`。

(b) Linux 检查 `uname -s` 为 `Linux`，`command -v nvidia-smi` 无输出；没有 CUDA/Mac sidecar，不能实测显存读数、子进程挂起或 sidecar 生命周期行为。静态代码确认第二次 subprocess 没有 timeout，且 `_proc` 是 client 私有成员。

(c) **成立**：实现边界/超时处理确有耦合；实际 GPU 故障尚未本机复现。

(d) 最小修法：把“读取指定 PID 的显存并加回 free”放进 `gpu_mem.py` 的公开探针（复用 device index 与 subprocess timeout，沿用 probe 未测的现有语义）；sidecar client 提供公开的存活 PID 查询，不暴露 `_proc`。新增 `gpu_mem` 的可选 PID 能力单消费者仍必要：它是隔离 `nvidia-smi` 与 GPU 私有状态的唯一边界；另需单测覆盖超时、缺 PID、设备选择。只影响 Qwen3 word_align，不需 FunASR parity；CUDA 环境做一次实际验证。

(e) **P2**：封装/可维护性和潜在挂起风险，当前无显存数据损坏证据。

### P2 — terms 错误信息借用“不支持的文件格式” marker

(a) `src/core/file_based_process_pool.py:617-625` 对代理字符 raise 的文字含“不支持的文件格式”；`src/core/task_manager.py:85-109` 将该 marker 分类为不可重试输入。内部日志 marker `WORKER-TERMS-INVALID` 本身不参与分类。

(b) Linux AST 探针执行当前 `classify_error`：

```text
classify_error=non_retryable_input retryable=False
user_message=参数错误: terms 必须是 UTF-8 可编码文本（不支持的文件格式）
```
分类语义正确，用户可见 message 却错误暗示文件格式不受支持。

(c) **成立**（可重试分类是正确的，问题是误导性文本）。

(d) 最小修法：将错误文案改为准确的 terms UTF-8 编码错误，并在 `_NON_RETRYABLE_MARKERS` 加入专属 terms marker（或改成共享的类型化输入异常），两处保持一对一；补 classifier 测试。涉及 pool、task_manager；池供 FunASR/Qwen3 共用，建议跑 FunASR parity 和 pool 单测。

(e) **P2**：用户排障误导；无数据丢失/崩溃。

### P3 — encoder provider 测试重复 mock fixture

(a) 同名 fixture 同时在 `tests/unit/conftest.py:44-52` 与 `tests/unit/test_config_qwen3_asr_encoder_provider.py:16-25` 定义。模块内 fixture 覆盖同名 conftest fixture；共享版本还显式 import schema。

(b) `cx references --name mock_vendor_llama_native_module` 定位到 conftest 定义、测试模块定义及该模块四个 fixture 使用点；该测试模块的 fixture 局部遮蔽共享 fixture。没有额外运行测试（当前 pytest 收集被缺少 `loguru` 阻断）。

(c) **成立**：测试层重复实现；局部版本少了共享版本最后的 schema import，合并时需先确认共享 fixture 的初始化行为覆盖现有测试。

(d) 最小修法：删除测试模块本地 fixture，使用 conftest fixture；保留现有四个测试调用，确保 conftest 版本的 schema import 在 Linux/Mac 都正确。只涉及 tests，不需 FunASR parity。

(e) **P2**：测试维护重复，无生产行为影响。

### P4 — upload 与 create_task 的 engine 校验重复

(a) `src/api/websocket_handler.py:409-424` 的 `_upload_engine_error` 与 `src/core/task_manager.py:187-196` 都 trim request.engine、和默认引擎比较并生成相同错误文字。

(b) 静态对照两处分支：相同的空值回退/不匹配拒绝语义和错误 message；websocket 层需要在缓存短路及分片准入前提前拒绝，`create_task` 还必须保留最终防线。

(c) **成立**：判定逻辑重复；保留两处执行时机是必要的。

(d) 最小修法：提取纯共享函数，输入 requested/default engine，返回有效 engine 或同一错误文字；两个消费者共用，create_task 仍负责强制拒绝，websocket 仍负责早拒绝。第二消费者明确存在，值得轻量提取；不增加请求级路由/config。涉及 `websocket_handler.py`、`task_manager.py` 及共享 helper/单测；按项目约定跑 FunASR parity。

(e) **P2**：规则漂移的维护风险，目前未观察到行为不一致。

## 汇总

| 条目 | 结论 | 严重度 | 核心发现 |
|---|---|---:|---|
| R1 | 成立 | P2 | Qwen context 无 token/音频联合预算；具体 abort 未本机验证 |
| R2 | 成立 | P1 | 共享上传路径 `wb` 原地截断，真实读者探针读到 0 字节 |
| R3 | 成立 | P1 | 成功 result 先回收 lease，回收异常会掩盖成功 |
| R4 | 成立 | P2 | `[]` 变 `ValueError` → `ENGINE_ERROR` → 可重试 |
| R5 | 无法判定 | P2 风险 | 公式不含 word_align 增量，Mac 实际超时尚无同 workload 证据 |
| L1 | 成立 | P2 | 若输入有小数毫秒则截断；部分 legacy 路径无调用者 |
| L2 | 成立 | P2 | 本地 `datetime.now()` 与 SQLite UTC 比较偏移时区小时数 |
| L3 | 不成立 | P2 旁述 | 不存在键有 membership guard；cache gauge 键名不匹配导致省略 |
| L4 | 成立 | P2 | 两个限定模块内函数无 cx 可见调用者 |
| P1 | 成立 | P2 | Qwen3 显存路径直调 subprocess、访问两个私有成员 |
| P2 | 成立 | P2 | 非 UTF-8 terms 误报“不支持的文件格式”，分类结果本身正确 |
| P3 | 成立 | P2 | 测试模块重复定义 conftest fixture |
| P4 | 成立 | P2 | 两个必要准入点重复相同 engine 判定 |
