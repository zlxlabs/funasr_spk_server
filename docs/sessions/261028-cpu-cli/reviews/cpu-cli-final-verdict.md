<!-- delegate-outcome: succeeded -->
# CPU CLI 最终独立审查

结论：PASS，不阻断交付。固定范围内未发现真实使用方式下的数据丢失、静默错误或崩溃缺陷；P1：0，其他有效 finding：0。

failure-visibility: clean

- 风险：personal；Task-Id：funasr_spk_server-20261028-06。
- Dispatch-Id：dlg-20261028-083851-101cfa。
- Base：b57873fe348247c3746cdc96b16e5116e4f1a465。
- H0：09f52a6db513864e511d209bfe4b2013e84de804；审查开始时 HEAD 与 H0 一致，工作区干净。
- 规格：`docs/CLI.md`。
- 方法：独立新会话读取全部产品、测试、依赖和使用文档 diff，并核对共享内核调用上下文；实现进度报告按输入隔离要求不读取、不作为依据。未运行测试、模型、安装或环境探针，未修产品。

规格核对：

| 不变式 | 代码与已有证据 | 裁决 |
|---|---|---|
| 传统完整模型 ID，CPU、lock、各模型两线程 | `src/cli.py:27` 注入完整配置；`funasr_transcriber.py:111` 透传三个子模型 kwargs；`test_json_real_entry` 断言实际模型入参；真实运行数据四模型均 cpu/ncpu=2 | 满足 |
| 不导入服务配置、不建模型池 | 内核仅无配置注入时延迟导入；文件工具仅服务函数延迟导入配置；CLI 走原文件时长、哈希和转录路径；真实 `service_modules=[]`、`model_pool_is_none=true` | 满足 |
| 无参服务构造保持 | 默认分支仍将同一全局配置转换为原有字典，模型参数默认值保持；原有 FunASR golden、服务及分片回归八项通过 | 满足 |
| stdout 单一 JSON，stderr 日志，错误非零 | `main` 将导入、加载和推理 stdout 重定向至 stderr，成功后才序列化；入口测试穿过 shell 和共享投影，模型异常测试检查空 stdout 与非零退出；原始实模 stdout 可直接 JSON 解码 | 满足 |
| 参数与文件先失败，仓外 cwd、空格路径 | 参数解析和 `is_file` 先于模型导入；shell 所有路径带引号且不改变调用者 cwd；`test_json_real_entry` 使用空格 clone、相对空格文件名；真实入口早失败记录一致 | 满足 |
| 精简 CPU 安装、缓存独立、预取加载同路径 | 12 行安装脚本定向 `.venv-cpu`，CPU wheel 源，随后同入口预取；依赖未引入服务器或 Qwen 专用包；安装 producer 测试捕获实际 argv/缓存环境，覆盖默认与空格缓存目录 | 满足 |
| 复用结果投影，保留 spk=0 | CLI 直接复用内核 `TranscriptionResult.model_dump`；原有句级投影将 spk=0 映射到 Speaker1；入口断言秒级时间戳、文本与 Speaker1，真实模型给出有效片段 | 满足 |
| 最小实现，无新增容错层或过度抽象 | CLI 69 行、包装 5 行、安装 12 行；配置注入服务与 CLI 两条现有路径，无新增工厂、状态层、重试或吞错分支 | 满足 |

原始证据指针：派发dlg-20261028-081117-35b156的本机evidence归档。

- 独立解析 `full-warm.xml`：43 项，19 passed、8 skipped、3 failure、13 error；全套不是全绿。
- 直接核对 passed 节点：FunASR 三项 golden（单人、静音、双人）、服务 JSON/SRT/cache 三项、chunked 两项，共八项回归；另有 CLI 实模一项通过。
- 继承红：与 `failures-base.xml` 的 16 个非 FunASR 红项逐节点核对，failure/error 类型及失败阶段相同；13 条 message 完全一致，其余三项均为 Qwen 服务 setup 的 60 秒启动失败，日志时间及上下文不同。新红：在该记录覆盖范围内未发现。未扩查 Qwen 依赖；远端主干/CI 基线不可用，以上为已有本地基线对照。
- `unit-cli.xml`：5 passed。`red-tip-ncpu.result.json` 记录独立临时树、真实命令和退出码 1，日志为 ncpu 断言失败；`red-base.log` 是缺少新入口的 setup FileNotFoundError，不冒充功能断言红验。
- `core-runtime.json`：源码 root 指向本审查树，torch/torchaudio 2.8.0+cpu、FunASR 1.2.7；torch 两线程、lock、无模型池、四模型 CPU/ncpu=2、无服务模块。
- `entry-real.stdout`：4209 字节，独立 JSON 解码成功；60 秒、31 片段、Speaker1/Speaker2，文本非空且全部片段时间戳有效。`entry-summary.json` 记录真实入口退出 0、单次总耗时 32.87 秒；help 退出 0，缺文件退出 1，无参/非法参数/互斥组合退出 2，错误 stdout 均为空。
- `golden-unchanged.json` 三项 SHA-256 与仓内实际 golden 字节逐项一致，固定 diff 未改 golden。
- 未使用较早的 `full-tip.xml`；不存在的 `funasr-required.xml` 不作为证据。

问题清单：无阻断项，无需产品修复。OCR 未取得有效 envelope：初筛未完成，非 clean；本结论来自独立代码审查及上述原始证据，不代表 OCR 扫描通过。记录范围外环境及未覆盖输入不作通过承诺。
