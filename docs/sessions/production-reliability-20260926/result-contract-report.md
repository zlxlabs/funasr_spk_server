# A 结果解析边界：实现与验证记录

- Task-Id：`funasr_spk_server-20260926-A`；任务卡：`/home/zlx/.local/state/delegate/cards/funasr-reliability-0926/A-result-contract.md`。
- 业务冻结提交：`4e49d35070b49f678ea9b4692d7a39c37c49da40`；基线：`ec0b892035ff1def7e1168254b5b89622c5e7bae`；PR #12 保持 draft。
- 风险档按 internal 处理；仓库没有 `risk-tier` 声明，需主脑提醒补声明。本报告不修改全局规则。

## 结果契约

- JSON 与新鲜 SRT 共用 FunASR 句子结构校验。未知/损坏顶层结构，以及非空 `text` 配缺失或空 `sentence_info`，在两种 `transcribe` 出口都显式抛解析异常；不会返回成功 payload 或进入新鲜成功缓存写入。
- 合法空输出仍为空成功：明确空 `text` 可缺少 `sentence_info` 或带显式空数组；`text` 缺失时仅显式 `sentence_info: []` 兼容。`{}` 属于未知结构并失败；非空 `text` 配空/缺失句子也失败。实现不猜时间戳、不 fallback、不增加 retry。
- 错误与诊断日志只包含 task id 和客观结构状态，不记录识别正文、文件名或 URL。异常沿 TaskManager 既有错误分类和重试策略处理，可能重试后才进入失败终态。
- 新校验只覆盖新鲜模型结果。历史 SRT cache hit 不经过此校验；本轮不改写缓存、数据库或 schema。

## 验证证据

- **基线红验**：scratch 树从基线导入新增测试文件，JSON 与 SRT 两个实际 `transcribe` 出口均报 `DID NOT RAISE Exception`，`2 failed`。日志：`/tmp/funasr-result-contract-base-red.log`。H0 同主题定向单测为 `15 passed`。
- **Linux 全量 unit**：`1 failed, 1111 passed, 6 skipped, 10 errors`（45.10 秒）。只对其中 3 个 Qwen encoder/provider 测试文件在基线与 H0 各复跑一次；两边都是 `1 failed, 9 passed, 10 errors`，11 个 FAILED/ERROR nodeid 完全相同，异常类型计数同为 `OSError: 7, ImportError: 4`。日志：`/tmp/funasr-A-qwen-unit-base-ec0b892-20260926.log`、`/tmp/funasr-A-qwen-unit-h0-4e49d35-20260926.log`；这是存量环境问题对照，不代表全量基线重跑。
- **Mac A-only FunASR 实测**：隔离候选中 3 条既有 golden parity 加 raw-shape probe 为 `4 passed, 5 warnings in 11.84s`。静音文件 raw 顶层是 list，首项 dict，`text` 存在且为空，`sentence_info` 缺失；属于本轮明确保留的合法模型空输出形态。前后 golden 哈希一致。
- **Mac A+B 组合验证**：B 冻结源码 `e602f796a8b4d51c8681ede567f8b9422b52f40f` 仅应用到候选，不回写 A 分支。原仓 42 个 integration 节点为 `4 failed, 18 passed, 7 skipped, 13 errors`；另有 1 个临时 probe 失败，因此候选总计 `5 failed, 18 passed, 7 skipped, 13 errors`（248.13 秒）。B 两个定向单测在 Mac 为 `17 passed`。
- **Qwen integration 基线对照**：从原仓组合结果中抽出的 17 个 Qwen FAILED/ERROR nodeid，在基线 Mac 候选全部同状态；基线摘要 `4 failed, 13 errors in 123.53s`。对照日志：`/tmp/funasr-result-contract-4e49d35/logs/base-qwen-integration.log`；组合日志：`/tmp/funasr-result-contract-4e49d35/logs/combined-integration.log`。未安装依赖、下载模型或修无关 Qwen 路径。
- **候选额外 probe 的单独失败**：全套运行顺序中，先前 integration 用例把全局默认引擎改为 `qwen3`，probe 随后请求 `funasr`，报 `ValueError: Server configured with engine='qwen3', cannot accept engine='funasr'.`。该候选临时 probe 在 A-only 运行通过；它不属于原仓 42 项，未据此改产品代码或重跑全套。
- **设备与并发边界**：同候选 Python、配置和 FunASR `DeviceManager` 解析得到 `mps`；FunASR parity fixture 强制 `lock` 模式。未验证生产 3-worker pool 并发，不能把真实模型 parity 解读为生产并发验收。
- **代码与 golden 指纹**：候选 `funasr_transcriber.py` SHA256 `1255f8b49b645a30902faea799a8145daac889c8ca9f60620c59d1a0d00a3ac7`；组合候选 `websocket_handler.py` SHA256 `12fb756c6b2714e3ac893e5bfdeb3598487c8e512c47f6d814e87356caa3f112`。三条 golden 前后 SHA256 一致：`tts_1speaker_5s` `64896ed50b9b06e0b00844e8b6eb2b00be7bffaf9bcba9055299f1ad97c763fc`、`silence_5s` `1d8a2f46499508c64472035b2da14588968bd40abd0e1bc093d17fa7b58e7ba2`、`podcast_2speakers_60s` `6f441d1aed54d782e441d03d1e6e78ff949d3ef49e75de79157e1e2ade538ba2`。
- 候选位于 Mac `/tmp/funasr-result-contract-4e49d35`；通知关闭，模型只使用从既有缓存复制的 FunASR 工件，网络限 localhost 测试。一个既有 Qwen WebSocket integration fixture 将子进程 `TMPDIR` 写死为系统 `/tmp`，因此该用例可能在系统临时目录产生临时文件；base 与候选采用相同入口。未复制 `.env`、未触碰生产源码/DB/原模型缓存、未启动或重启生产服务。

## 剩余事项

- OCR 提出的历史 SRT cache-hit 校验遗漏是 P2 残余。本轮只读样本（2026-09-26 12:05:46+08）545 条：540 条非空 text 中缺失/空 `sentence_info` 均为 0；其余 5 条空 segments 均有明确空字符串 `text`。样本未发现已知坏历史记录，但不证明历史缓存永久无问题；因此未扩大到 database/cache 迁移。
- 仓库没有 GitHub Actions workflow，PR checks 为空；全套 unit 与 integration 并非全绿，Qwen 失败仅凭基线对照归为存量环境问题。PR 保持 draft，待主脑完成最终审查与后续交付判定。
