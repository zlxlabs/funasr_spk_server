# A-r2 运行时证据审查 [codex]

- 冻结范围：`ec0b892035ff1def7e1168254b5b89622c5e7bae..4e49d35070b49f678ea9b4692d7a39c37c49da40`。
- 风险等级：internal；判定：未发现本次结果校验改动的新 P1。
failure-visibility: clean

本轮新证据是独立基线红验和 Mac 候选的真实模型输出，区别于 A-r1 的静态源码审查。

1. 主脑回读 `/tmp/funasr-result-contract-base-red.log`：运行根为 ec0b892 的 scratch 树，新测试的 JSON、SRT 两个 `transcribe` 出口用例均报 `DID NOT RAISE Exception`，2 failed。候选的同文件定向测试 15 passed；断言能区分旧版的空成功与新版显式失败。
2. 主脑从 Mac 候选 `/tmp/funasr-result-contract-4e49d35/logs/A-parity.log` 独立白名单提取到 `4 passed, 5 warnings in 11.84s`；对应三份既有 FunASR golden 和一次静音原始结构探针。执行器核对 golden 前后哈希未变。实际静音输出为 list，首项 text 存在且为空、sentence_info 缺失，与新增允许空结果的分支一致。
3. 重新核对 JSON 与新鲜 SRT 均经过 `_validated_funasr_sentences`；非空 text 配缺失/空 sentence_info 不返回成功 payload。TaskManager 的新鲜成功缓存写入发生于 transcribe 返回后；错误仍沿既有错误分类和重试，不承诺即时终态。
4. OCR 的历史 SRT cache-hit 候选不属于新鲜模型校验作用域；同日生产只读计数中 540 条非空 text 缓存没有缺失/空 sentence_info，5 条空 segments 均为空 text。此为当时样本事实，不证明历史缓存永久无问题，不更改既有缓存。

证据边界：真实模型测试按既有 fixture 走 lock 模式，不能替代生产进程池并发验收；未运行生产重启、任务提交或通知。Linux 全量 unit 的 Qwen vendor 平台错误在基线复现，未把全套写成全绿。此结论不涵盖模型识别准确率，也不证明生产两段空输出音频实际无语音。组合 integration 的最终计数另见实现验证报告。
