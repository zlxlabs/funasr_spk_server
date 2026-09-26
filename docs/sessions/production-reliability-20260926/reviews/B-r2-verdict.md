# B-r2 运行时证据审查 [codex]

- 冻结范围：`ec0b892035ff1def7e1168254b5b89622c5e7bae..e602f796a8b4d51c8681ede567f8b9422b52f40f`。
- 风险等级：internal。结论：本次连接诊断与恢复测试未发现新增 P1。
failure-visibility: clean

本轮新证据是 B-r1 独立静态审查后取得的 Mac 组合候选运行结果。候选同时含 A `4e49d35` 与 B `e602f79`；主脑独立计算 git 提交与 Mac 文件 SHA256：FunASR parser `1255f8b49b645a30902faea799a8145daac889c8ca9f60620c59d1a0d00a3ac7`，WebSocket handler `12fb756c6b2714e3ac893e5bfdeb3598487c8e512c47f6d814e87356caa3f112`，两端一致。

1. 主脑白名单回读 Mac `/tmp/funasr-result-contract-4e49d35/logs/B-targeted-unit.log`：真实 socket 恢复与日志诊断两个文件 `17 passed, 1 warning in 0.58s`。Linux 包含既有 batch/终态通知的四文件定向验证为 40 passed。
2. 重新核对源码：恢复用例读真实传输帧，先查到 processing，fake task 完成后在同一重连 socket 查询原 task_id 的 JSON/SRT 内容，并断言无重复 create/submit、无 cancel。日志测试用真实 formatter 输出核对 UUID 关联、异常类别/关闭码、敏感标记不回显以及 ConnectionClosed 只在外层分类。
3. Mac 完整 integration 中 FunASR 真服务子进程的 JSON、SRT、缓存命中三项及三项模型 parity 均通过，提供组合路径的补充运行证据。此套件整体未全绿，Qwen 环境失败及额外探针问题见最终验证报告，不隐去失败。
4. Linux 全量 unit 的 11 个失败/错误节点，主脑从 base 与 A H0 新保存日志提取后，与 B 报告节点集合逐项比较，三者完全一致。这是已测失败子集的基线证据，不将未通过的全套改写成全绿。

边界：断线恢复用例的 TaskManager 与模型仍为 fake，真服务 E2E 用例没有断线步骤，所以二者不能拼接成“真实生产 worker 断线持续执行已经验证”。没有修改外部客户端、部署生产或调整超时/重试。仓库没有远端 CI workflow；本审查不构成绕过 CI 或生产发布的授权。
