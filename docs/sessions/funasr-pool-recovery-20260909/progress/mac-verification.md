# Mac 候选验收

- 候选 HEAD：`54cd1f69cc7e0f04d884a171a4ac31514c5b7f1f`；server PID `39513`，池大小 3，初始 worker 为 `39560/39561/39562`。
- 真实 WebSocket 长入口使用 `temp/samples/4-person-example.m4a`，`force_refresh=true`、`cached=false`；task `1e85ca9e-ce34-4db5-95f7-1c59e50ddcc3`，池任务 `9b409105-4dee-4d8b-950b-87dd8459885b`，slot 0/PID `39560`，2216.16 秒音频耗时 139.697 秒，202 段、4 位说话人，结果 68930 bytes。
- 运行中短 WebSocket 任务 task `0206a512-7e1a-436e-b422-422bff62193c` 使用 slot 1/PID `39561`；`Popen` suffix 为 `funasr-worker-39513-1-5e16f1bd`，三个 Darwin 私有根运行时均为 `0700`。02:04:45 捕获的 MPSGraph 路径含 PID `39561`；长任务自身没有 MPSGraph 路径捕获，该目录证据属于短任务，不冒充长任务 PID。
- server 日志记录长任务完成、slot 0 自动补为 PID `42671`，随后短任务后 slot 1 补为 PID `46077`。关闭时 server、初始/补充 worker 均退出；`temp/tasks/run-dbb250a958f2406d9e41147629904396`、`candidate3-temp` 和私有 worker 根均无残留。
- 共享 `/var/folders/.../T/com.apple.MetalPerformanceShadersGraph` 的哨兵仍存在，前后样本均 64 bytes；修正后的 `shared-mps-comparison.json` 可解析，原非法尾字节产物保留为 `.invalid.json`。
- 候选 integration：FunASR semantic parity 3 项通过，smoke engine wiring 6 项通过；全量为 `5 failed, 18 passed, 6 skipped, 13 errors`。Qwen3 失败/错误均受生产 venv 缺 `sherpa_onnx` 或 `gguf` 限制。
- 在相同 Mac 消费环境、相同 venv/model 下对基线 `43e8e949d8a860cdc9ab0b58e50a20f279d90861` 只跑上述 18 个失败 nodeid，结果 `5 failed, 13 errors`，nodeid 与导入错误完全一致；parity 失败由基线复现，不能判作候选回归。
- 三个 tracked golden SHA256 与基线一致（完整值见 JSON）：`podcast=6f441d1aed54d782e441d03d1e6e78ff949d3ef49e75de79157e1e2ade538ba2`、`silence=1d8a2f46499508c64472035b2da14588968bd40abd0e1bc093d17fa7b58e7ba2`、`tts=64896ed50b9b06e0b00844e8b6eb2b00be7bffaf9bcba9055299f1ad97c763fc`；本次未生成新 golden。
- R1 文档已由 `69e9021` 导入；账本最后 `p1=0` 记录来源为 `219832cbef460b74fa3606df5e839bd674b9a4c0`。证据明细见同目录 `evidence/mac-pool3.json`。
残余限制：未安装 Qwen3 依赖，因此真实 diarize/word-align/ORT/Qwen3 pool 行为仍待具备依赖的 Mac 环境验证。
