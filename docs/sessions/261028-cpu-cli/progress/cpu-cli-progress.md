# CPU 单文件 CLI 进度

## 第一阶段：入口与精简安装配方
- 红：新增真实包装脚本入口测试，基线 1 failed / 3 errors（两个脚本均不存在）。证据 `/tmp/cpu-cli-red.log`。
- 绿：4 个入口/安装边界测试通过；主仓测试解释器在本工作树跑全套，1255 passed / 39 skipped（71.40 秒）。
- 入口只注入一份 CPU/lock/2 线程配置，沿用内核返回二元组及 Pydantic JSON；服务无参构造仍读原配置。file_utils 仅将服务配置导入挪至需要它的函数内，避免 CLI 建服务目录。
- 否决：服务配置覆盖、池、第二套结果 schema、第二套模型清单、重试和常驻进程。
- 独立 `.venv-cpu` 已安装真实 PyPI FunASR 1.2.7 / CPU torch 2.8.0，首次模型预取后台进行中；不能以单测绿代替真实 CPU 验收。
- 下一步：完成预取，运行真实 60 秒音频与共享路径集成测试，保留实际输出，再发布 draft PR。

## 第二阶段：真实模型与约束力
- 独立 `.venv-cpu` 安装与预取成功；真实入口 60 秒 fixture 集成测试 1 passed（30.49 秒），31 个片段、2 个说话人。实际字节保留在 `.venv-cpu/cli-evidence/result.json` 与 `stderr.log`。
- stdout 隔离故障注入：仅将重定向目标改为 stdout，入口测试因 JSONDecodeError 转红；已逐行还原，再跑绿。证据 `/tmp/cpu-cli-mutation-red.log`。
- 安装边界增加未设置缓存 / 指定带空格缓存两个环境方向，当前入口测试 5 passed。
- 按要求尝试完整真实环境 integration：精简环境缺 pytest-asyncio 且服务配置要求通知凭据；补测试插件并关闭测试通知后，收集仍因缺 aiosqlite 中断。在基线 b57873fe 的仓外快照用同一环境复现相同收集错误，属于继承环境红，非 CLI 新红；不为此安装禁止的服务器依赖，也不宣称完整 parity 已通过。
- 下一步：最终全套单测、无会话环境探针、发布分支和 draft PR。

## 第三阶段：最终验收
- 最终无并行模型负载/无注入的完整测试：1256 passed / 39 skipped（84.27 秒），入口专项 5 passed；shell syntax / diff check 通过。
- 安装版本故障注入将 torch argv 改为 2.8.1，两个缓存方向均因实际 argv 不符转红；逐行还原后 5 passed。证据 `/tmp/cpu-cli-installer-mutation-red.log`。
- 曾把故障注入与全套测试重叠运行，导致该轮安装用例也读到了注入版本，另有已有池调度计时断言在并行模型负载下失败；保留该轮失败日志，不把它当实现新红。最终单独完整运行全绿。
- `env -i HOME=... PATH=/usr/bin:/bin` 裸 shell 的实际 60 秒转录仍输出 31 片段、2 标签；保存 `bare-result.json` / `bare-stderr.log`。所有执行也位于本卡 delegate systemd unit 环境，仓库无远程 pytest CI，不宣称有 CI 真模型验证。
- 当前核心功能与四条 Accept-Check 均通过；完整真实集成套件的基线环境阻塞仍明确保留。下一步由 Pi 独立审查 draft PR、标 ready 后跑完整 gate；执行器不标 ready、不合并。
