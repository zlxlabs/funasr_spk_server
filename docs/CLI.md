# 单文件 CPU 转录

无需启动服务器。每次命令在单进程内加载一套传统 Paraformer + VAD + 标点 + CAM++，CPU 两线程；退出即释放模型，不使用数据库或进程池，也不读取服务配置。

## 安装

前置：已有 `uv`、`ffmpeg`、`ffprobe`（安装脚本不使用 sudo，不安装系统软件）。在自己的 clone 中执行：

```bash
bash scripts/install_cpu_cli.sh
```

脚本用 uv 和 Python 3.12 创建本 clone 的 `.venv-cpu`，安装 CPU torch/torchaudio 2.8.0、FunASR 1.2.7 及精简依赖，然后预取并加载同一套模型。首次需要联网下载；任一步失败即停止，不复用服务器 venv。

## 转录

从任意当前目录调用（将路径替换为自己的稳定 clone 路径）：

```bash
/path/to/repo/scripts/funasr-transcribe "input.wav" > result.json
```

相对输入路径以调用者当前目录为准；包含空格时加引号。stdout 只有一个 UTF-8 JSON 文档；模型下载、加载、推理日志均写 stderr。参数、文件、模型或转录失败返回非零，不输出假成功 JSON。

JSON 直接使用现有内核结果：`segments` 含 `text`、秒单位 `start_time` / `end_time`、`speaker`（例如 `Speaker1`），`speakers` 列出说话人；另有文件名、哈希、时长、处理时间和任务标识。静音可以正常返回空片段。标签数量不是准确率保证。

## 缓存、重建和成本

模型缓存尊重非空 `MODELSCOPE_CACHE`；未指定时固定为 `~/.cache/funasr-cli/modelscope`，与虚拟环境分离。可在安装与转录前设置同一 `MODELSCOPE_CACHE` 指向其他用户可写目录。

重建时删除自己 clone 的 `.venv-cpu` 后重新执行安装命令即可，模型缓存可保留；需要重新下载时，仅清理自己指定的模型缓存目录。不要删除共享缓存或其他人的环境。

建议至少预留 8 GiB 可用内存和数 GiB 磁盘。每次运行都有模型冷加载成本（通常数十秒，依机器及缓存而变），并非常驻服务。这里只提供单文件 JSON，不提供批处理、SRT、GPU 或全局命令注册。
