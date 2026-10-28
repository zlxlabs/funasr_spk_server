# Grok bot VM 单进程真实转录短测 — 进度存档

Task: dlg-20261028-051216-dbf286 | Base: 43390e5 | 分支: card/grok-vm-smoke-261028
执行器: 本卡 delegate（implementer 角色，亲自落盘）

## 阶段 0 — pickup 与现场核查（13:12–13:14）

- Dispatch/报告路径/worktree 核对一致；git status 干净，HEAD = 43390e5（=卡面 Base）。
- fixture `tests/fixtures/audio/podcast_2speakers_predownload` 见 benchmark.md（SHA256=e304d0c1…8104，1,920,104 bytes）。
- 老脚本 `bench_run.py` 已按要求通读，确认两处不能照抄的坑：
  1. raw 解析段 `except Exception` 吞错降级为 parse_note——新脚本一律让异常带 traceback 上抛；
  2. 字段名 `*_kb` 实为 bytes（VmRSS kB×1024）——新脚本字段一律 `_bytes` 后缀。
- funasr 1.2.7 源码核实（只读老 venv）：`get_or_download_model_dir` 调 `snapshot_download`
  **不传 cache_dir**，只认 `MODELSCOPE_CACHE` 环境变量（modelscope/hub/utils/utils.py:41），
  下载与推理两侧必须对齐该变量，不能信 AutoModel(cache_dir=…)。
- transcribe 返回元组 `(TranscriptionResult, raw)`（funasr_transcriber.py:397，json+speaker 路径）；
  DB 缓存写在 task_manager 层，transcribe 自身无缓存捷径。

## 阶段 0.5 — 目标机不可达（阻塞中）

- 目标机 100.75.27.5（zlx-server-cloud-grokbot-1）在 Tailscale 注册但 **offline**：
  13:14 首查时 last seen 18m ago（即 ≥12:56 起离线，早于本卡派发）。
- 有界轮询：13:13 起 20s 间隔 SSH 探测，至 13:40 共 41 次全部 connect timeout
  （日志：~/.local/state/benchmarks/funasr-grok-vm-smoke-20261028/logs/ssh-poll.log）。
- 决策点定为 14:00（本地时间）：恢复则立即执行准备→下载→推理；未恢复则本卡
  outcome=failed（目标机不可达），保留全部诊断。

## 阶段 0.7 — 本地准备已完成（可随时上传执行）

- `src-snapshot.tar.gz`（git archive 43390e5，仅 src/config.json/requirements.txt，2,173,343 bytes，
  SHA256=f84dda06f4b03e981025bf80cbbb9e8f0ef02cc0d31958a22164ce78045efeeb，已验证无 .env/secret）。
- 五个一次性脚本（AST/bash -n 全过；验证入口已本地正反冒烟：真实样本 VALID_OK、
  删 spk 样本 ASSERT_FAIL exit 2、全 spk=0 样本 VALID_OK）：
  - `remote_prep.sh` — 私有 uv+Py3.12+锚定依赖（torch CPU index 帮助输出留证）+快照解包校验
  - `download_models.py` — 模型名/revision 读快照 config，funasr 自带映射下载，MODELSCOPE_CACHE 对齐
  - `monitor.py` — 父控制器：0.5s 采样进程树 RSS+MemAvailable，512MiB 止损（TERM→5s→KILL），
    180s 硬界限，退出码 83/84；`--selftest` 用已知小子进程做判据正反检查
  - `bench_infer.py` — 单进程：env 预置→lock/ncpu 显式→model_pool is None 断言→initialize 计时→
    四子模型设备全 cpu 断言→无 Qwen3 模块断言→两轮 transcribe 计时→canonical 非空断言→raw/反验材料落盘
  - `validate_output.py` — 唯一验证入口（正验/反验同入口）
- 本地证据目录：~/.local/state/benchmarks/funasr-grok-vm-smoke-20261028/{logs,results,scripts,evidence}

### 关键决策与已否决方案

- 否决「下载脚本自己写 repo 映射」：改用 funasr 1.2.7 的 `name_maps_ms` + 快照 config 同源读参，零漂移。
- 否决「AutoModel 顺便下载」：AutoModel 实例化会把四模型载入内存（旧实测峰 ~3.9GiB），
  在 MemAvailable 仅 3.1GiB 的共享机上冒 OOM 险；预下载走纯文件 IO。
- 父控制器选独立 monitor 进程（非线程）：内存止损 killpg 需要从模型进程外部打信号，
  进程内采样线程无法执行止损，也不能二次加载模型。

## 下一步唯一动作

等待目标机恢复至 14:00 决策点；恢复 → 上传五脚本+快照+fixture，按 prep(20min 硬界限)→
download→selftest→run(180s 硬界限)→validate→fetch 顺序执行；未恢复 → outcome=failed。
