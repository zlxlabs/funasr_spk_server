"""
Qwen3-Diarize 工作进程入口

PR3: Qwen3 引擎走 multi-process worker pool, 跟 FunASR 同一套架构.
- 每个 worker subprocess 独立加载 Qwen3DiarizeTranscriber (libllama + Metal + sherpa 全部 per-worker)
- 主进程不再 import Qwen3DiarizeTranscriber, libllama context 自然 per-worker 隔离
- worker 内单线程跑任务, 不需要 asyncio.Lock

协议与 FunASR worker_process.py 对齐:
- argparse: --worker-id / --task-dir
- 启动后写 ready 文件 (worker_{id}.ready)
- 轮询任务文件 (worker_{id}_*.task)
- pickle 写结果 (worker_{id}_*.pkl)
- 单任务后退出 (sys.exit(0)), 由主进程的 health monitor 重启
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

# 添加项目根目录到 Python 路径 (worker 进程独立运行时需要)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 抑制配置打印 (与 FunASR worker 一致)
os.environ["FUNASR_WORKER_MODE"] = "1"

from src.core.qwen3_transcriber import get_qwen3_transcriber  # noqa: E402
from src.core.atomic_result_publish import publish_pickle_result, publish_text_marker  # noqa: E402


def load_qwen3_transcriber():
    """加载 Qwen3 转录器 (worker 进程内单例, 配置从 config.qwen3 读)"""
    return get_qwen3_transcriber()


def _rewrite_file_name(result, original_basename: str, output_format: str):
    """把 transcribe 返回值中的 file_name 改写为原始上传文件名

    pool 派发把 audio file 复制到 task_dir/{uuid}.wav, 因此 worker 拿到的
    audio_path basename 是 UUID. 这里改写回原始 basename, 保持主进程
    收到的语义与 FunASR 路径一致.
    """
    if output_format == "srt":
        # SRT 模式返回 dict
        if isinstance(result, dict) and "file_name" in result:
            result["file_name"] = original_basename
        return result

    # JSON 模式返回 (TranscriptionResult, raw)
    if isinstance(result, tuple) and len(result) == 2:
        tres, raw = result
        if hasattr(tres, "file_name"):
            tres.file_name = original_basename
    return result


def process_task(worker_id: int, transcriber, task_file: str, task_dir: str) -> None:
    """处理单个任务并写 pickle 结果"""
    # 读任务
    with open(task_file, "r", encoding="utf-8") as f:
        task = json.load(f)

    task_id = task["task_id"]
    audio_path = task["audio_path"]
    output_format = task.get("output_format", "json")
    # per-request 选项: 新协议是嵌套 options dict (pool 端 model_dump 序列化);
    # 老任务文件 (无 options, 平铺 language) 兜底构造, diarize 走默认 True.
    from src.models.schemas import TranscribeOptions

    options_dict = task.get("options")
    if options_dict is None:
        options = TranscribeOptions(language=task.get("language"))
    else:
        options = TranscribeOptions(**options_dict)

    result_file = task_file.replace(".task", ".pkl")
    original_audio_path = task.get("source_audio_path", audio_path)
    original_basename = os.path.basename(original_audio_path)

    print(f"[Qwen3-Worker-{os.getpid()}] 处理任务 {task_id}: {original_basename} (format={output_format})")

    # Qwen3 vendor 内部 sherpa diarize 用 libsndfile + librosa fallback (qwen3/diarize.py
    # _load_audio_mono_16k) 读 audio. wav/flac/ogg 走 libsndfile, mp3/opus 走 librosa fallback,
    # 都不需要 ffmpeg 预转码. 只有 libsndfile + librosa 都读不了的格式 (m4a/aac/mp4/mov/webm) 才必须 ffmpeg.
    # 历史教训: cd578a8 把 mp3 一并 ffmpeg 转 wav, 改变 audio 字节 → 触发 sherpa FastClustering
    # over-detect (60min-2spk → 4 spk). 见 docs/开发/archive/spk-over-detect-归因调研结果.md.
    SHERPA_SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".opus"}
    converted_wav_path = None
    actual_audio_path = audio_path
    audio_ext = os.path.splitext(audio_path)[1].lower()
    if audio_ext not in SHERPA_SUPPORTED_EXTENSIONS:
        try:
            from src.utils.file_utils import convert_to_wav
            converted_wav_path = audio_path.rsplit(".", 1)[0] + ".converted.wav"
            convert_to_wav(audio_path, output_path=converted_wav_path)
            actual_audio_path = converted_wav_path
            print(f"[Qwen3-Worker-{os.getpid()}] audio 转换 wav 成功: {os.path.basename(audio_path)} → {os.path.basename(converted_wav_path)}")
        except Exception as conv_err:
            print(f"[Qwen3-Worker-{os.getpid()}] audio 转换失败 (将尝试直接读): {conv_err}")
            converted_wav_path = None

    try:
        # transcribe 是 async, 在 worker 内起独立 event loop
        result = asyncio.run(
            transcriber.transcribe(
                audio_path=actual_audio_path,
                task_id=task_id,
                progress_callback=None,
                output_format=output_format,
                options=options,
            )
        )

        # 把 file_name 改写为用户上传的原始文件名
        # (pool 派发把 audio 复制到 task_dir/{uuid}.wav, worker 拿到的 basename 是 UUID, 不是原始名)
        result = _rewrite_file_name(result, original_basename, output_format)

        result_data = {
            "task_id": task_id,
            "success": True,
            "result": result,
            "worker_pid": os.getpid(),
        }

        publish_pickle_result(result_file, result_data)

        print(f"[Qwen3-Worker-{os.getpid()}] 任务 {task_id} 完成, 结果写入 {os.path.basename(result_file)}")

    except Exception as exc:
        error_msg = str(exc)
        traceback_str = traceback.format_exc()
        print(f"[Qwen3-Worker-{os.getpid()}] 任务 {task_id} 失败: {error_msg}")
        print(traceback_str)

        error_data = {
            "task_id": task_id,
            "success": False,
            "error": error_msg,
            "traceback": traceback_str,
            "worker_pid": os.getpid(),
            "audio_path": original_audio_path,
        }
        publish_pickle_result(result_file, error_data)

    finally:
        # 删除 .task 文件 (与 FunASR worker 一致, 表示任务已处理)
        try:
            os.remove(task_file)
        except Exception:
            pass
        # 清理 ffmpeg 转换出的临时 wav (如果有)
        if converted_wav_path:
            try:
                os.remove(converted_wav_path)
            except Exception:
                pass


def worker_loop(worker_id: int, task_dir: str) -> None:
    """worker 主循环: 加载模型 → 写 ready → 轮询任务 → 单任务后退出"""
    print(f"[Qwen3-Worker-{worker_id}] ========== 启动 (PID: {os.getpid()}) ==========")
    print(f"[Qwen3-Worker-{worker_id}] 任务目录: {task_dir}")

    # 加载 Qwen3 转录器 (首次会触发 libllama + Metal + sherpa 模型加载)
    print(f"[Qwen3-Worker-{worker_id}] 加载 Qwen3 转录器...")
    transcriber = load_qwen3_transcriber()
    # 提前触发引擎构建(避免第一个任务的 latency 异常)
    try:
        asyncio.run(transcriber.initialize())
        print(f"[Qwen3-Worker-{worker_id}] 引擎预热完成")
    except Exception as e:
        print(f"[Qwen3-Worker-{worker_id}] 引擎预热失败(将在首次 transcribe 时 lazy 加载): {e}")

    # 写 ready 文件
    ready_file = os.path.join(task_dir, f"worker_{worker_id}.ready")
    publish_text_marker(ready_file, str(os.getpid()))

    print(f"[Qwen3-Worker-{worker_id}] ========== 就绪, 等待任务 ==========")

    restart_requested = False
    while True:
        try:
            # 检查停止信号
            stop_file = os.path.join(task_dir, f"worker_{worker_id}.stop")
            if os.path.exists(stop_file):
                print(f"[Qwen3-Worker-{worker_id}] 收到停止信号")
                for f_path in (stop_file, ready_file):
                    try:
                        os.remove(f_path)
                    except Exception:
                        pass
                break

            # 查找分配给本 worker 的任务
            task_pattern = f"worker_{worker_id}_*.task"
            task_files = list(Path(task_dir).glob(task_pattern))

            if task_files:
                task_file = str(task_files[0])
                process_task(worker_id, transcriber, task_file, task_dir)
                print(f"[Qwen3-Worker-{worker_id}] 单任务完成, 准备退出以释放资源")
                try:
                    os.remove(ready_file)
                except Exception:
                    pass
                restart_requested = True
                break
            else:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print(f"[Qwen3-Worker-{worker_id}] 收到中断信号")
            break
        except Exception as e:
            print(f"[Qwen3-Worker-{worker_id}] 循环异常: {e}")
            traceback.print_exc()
            time.sleep(1)

    print(f"[Qwen3-Worker-{worker_id}] 退出")
    if restart_requested:
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-Diarize 工作进程")
    parser.add_argument("--worker-id", type=int, required=True, help="工作进程 ID")
    parser.add_argument("--task-dir", type=str, default="./temp/tasks", help="任务目录")
    args = parser.parse_args()

    worker_loop(args.worker_id, args.task_dir)
