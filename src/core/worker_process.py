"""
FunASR工作进程 - 支持pickle序列化处理大型结果
集成设备管理器，支持 MPS 加速
"""
import os
import sys
import json
import pickle
import time
import argparse
import traceback
from pathlib import Path
# 添加项目根目录到 Python 路径（worker 进程独立运行时需要）
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置环境变量以抑制配置打印（在导入config之前）
os.environ['FUNASR_WORKER_MODE'] = '1'

# 平台相关优化需要在导入 torch 之前完成，确保 MPS 高水位阈值等环境变量生效
from src.utils.platform_utils import setup_platform_specific_env
from src.utils.torch_utils import release_accelerator_memory

setup_platform_specific_env()

from funasr import AutoModel

# 导入设备管理器和全局配置
from src.core.device_manager import DeviceManager
from src.core.config import config as global_config


def setup_device() -> str:
    """
    设置设备并应用必要的补丁

    Returns:
        选定的设备名称
    """
    print(f"[Worker] 检测和配置设备...")

    # 将全局配置转换为字典用于设备管理器
    config_dict = global_config.model_dump()

    # 选择设备
    device = DeviceManager.select_device(config_dict)
    print(f"[Worker] 选定设备: {device}")

    # 应用设备补丁
    DeviceManager.apply_patches(device)

    # 记录设备信息（简化版，不使用 logger）
    device_info = DeviceManager.get_device_info(device)
    print(f"[Worker] 设备信息: {device_info['device_name']}")

    return device


def initialize_model(device: str):
    """
    初始化FunASR模型

    Args:
        device: 设备名称（已选定）
    """
    # 使用全局配置
    funasr_config = global_config.funasr
    cache_dir = funasr_config.model_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Worker] 初始化模型（设备: {device}）...")
    model = AutoModel(
        model=funasr_config.model,
        model_revision=funasr_config.model_revision,
        vad_model=funasr_config.vad_model,
        vad_model_revision=funasr_config.vad_model_revision,
        punc_model=funasr_config.punc_model,
        punc_model_revision=funasr_config.punc_model_revision,
        spk_model=funasr_config.spk_model,
        spk_model_revision=funasr_config.spk_model_revision,
        cache_dir=cache_dir,
        ncpu=funasr_config.ncpu,
        device=device,  # 使用已选定的设备
        disable_update=funasr_config.disable_update,
        disable_pbar=funasr_config.disable_pbar
    )
    print(f"[Worker] 模型初始化完成")
    return model


def process_task(
    worker_id: int,
    model,
    task_file: str,
    task_dir: str,
) -> None:
    """处理单个任务并在结束后退出"""
    result = None
    result_data = None

    # 读取任务
    with open(task_file, 'r', encoding='utf-8') as f:
        task = json.load(f)

    task_id = task['task_id']
    audio_path = task['audio_path']
    local_audio_path = Path(audio_path)
    batch_size_s = task.get('batch_size_s', global_config.funasr.batch_size_s)
    hotword = task.get('hotword', '')
    use_pickle = task.get('use_pickle', True)  # 默认使用pickle
    original_audio_path = task.get('source_audio_path', audio_path)

    print(f"[Worker-{os.getpid()}] 处理任务 {task_id}: {os.path.basename(original_audio_path)}")

    # 结果文件路径
    if use_pickle:
        result_file = task_file.replace('.task', '.pkl')
    else:
        result_file = task_file.replace('.task', '.result')

    try:
        # ========== 诊断日志：任务开始 ==========
        print(f"[Worker-{os.getpid()}] [诊断] 任务参数:")
        print(f"  - 音频路径: {audio_path}")
        print(f"  - 文件存在: {os.path.exists(audio_path)}")
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"  - 文件大小: {file_size / 1024 / 1024:.2f} MB")
        print(f"  - batch_size_s: {batch_size_s}")
        print(f"  - hotword_length={len(hotword)}")

        # ========== 诊断日志：检查设备状态 ==========
        import torch
        print(f"[Worker-{os.getpid()}] [诊断] 设备状态:")
        print(f"  - MPS 可用: {torch.backends.mps.is_available()}")
        print(f"  - MPS 已构建: {torch.backends.mps.is_built()}")

        # 检查模型设备
        if hasattr(model, 'device'):
            print(f"  - 模型设备: {model.device}")

        # ========== 执行模型推理 ==========
        print(f"[Worker-{os.getpid()}] [诊断] 开始调用 model.generate()...")
        start_time = time.time()

        result = model.generate(
            input=audio_path,
            batch_size_s=batch_size_s,
            hotword=hotword
        )

        elapsed = time.time() - start_time
        print(f"[Worker-{os.getpid()}] [诊断] model.generate() 完成，耗时: {elapsed:.2f}秒")

        # ========== 诊断日志：检查结果 ==========
        print(f"[Worker-{os.getpid()}] [诊断] 结果检查:")
        print(f"  - 结果类型: {type(result)}")
        if isinstance(result, list):
            print(f"  - 结果长度: {len(result)}")
            if len(result) > 0:
                print(f"  - 首个元素类型: {type(result[0])}")
                if isinstance(result[0], dict):
                    print(f"  - 首个元素键: {list(result[0].keys())}")

        # 保存结果
        result_data = {
            'task_id': task_id,
            'success': True,
            'result': result,
            'worker_pid': os.getpid()
        }

        print(f"[Worker-{os.getpid()}] [诊断] 保存结果到: {os.path.basename(result_file)}")

        if use_pickle:
            # 使用pickle保存（支持大型对象）
            with open(result_file, 'wb') as f:
                pickle.dump(result_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # 使用JSON保存（兼容性更好，但可能失败于大型结果）
            try:
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, separators=(',', ':'))
            except Exception as json_error:
                print(f"[Worker-{os.getpid()}] JSON保存失败，改用pickle: {json_error}")
                # 降级到pickle
                result_file = task_file.replace('.task', '.pkl')
                with open(result_file, 'wb') as f:
                    pickle.dump(result_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[Worker-{os.getpid()}] ✓ 任务 {task_id} 完成")

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()

        # ========== 诊断日志：错误详情 ==========
        print(f"[Worker-{os.getpid()}] ✗ 任务 {task_id} 失败!")
        print(f"[Worker-{os.getpid()}] [诊断] 错误详情:")
        print(f"  - 错误类型: {type(e).__name__}")
        print(f"  - 错误信息: {error_msg}")
        print(f"  - 错误堆栈:\n{traceback_str}")

        # 检查是否是 MPS 相关错误
        if "dimension" in error_msg.lower() or "tensor" in error_msg.lower():
            print(f"[Worker-{os.getpid()}] [诊断] 疑似 MPS 张量错误!")
            print(f"[Worker-{os.getpid()}] [诊断] 当前 MPS 状态:")
            import torch
            print(f"  - MPS 可用: {torch.backends.mps.is_available()}")
            try:
                test_tensor = torch.randn(2, 2, device='mps')
                print(f"  - MPS 测试张量创建: 成功")
                del test_tensor
            except Exception as mps_e:
                print(f"  - MPS 测试张量创建: 失败 - {mps_e}")

        # 保存错误结果
        error_data = {
            'task_id': task_id,
            'success': False,
            'error': error_msg,
            'traceback': traceback_str,
            'worker_pid': os.getpid(),
            'audio_path': original_audio_path,
            'file_size': local_audio_path.stat().st_size if local_audio_path.exists() else 0
        }

        if use_pickle:
            with open(result_file, 'wb') as f:
                pickle.dump(error_data, f)
        else:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False)

    finally:
        # 每次任务结束后主动释放加速设备缓存，避免内存占用持续增长
        release_accelerator_memory(tag=f"Worker-{os.getpid()}", log_fn=print)

        # 显式移除可能仍在引用的重对象，辅助 GC
        result = None
        result_data = None

        # 删除任务文件（表示任务已处理）
        try:
            os.remove(task_file)
        except:
            pass

        try:
            if local_audio_path.exists():
                local_audio_path.unlink()
        except Exception as cleanup_error:
            print(f"[Worker-{os.getpid()}] [诊断] 删除本地音频副本失败: {cleanup_error}")


def worker_loop(worker_id: int, task_dir: str):
    """工作进程主循环"""
    print(f"[Worker-{worker_id}] ========== 启动 (PID: {os.getpid()}) ==========")

    # ========== 诊断日志：环境信息 ==========
    print(f"[Worker-{worker_id}] [诊断] 环境信息:")
    print(f"  - Python 版本: {sys.version}")
    print(f"  - 工作目录: {os.getcwd()}")
    print(f"  - 任务目录: {task_dir}")

    import torch
    print(f"[Worker-{worker_id}] [诊断] PyTorch 信息:")
    print(f"  - 版本: {torch.__version__}")
    print(f"  - MPS 可用: {torch.backends.mps.is_available()}")
    print(f"  - MPS 已构建: {torch.backends.mps.is_built()}")

    # 检查环境变量
    mps_ratio = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'not set')
    omp_threads = os.environ.get('OMP_NUM_THREADS', 'not set')
    print(f"  - PYTORCH_MPS_HIGH_WATERMARK_RATIO: {mps_ratio}")
    print(f"  - OMP_NUM_THREADS: {omp_threads}")

    # 设置设备并应用补丁
    print(f"[Worker-{worker_id}] [诊断] 开始设备配置...")
    device = setup_device()
    print(f"[Worker-{worker_id}] [诊断] 设备配置完成: {device}")

    is_mps_device = str(device).lower().startswith("mps")
    if is_mps_device:
        print(f"[Worker-{worker_id}] [诊断] MPS 模式下任务完成后将自动退出以触发进程重启")
    else:
        print(f"[Worker-{worker_id}] [诊断] CPU 模式任务完成后同样将自动退出以触发进程重启")

    # 初始化模型
    print(f"[Worker-{worker_id}] [诊断] 开始模型初始化...")
    model = initialize_model(device)
    print(f"[Worker-{worker_id}] [诊断] 模型初始化完成")

    # 创建就绪标记文件
    ready_file = os.path.join(task_dir, f"worker_{worker_id}.ready")
    with open(ready_file, 'w') as f:
        f.write(str(os.getpid()))

    print(f"[Worker-{worker_id}] ========== 就绪，等待任务 ==========")
    
    # 监听任务
    restart_requested = False
    while True:
        try:
            # 检查停止信号
            stop_file = os.path.join(task_dir, f"worker_{worker_id}.stop")
            if os.path.exists(stop_file):
                print(f"[Worker-{worker_id}] 收到停止信号")
                try:
                    os.remove(stop_file)
                    os.remove(ready_file)
                except:
                    pass
                break
            
            # 查找分配给此工作进程的任务
            task_pattern = f"worker_{worker_id}_*.task"
            task_files = list(Path(task_dir).glob(task_pattern))
            
            if task_files:
                # 处理第一个任务
                task_file = str(task_files[0])
                process_task(
                    worker_id,
                    model,
                    task_file,
                    task_dir,
                )
                print(f"[Worker-{worker_id}] [诊断] 单个任务处理完成，准备退出以释放资源")
                try:
                    os.remove(ready_file)
                except:
                    pass
                restart_requested = True
                break
            else:
                # 没有任务，短暂休眠
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"[Worker-{worker_id}] 收到中断信号")
            break
        except Exception as e:
            print(f"[Worker-{worker_id}] 循环异常: {e}")
            traceback.print_exc()
            time.sleep(1)
    
    print(f"[Worker-{worker_id}] 退出")
    if restart_requested:
        print(f"[Worker-{worker_id}] [诊断] 因任务完成退出，等待主进程重启")
        sys.exit(0)


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="FunASR工作进程")
    parser.add_argument("--worker-id", type=int, required=True, help="工作进程ID")
    parser.add_argument("--task-dir", type=str, default="./temp/tasks", help="任务目录")

    args = parser.parse_args()

    # 运行工作进程（使用全局配置，不再需要传递 config）
    worker_loop(args.worker_id, args.task_dir)
