"""
文件处理工具模块
"""
import math
import os
import re
import time
import hashlib
import aiofiles
import ffmpeg
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
from src.core.config import config


async def calculate_file_hash(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    
    async with aiofiles.open(file_path, 'rb') as f:
        while True:
            chunk = await f.read(8192)
            if not chunk:
                break
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str) -> bool:
    """检查文件是否允许上传"""
    ext = get_file_extension(filename)
    return ext in config.server.allowed_extensions


async def save_uploaded_file(file_data: bytes, filename: str) -> Tuple[str, str]:
    """保存上传的文件"""
    # 确保上传目录存在
    Path(config.server.upload_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成唯一文件名
    file_hash = hashlib.md5(file_data).hexdigest()
    ext = get_file_extension(filename)
    safe_filename = f"{file_hash}{ext}"
    file_path = str(Path(config.server.upload_dir) / safe_filename)
    
    # 保存文件
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file_data)
    
    logger.debug(f"文件已保存: {file_path}")
    return file_path, file_hash


async def delete_file(file_path: str) -> bool:
    """删除文件。返回是否真正删除了文件(文件不存在 / 删除失败 → False)。

    返回 bool 让调用方(如孤儿 sweeper)能诚实统计实际回收数(codex 二审)。
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"文件已删除: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"删除文件失败 {file_path}: {e}")
        return False


def convert_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
    """将音视频文件转换为WAV格式"""
    if output_path is None:
        output_path = str(Path(config.server.temp_dir) / f"{Path(input_path).stem}.wav")
    
    # 确保临时目录存在
    Path(config.server.temp_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用ffmpeg转换为16kHz单声道WAV
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ar=16000, ac=1, format='wav')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.debug(f"音频转换完成: {input_path} -> {output_path}")
        return output_path
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"音频转换失败: {error_msg}")
        raise Exception(f"音频转换失败: {error_msg}")


def get_audio_duration(file_path: str) -> float:
    """通过容器、音频流元数据或 ffmpeg 解码统计获取音频时长。"""
    def fail(stage: str, cause: Exception) -> None:
        """记录有限长度的失败原因，并以明确异常向上报告。"""
        detail = getattr(cause, "stderr", None)
        if isinstance(detail, bytes):
            reason = detail.decode(errors="replace").strip()
        elif detail:
            reason = str(detail).strip()
        else:
            reason = str(cause).strip()
        reason = reason[-500:] or type(cause).__name__
        message = f"音频时长探测失败（{stage}）：{reason}"
        logger.error(
            "AUDIO-DURATION-PROBE-FAILED path={} stage={} reason={}",
            file_path,
            stage,
            reason,
        )
        raise RuntimeError(message) from cause

    def parse_duration(value) -> Optional[float]:
        """解析有效的非负时长；无效元数据交由下一探测来源处理。"""
        if value is None:
            return None
        try:
            duration = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(duration) or duration < 0:
            return None
        return duration

    try:
        probe = ffmpeg.probe(file_path)
    except Exception as exc:
        fail("ffprobe", exc)

    format_info = probe.get("format") or {}
    duration = parse_duration(format_info.get("duration"))
    if duration is not None:
        return duration

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            duration = parse_duration(stream.get("duration"))
            if duration is not None:
                return duration

    try:
        _, stderr = (
            ffmpeg.input(file_path)
            .output("-", format="null")
            .run(capture_stdout=True, capture_stderr=True)
        )
    except Exception as exc:
        fail("ffmpeg 解码", exc)

    stderr_text = (
        stderr.decode(errors="replace") if isinstance(stderr, bytes) else str(stderr)
    )
    timestamps = re.findall(
        r"\btime=((?:\d+):(?:\d{2}):(?:\d{2}(?:\.\d+)?))", stderr_text
    )
    if not timestamps:
        fail("ffmpeg 解码", ValueError("解码输出中没有最终 time 时间戳"))

    hours, minutes, seconds = timestamps[-1].split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def validate_file_size(file_size: int) -> bool:
    """验证文件大小"""
    max_size = config.server.max_file_size_mb * 1024 * 1024
    return file_size <= max_size


async def cleanup_temp_files():
    """清理临时文件"""
    temp_dir = Path(config.server.temp_dir)
    if not temp_dir.exists():
        return
    
    try:
        current_time = time.time()
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                # 删除超过1天的临时文件
                file_mtime = Path.stat(file_path).st_mtime
                if (current_time - file_mtime) > 86400:
                    file_path.unlink()
                    logger.debug(f"清理临时文件: {file_path}")
    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")