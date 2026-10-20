"""
企微通知模块
"""
import httpx
import asyncio
from datetime import datetime
from typing import Dict, Any
from loguru import logger
from src.core.config import config
from src.models.schemas import TranscriptionTask
from src.utils.device_info import get_cached_device_identifier


async def send_wework_notification(task: TranscriptionTask, event_type: str):
    """发送企微机器人通知"""
    if not config.notification.enabled or not config.notification.webhook_url:
        return
    
    try:
        # 构建消息内容
        message = _build_message(task, event_type)
        
        # 发送请求
        async with httpx.AsyncClient() as client:
            for attempt in range(config.notification.retry_times):
                try:
                    response = await client.post(
                        config.notification.webhook_url,
                        json=message,
                        timeout=config.notification.timeout_seconds
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("errcode") == 0:
                            logger.debug(f"企微通知发送成功: {event_type}")
                            return
                        else:
                            logger.error(f"企微通知失败: {result}")
                    else:
                        logger.error(f"企微通知HTTP错误: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"企微通知请求失败 (尝试{attempt + 1}): {e}")
                    if attempt < config.notification.retry_times - 1:
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                    
    except Exception as e:
        logger.error(f"构建企微通知消息失败: {e}")


def _build_message(task: TranscriptionTask, event_type: str) -> Dict[str, Any]:
    """构建企微消息"""
    # 获取设备信息
    device_info = get_cached_device_identifier()
    
    if event_type == "completed":
        title = "✅ 转录任务完成"
        content = f"""设备: {device_info}
任务ID: {task.task_id}
文件名: {task.file_name}
文件大小: {task.file_size / 1024 / 1024:.2f}MB
处理时长: {task.result.processing_time:.2f}秒
音频时长: {task.result.duration:.2f}秒
说话人数: {len(task.result.speakers)}
片段数量: {len(task.result.segments)}
完成时间: {task.completed_at.strftime("%Y-%m-%d %H:%M:%S")}"""
    elif event_type == "failed":
        title = "❌ 转录任务失败"
        content = f"""设备: {device_info}
任务ID: {task.task_id}
文件名: {task.file_name}
错误信息: {task.error}
重试次数: {task.retry_count}
失败时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
    else:
        title = f"📢 转录任务通知 - {event_type}"
        content = f"""设备: {device_info}
任务ID: {task.task_id}
文件名: {task.file_name}
状态: {task.status.value}
时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
    
    # 企微text消息格式
    message = {
        "msgtype": "text",
        "text": {
            "content": f"{title}\n\n{content}"
        }
    }
    
    return message


async def send_custom_notification(title: str, content: str, msg_type: str = "text") -> bool:
    """发送自定义通知。

    Returns:
        True: 成功送达 (HTTP 200 且 errcode==0)。
        False: 通知未启用 / 无 webhook / 发送失败。失败仍记日志。
    现有调用方 (main.py) 忽略返回值, 行为不变。
    """
    if not config.notification.enabled or not config.notification.webhook_url:
        return False
    
    try:
        # 获取设备信息
        device_info = get_cached_device_identifier()
        
        # 强制使用text模式，添加设备信息
        message = {
            "msgtype": "text",
            "text": {
                "content": f"{title}\n\n设备: {device_info}\n{content}"
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                config.notification.webhook_url,
                json=message,
                timeout=config.notification.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("errcode") == 0:
                    logger.debug("自定义通知发送成功")
                    return True
                logger.error(f"自定义通知失败: {result}")
                return False
            logger.error(f"自定义通知HTTP错误: {response.status_code}")
            return False
                
    except Exception as e:
        logger.error(f"发送自定义通知失败: {e}")
        return False