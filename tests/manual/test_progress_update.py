#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新的进度更新逻辑
"""

import asyncio
import json
import websockets
import time
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_progress_update():
    """测试进度更新功能"""
    server_url = "ws://localhost:8767"
    
    # 准备测试音频文件路径（相对本目录推导，不写死某台机器的绝对路径）
    audio_file = str(Path(__file__).parent.parent / "test_data" / "test_audio_60s.wav")
    
    if not Path(audio_file).exists():
        logger.error(f"测试音频文件不存在: {audio_file}")
        logger.info("请确保有一个测试音频文件用于测试")
        return
    
    try:
        async with websockets.connect(server_url) as websocket:
            logger.info("已连接到服务器")
            
            # 读取音频文件
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # 准备请求
            request = {
                "type": "transcribe",
                "audio_data": audio_data.hex(),  # 转换为十六进制字符串
                "format": "wav",
                "options": {
                    "language": "zh",
                    "enable_timestamps": True,
                    "enable_speaker": False
                }
            }
            
            # 发送请求
            logger.info("发送转录请求...")
            await websocket.send(json.dumps(request))
            
            # 接收响应和进度更新
            start_time = time.time()
            progress_updates = []
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=300)
                    data = json.loads(response)
                    
                    if data.get("type") == "progress":
                        progress = data.get("progress", 0)
                        elapsed = time.time() - start_time
                        progress_updates.append({
                            "progress": progress,
                            "elapsed_time": elapsed
                        })
                        logger.info(f"进度更新: {progress}% (已用时: {elapsed:.1f}秒)")
                    
                    elif data.get("type") == "result":
                        total_time = time.time() - start_time
                        logger.success(f"转录完成，总用时: {total_time:.1f}秒")
                        
                        # 显示进度更新统计
                        if progress_updates:
                            logger.info("\n进度更新统计:")
                            logger.info(f"更新次数: {len(progress_updates)}")
                            for update in progress_updates:
                                logger.info(f"  {update['elapsed_time']:.1f}秒: {update['progress']}%")
                        
                        # 显示转录结果摘要
                        if "result" in data:
                            text = data["result"].get("text", "")
                            logger.info(f"转录文本长度: {len(text)} 字符")
                            if len(text) > 100:
                                logger.info(f"转录文本摘要: {text[:100]}...")
                        break
                    
                    elif data.get("type") == "error":
                        logger.error(f"转录错误: {data.get('message')}")
                        break
                        
                except asyncio.TimeoutError:
                    logger.error("接收响应超时")
                    break
                    
    except Exception as e:
        logger.error(f"测试失败: {e}")


async def main():
    """主函数"""
    logger.info("开始测试进度更新逻辑")
    logger.info("配置说明: transcription_speed_ratio=10 表示转录时间约为音频时长的1/10")
    
    await test_progress_update()
    
    logger.info("测试完成")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")
    
    # 运行测试
    asyncio.run(main())