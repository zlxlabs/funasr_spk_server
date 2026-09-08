"""
FunASR转录服务器主程序
"""
import asyncio
import signal
import sys
import time
from pathlib import Path
import websockets
from loguru import logger

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config
# 初始化日志系统(必须在 config 加载之后)
from src.utils.logger import setup_logger
setup_logger()

from src.core.database import db_manager
from src.core.task_manager import task_manager
# PR3: 引擎选择通过 dispatch 完成, 主进程不再硬编码 FunASR
from src.core.transcriber_dispatch import resolve_transcriber
logger.info(f"使用 ASR 引擎: {config.transcription.default_engine}")
from src.api.websocket_handler import ws_handler
from src.api.http_endpoints import HttpEndpoints
from src.utils.notification import send_custom_notification
from src.utils.platform_utils import log_platform_info, check_system_requirements


class FunASRServer:
    """FunASR转录服务器"""
    
    def __init__(self):
        self.server = None
        self.transcriber = None
        self.is_running = False
    
    async def start(self):
        """启动服务器"""
        try:
            logger.info("正在启动FunASR转录服务器...")
            
            # 初始化数据库
            await db_manager.init_db()
            
            # 初始化转录器 (PR3: 根据 default_engine 选择, FunASR / Qwen3 走同一套 dispatch)
            engine_name = config.transcription.default_engine
            logger.info(f"初始化 {engine_name} 引擎...")
            transcriber = resolve_transcriber(None)  # None → 走 default_engine
            self.transcriber = transcriber
            if hasattr(transcriber, "initialize"):
                await transcriber.initialize()
            logger.success(f"{engine_name} 引擎初始化完成")
            
            # 启动任务管理器
            await task_manager.start()

            # 可观测性仪表盘 (P1): /health + /metrics 走 websockets 同端口 process_request 钩子
            # (零新依赖, 不多开端口). started_at 给 uptime 指标.
            observability_endpoints = HttpEndpoints(
                task_manager=task_manager,
                db_manager=db_manager,
                config=config,
                started_at=time.monotonic(),
            )

            # 启动WebSocket服务器（优化大文件传输配置）
            self.server = await websockets.serve(
                ws_handler.handle_connection,
                config.server.host,
                config.server.port,
                # 同端口 HTTP 端点: GET /health|/metrics 返 HTTP, 其它走 ws 升级
                process_request=observability_endpoints.process_request,
                max_size=config.server.max_file_size_mb * 1024 * 1024,
                max_queue=config.server.max_connections,
                ping_interval=60,   # 增加到60秒发送一次 ping（避免大文件传输时超时）
                ping_timeout=120,   # ping 响应超时120秒（给客户端更多处理时间）
                close_timeout=60,   # 关闭连接超时60秒
                # 增加读写缓冲区大小以支持大文件传输
                read_limit=2**20,   # 1MB读缓冲
                write_limit=2**20,  # 1MB写缓冲
                compression=None    # 禁用压缩以减少CPU负载
            )
            
            self.is_running = True
            
            logger.success(f"服务器已启动: ws://{config.server.host}:{config.server.port}")
            
            # 发送启动通知
            await send_custom_notification(
                "🚀 FunASR转录服务器已启动",
                f"地址: ws://{config.server.host}:{config.server.port}\n"
                f"最大并发: {config.transcription.max_concurrent_tasks}\n"
                f"认证: {'启用' if config.auth.enabled else '禁用'}",
                "markdown"
            )
            
            # 定期清理任务
            asyncio.create_task(self._periodic_cleanup())
            
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            raise
    
    async def stop(self):
        """停止服务器"""
        logger.info("正在停止服务器...")
        
        self.is_running = False

        # 先停止 WebSocket 新连接，避免任务管理器停止期间继续接纳任务。
        if self.server:
            self.server.close()
        
        # 停止任务管理器
        await task_manager.stop()

        # 任务协程已停止后，回收引擎 wrapper 持有的外部 worker
        if self.transcriber is not None and hasattr(self.transcriber, "cleanup"):
            await self.transcriber.cleanup()
        
        # 等待 WebSocket 连接完成关闭，确保池和任务资源已经回收。
        if self.server:
            await self.server.wait_closed()
        
        # 发送停止通知
        await send_custom_notification(
            "🛑 FunASR转录服务器已停止",
            "服务器已正常关闭",
            "text"
        )
        
        logger.info("服务器已停止")
    
    async def _periodic_cleanup(self):
        """定期清理任务"""
        while self.is_running:
            try:
                # 清理过期缓存
                await db_manager.clean_old_cache()
                
                # 清理临时文件
                from src.utils.file_utils import cleanup_temp_files
                await cleanup_temp_files()
                
                # 获取统计信息
                stats = task_manager.get_stats()
                cache_stats = await db_manager.get_cache_stats()
                
                logger.info(f"系统状态 - 任务: {stats}, 缓存: {cache_stats}")
                
            except Exception as e:
                logger.error(f"定期清理失败: {e}")
            
            # 每小时执行一次
            await asyncio.sleep(3600)


async def main():
    """主函数"""
    # 记录平台信息和检查系统要求
    log_platform_info()
    check_system_requirements()
    
    # 输出性能相关配置
    logger.info("========== 性能相关配置 ==========")
    logger.info(f"FunASR配置:")
    logger.info(f"  - CPU核心数 (ncpu): {getattr(config.funasr, 'ncpu', 8)}")
    logger.info(f"  - 批处理大小 (batch_size_s): {config.funasr.batch_size_s}秒")
    logger.info(f"  - 设备类型 (device): {config.funasr.device}")
    logger.info(f"转录配置:")
    logger.info(f"  - 最大并发任务数: {config.transcription.max_concurrent_tasks}")
    logger.info(f"  - 任务超时时间: {config.transcription.task_timeout_minutes}分钟")
    logger.info(f"服务器配置:")
    logger.info(f"  - 最大连接数: {config.server.max_connections}")
    logger.info(f"  - 最大文件大小: {config.server.max_file_size_mb}MB")
    logger.info("=====================================")
    
    server = FunASRServer()
    shutdown_event = asyncio.Event()
    
    # 设置信号处理
    def signal_handler(sig, frame):
        logger.info("收到终止信号，正在关闭服务器...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 启动服务器
        await server.start()
        
        # 等待关闭信号
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"服务器错误: {e}")
    finally:
        await server.stop()


if __name__ == "__main__":
    # Windows平台特殊处理
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行主程序
    asyncio.run(main())
