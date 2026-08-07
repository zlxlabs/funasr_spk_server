"""
WebSocket处理器
"""
import json
import asyncio
from typing import Dict, Set, Optional
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from loguru import logger
from src.models.schemas import (
    WebSocketMessage,
    FileUploadRequest,
    TaskStatusResponse,
    TaskStatusBatchItem,
    TaskStatusBatchResponse,
    TaskStatus,
    ErrorResponse,
)
from src.core.config import config
from src.utils.auth import verify_token
import base64
import hashlib
import tempfile
import os
import time
import uuid
from pydantic import ValidationError
from src.core.capabilities import build_asr_capabilities
from src.core.runtime import detect_runtime


# 批量状态查询的 task_ids 硬上限（控帧大小）。超出截断 + warn。
# 长音频 JSON 50 份单帧可能数 MB，靠"client 把已完成 id 移出轮询集、每 result 只发一次"摊平。
# 50 是起步硬上限，实测帧过大再降。
TASK_STATUS_BATCH_MAX = 50


def _invalid_terms_reason(validation_error: ValidationError) -> Optional[str]:
    """从 Pydantic errors() 精确提取 invalid_terms 的 reason。"""
    for detail in validation_error.errors():
        if detail.get("type") != "invalid_terms":
            continue
        reason = (detail.get("ctx") or {}).get("reason")
        if reason:
            return str(reason)
    return None


class WebSocketHandler:
    """WebSocket连接处理器"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.task_connections: Dict[str, Set[str]] = {}  # task_id -> connection_ids
        self.upload_sessions: Dict[str, dict] = {}  # 分片上传会话管理
        
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """处理WebSocket连接"""
        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}_{datetime.now().timestamp()}"
        
        try:
            # 认证
            if config.auth.enabled:
                auth_success = await self._authenticate(websocket)
                if not auth_success:
                    await self._send_error(websocket, "auth_failed", "认证失败")
                    return
            
            # 注册连接
            self.connections[connection_id] = websocket
            logger.info(f"WebSocket连接建立: {connection_id}")
            
            # 发送欢迎消息
            await self._send_message(websocket, "connected", {
                "connection_id": connection_id,
                "message": "连接成功",
                "server_time": datetime.now().isoformat(),
                "capabilities": self._build_capabilities(),
            })
            
            # 处理消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, connection_id, data)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "invalid_json", "无效的JSON格式")
                except Exception as e:
                    logger.error(f"处理消息失败: {e}")
                    await self._send_error(websocket, "message_error", str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket连接关闭: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket处理错误: {e}")
        finally:
            # 清理连接
            self._cleanup_connection(connection_id)

    def _build_capabilities(self) -> dict[str, object]:
        """Compose the same minimal capability contract advertised over HTTP."""
        engine = config.transcription.default_engine
        return build_asr_capabilities(
            schema_version=1,
            engine=engine,
            runtime=detect_runtime().name,
            features={"terms": engine in ("funasr", "qwen3")},
        )
    
    async def _authenticate(self, websocket: WebSocketServerProtocol) -> bool:
        """认证WebSocket连接"""
        try:
            # 等待认证消息
            await self._send_message(websocket, "auth_required", {
                "message": "请提供认证令牌"
            })
            
            # 设置超时
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            data = json.loads(auth_message)
            
            if data.get("type") != "auth":
                return False
            
            token = data.get("token")
            if not token:
                return False
            
            # 验证令牌
            from src.utils.auth import verify_token
            user_info = verify_token(token)
            if not user_info:
                return False
            
            await self._send_message(websocket, "auth_success", {
                "message": "认证成功",
                "user": user_info
            })
            return True
            
        except (asyncio.TimeoutError, json.JSONDecodeError):
            return False
        except Exception as e:
            logger.error(f"认证失败: {e}")
            return False
    
    async def _handle_message(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理接收到的消息"""
        msg_type = data.get("type")
        msg_data = data.get("data", {})
        
        if msg_type == "ping":
            # 心跳响应
            await self._send_message(websocket, "pong", {
                "timestamp": datetime.now().isoformat()
            })
            
        elif msg_type == "upload_request":
            # 文件上传请求
            await self._handle_upload_request(websocket, connection_id, msg_data)
            
        elif msg_type == "upload_data":
            # 文件数据上传（单文件模式）
            await self._handle_upload_data(websocket, connection_id, msg_data)
            
        elif msg_type == "upload_chunk":
            # 分片文件上传
            await self._handle_chunk_upload(websocket, connection_id, msg_data)

        elif msg_type == "finalize_upload":
            # 显式重试 finalize（高负载止血：queue_full 后客户端不重传大文件，仅重发此消息）
            if "terms" in msg_data:
                await self._send_error(
                    websocket,
                    "protocol_error",
                    "finalize_upload 不允许 terms 字段",
                )
                return
            task_id = msg_data.get("task_id")
            if task_id:
                await self._handle_finalize_upload(websocket, task_id)
            else:
                await self._send_error(websocket, "missing_task_id", "缺少task_id参数")


        elif msg_type == "task_status":
            # 查询任务状态
            task_id = msg_data.get("task_id")
            if task_id:
                await self._send_task_status(websocket, task_id)
            else:
                await self._send_error(websocket, "missing_task_id", "缺少task_id参数")

        elif msg_type == "task_status_batch":
            # 批量查询任务状态（异步轮询契约 TODOS #20）：一帧拿全集，完成项内联 result，
            # 防 per-task N+1 拉取风暴 + TTL race。
            await self._handle_task_status_batch(websocket, msg_data.get("task_ids"))

        elif msg_type == "cancel_task":
            # 取消任务
            task_id = msg_data.get("task_id")
            if task_id:
                await self._handle_cancel_task(websocket, task_id)
            else:
                await self._send_error(websocket, "missing_task_id", "缺少task_id参数")
                
        else:
            await self._send_error(websocket, "unknown_message_type", f"未知的消息类型: {msg_type}")
    
    async def _handle_upload_request(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理文件上传请求"""
        try:
            # 提取输出格式（默认为json）
            output_format = data.get("output_format", "json")
            if output_format not in ["json", "srt"]:
                await self._send_error(websocket, "invalid_format", 
                                     "不支持的输出格式，支持: json, srt")
                return
            
            # 验证请求数据
            request = FileUploadRequest(**data)
            
            # 检查文件大小
            from src.utils.file_utils import validate_file_size
            if not validate_file_size(request.file_size):
                await self._send_error(websocket, "file_too_large", 
                                     f"文件太大，最大支持{config.server.max_file_size_mb}MB")
                return
            
            # 检查文件类型
            from src.utils.file_utils import is_allowed_file
            if not is_allowed_file(request.file_name):
                await self._send_error(websocket, "invalid_file_type", 
                                     f"不支持的文件类型，支持: {', '.join(config.server.allowed_extensions)}")
                return
            
            # 检查是否为分片上传模式
            upload_mode = data.get("upload_mode", "single")
            
            if upload_mode == "chunked":
                # 处理分片上传请求
                await self._handle_chunked_upload_request(
                    websocket, connection_id, data, request=request
                )
                return
            
            # 创建任务（单文件模式）
            from src.core.task_manager import task_manager
            task = await task_manager.create_task(request)
            
            # 关联任务和连接
            if task.task_id not in self.task_connections:
                self.task_connections[task.task_id] = set()
            self.task_connections[task.task_id].add(connection_id)
            
            # 检查缓存（如果不强制刷新）
            # cache key 含 engine; word_align / diarize 折维收拢在 cache_params_for (D4)
            from src.core.database import cache_allowed_for
            if not request.force_refresh and cache_allowed_for(task.options):
                from src.core.database import db_manager, cache_params_for
                _ce, _allow = cache_params_for(task)
                cached_result = await db_manager.get_cached_result(
                    request.file_hash, output_format, engine=_ce, allow_cross_engine=_allow,
                    options=task.options,
                )
                if cached_result:
                    logger.info(f"使用缓存结果（upload_request阶段）: {task.task_id}")

                    # E2: 早返回出口同样组装 effective options 回显
                    # DRY(commit 1): projected 提取 + metadata 构建走共享纯函数 cache_hit_metadata
                    from src.core.result_projection import cache_hit_metadata
                    _md, _proj, _srt_ok = cache_hit_metadata(
                        cached_result, engine=task.engine, options=task.options,
                        output_format=output_format,
                    )

                    # 根据输出格式准备结果
                    if output_format == "srt":
                        if _srt_ok:
                            # cache_hit_metadata 用 get 不 pop, 此处排除 projected key 不泄漏给客户端
                            result_data = {k: v for k, v in cached_result.items() if k != "projected"}
                            result_data["metadata"] = _md
                        else:
                            # 如果缓存中没有SRT格式，跳过缓存
                            logger.info("缓存中没有SRT格式，继续处理")
                            result_data = None
                    else:
                        # JSON格式
                        cached_result.metadata = _md
                        result_data = cached_result.model_dump()
                    
                    if result_data:
                        # 直接返回缓存结果
                        await self._send_message(websocket, "task_complete", {
                            "task_id": task.task_id,
                            "result": result_data
                        })
                        return
            
            # 发送响应
            await self._send_message(websocket, "upload_ready", {
                "task_id": task.task_id,
                "message": "准备接收文件数据"
            })
            
        except ValidationError as e:
            reason = _invalid_terms_reason(e)
            if reason:
                await self._send_message(websocket, "error", {
                    "error": "invalid_terms",
                    "reason": reason,
                    "message": "术语列表校验失败",
                })
                return
            validation_summary = [(detail.get("type"), detail.get("loc")) for detail in e.errors()]
            logger.error("处理上传请求失败: validation_errors={}", validation_summary)
            await self._send_error(websocket, "upload_error", "上传请求参数无效")
        except Exception as e:
            logger.error(f"处理上传请求失败: {e}")
            await self._send_error(websocket, "upload_error", str(e))
    
    async def _handle_upload_data(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理文件数据上传"""
        try:
            task_id = data.get("task_id")
            if not task_id:
                await self._send_error(websocket, "missing_task_id", "缺少task_id参数")
                return
            
            # 获取任务
            from src.core.task_manager import task_manager
            task = task_manager.get_task(task_id)
            if not task:
                await self._send_error(websocket, "task_not_found", "任务不存在")
                return
            
            # 处理文件数据
            file_data_base64 = data.get("file_data")
            if not file_data_base64:
                await self._send_error(websocket, "missing_file_data", "缺少文件数据")
                return
            
            # 解码base64数据
            file_data = base64.b64decode(file_data_base64)
            
            # 验证文件大小
            if len(file_data) != task.file_size:
                await self._send_error(websocket, "size_mismatch", 
                                     f"文件大小不匹配: 期望{task.file_size}, 实际{len(file_data)}")
                return
            
            # 验证文件哈希
            actual_hash = hashlib.md5(file_data).hexdigest()
            if actual_hash != task.file_hash:
                await self._send_error(websocket, "hash_mismatch", "文件哈希不匹配")
                return
            
            # 保存文件
            from src.utils.file_utils import save_uploaded_file, delete_file
            file_path, _ = await save_uploaded_file(file_data, task.file_name)

            # 提交任务到队列，获取排队信息
            from src.core.task_manager import QueueFullError
            try:
                queue_info = await task_manager.submit_task(task_id, file_path)
            except QueueFullError as qe:
                # 准入控制拒绝：清理已落地文件（非分片无 session 可复用，整包重传），
                # self.tasks 回滚已在 task_manager 完成。客户端据 retry_after 退避重传。
                await delete_file(file_path)
                logger.warning(f"单文件任务入队被拒(queue_full): {task_id}, retry_after={qe.retry_after}s")
                await self._send_message(websocket, "queue_full", {
                    "task_id": task_id,
                    "retry_after": qe.retry_after,
                    "queue_size": qe.queue_size,
                    "max_queue_size": qe.max_queue_size,
                    "error": "queue_full",
                    "message": str(qe),
                })
                return

            # 发送响应，包含排队信息
            response_data = {
                "task_id": task_id,
                "message": "文件上传成功，开始处理"
            }

            # 如果启用了排队状态通知且任务在排队
            if config.transcription.queue_status_enabled and queue_info:
                if queue_info.get("queued", False):
                    response_data.update({
                        "queue_position": queue_info["position"],
                        "estimated_wait_seconds": queue_info["estimated_wait"],
                        "message": f"文件上传成功，排队位置: {queue_info['position']}"
                    })
                    # 发送排队状态
                    await self._send_message(websocket, "task_queued", response_data)
                    return

            await self._send_message(websocket, "upload_complete", response_data)

        except Exception as e:
            logger.error(f"处理文件数据失败: {e}")
            await self._send_error(websocket, "upload_data_error", str(e))
    
    async def _send_task_status(self, websocket: WebSocketServerProtocol, task_id: str):
        """发送任务状态"""
        try:
            from src.core.task_manager import task_manager
            task = task_manager.get_task(task_id)

            if not task:
                # 区分 expired(曾存在被内存清理) vs not_found(从未存在)，
                # 让客户端能据此走缓存兜底 / 报错（codex: size-cap 挤掉结果别返回含糊 not_found）
                if task_manager.was_evicted(task_id):
                    await self._send_error(
                        websocket, "task_expired",
                        "任务结果已过期清理，请用 file_hash 重新提交（命中缓存秒回）",
                    )
                else:
                    await self._send_error(websocket, "task_not_found", "任务不存在")
                return
            
            response = TaskStatusResponse(
                task_id=task.task_id,
                status=task.status,
                progress=task.progress,
                result=task.result,
                error=task.error
            )
            
            await self._send_message(websocket, "task_status", response.model_dump())
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            await self._send_error(websocket, "status_error", str(e))
    
    async def _handle_task_status_batch(self, websocket: WebSocketServerProtocol, task_ids):
        """批量查询任务状态（异步轮询契约 TODOS #20）。

        客户端批量场景必须用此单条消息（禁 per-task 各自轮询），完成项 result 随响应内联。
        - task_ids 上限 TASK_STATUS_BATCH_MAX（50）：超出截断 + warn，控帧大小。
        - 逐 id 复用 _build_task_status_batch_item 组装（DRY：result/SRT/expired 形态收拢一处）。
        - **逐 id 组装是同步循环、中间不 await**：避免撞 task_manager.py:499-500 注释的
          "COMPLETED 翻转在 result 组装后"窗口，否则会返回 status=completed 但 result=null。
        """
        if not isinstance(task_ids, list) or not task_ids:
            await self._send_error(websocket, "missing_task_ids", "缺少task_ids参数（应为非空列表）")
            return

        if len(task_ids) > TASK_STATUS_BATCH_MAX:
            logger.warning(
                f"task_status_batch 请求 {len(task_ids)} 个 id 超上限 "
                f"{TASK_STATUS_BATCH_MAX}，截断（控帧大小）"
            )
            task_ids = task_ids[:TASK_STATUS_BATCH_MAX]

        from src.core.task_manager import task_manager
        # 同步组装全部 item（无 await → 不可能跨协程切换 → 状态-结果读取原子）
        items = [self._build_task_status_batch_item(task_manager, tid) for tid in task_ids]

        response = TaskStatusBatchResponse(items=items)
        await self._send_message(websocket, "task_status_batch", response.model_dump())

    def _build_task_status_batch_item(self, task_manager, task_id: str) -> TaskStatusBatchItem:
        """组装单个 batch 项（**同步**，绝不能 await：钉 task_manager.py:499-500 原子性不变量）。

        - 不存在：was_evicted → task_expired（曾存在被清），否则 task_not_found（从未存在），
          status=None（不整批失败，client 据 error 凭 file_hash 重投）。
        - COMPLETED：JSON 内联 result；SRT 内联 srt_content（codex #11：result 装不下 SRT）。
        - failed/timed_out/cancelled：终态 + error（codex #10：client 据此停轮询）。
        - PENDING/PROCESSING：result/srt_content/error 全 None（小帧）。
        """
        task = task_manager.get_task(task_id)
        if not task:
            err = "task_expired" if task_manager.was_evicted(task_id) else "task_not_found"
            return TaskStatusBatchItem(task_id=task_id, status=None, error=err)

        # 读 status 后据其决定挂 result/srt/error，全程同步
        status = task.status
        item = TaskStatusBatchItem(task_id=task_id, status=status, progress=task.progress)
        if status == TaskStatus.COMPLETED:
            if task.output_format == "srt":
                # SRT 走 srt_content（非 result 字段）
                item.srt_content = task.srt_content
            else:
                item.result = task.result
        elif status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED):
            item.error = task.error
        # PENDING/PROCESSING → 小帧（result/srt_content/error 保持 None）
        return item

    async def _handle_cancel_task(self, websocket: WebSocketServerProtocol, task_id: str):
        """处理取消任务请求"""
        try:
            from src.core.task_manager import task_manager
            success = await task_manager.cancel_task(task_id)
            
            if success:
                await self._send_message(websocket, "task_cancelled", {
                    "task_id": task_id,
                    "message": "任务已取消"
                })
            else:
                await self._send_error(websocket, "cancel_failed", "任务取消失败")
                
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            await self._send_error(websocket, "cancel_error", str(e))
    
    async def notify_task_progress(self, task_id: str, progress: float, status: str, message: str = None):
        """通知任务进度"""
        connection_ids = self.task_connections.get(task_id, set())
        
        # 对于失败状态，提供更详细的错误分类
        error_type = None
        if status == "failed" and message:
            if "VAD" in message:
                error_type = "vad_error"
            elif "索引" in message or "index" in message:
                error_type = "index_error"
            elif "音频" in message:
                error_type = "audio_error"
            elif "模型" in message:
                error_type = "model_error"
            else:
                error_type = "unknown_error"
        
        data = {
            "task_id": task_id,
            "progress": progress,
            "status": status,
            "message": message,
            "error_type": error_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # 向所有相关连接发送进度更新
        for conn_id in list(connection_ids):
            if conn_id in self.connections:
                websocket = self.connections[conn_id]
                try:
                    await self._send_message(websocket, "task_progress", data)
                except Exception as e:
                    logger.error(f"发送进度通知失败: {e}")
                    self._cleanup_connection(conn_id)

    async def notify_task_error(
        self, task_id: str, error_type: str, message: str, status: str
    ):
        """向任务连接推送显式 error 终态通知，避免客户端继续等待进度消息。"""
        connection_ids = self.task_connections.get(task_id, set())
        for conn_id in list(connection_ids):
            if conn_id in self.connections:
                websocket = self.connections[conn_id]
                try:
                    await self._send_message(
                        websocket,
                        "error",
                        ErrorResponse(
                            error=error_type,
                            message=message,
                            task_id=task_id,
                            details={"status": status},
                        ).model_dump(),
                    )
                except Exception as e:
                    logger.error(f"发送任务错误通知失败: {e}")
                    self._cleanup_connection(conn_id)
    
    async def notify_task_complete(self, task_id: str, result: dict):
        """通知任务完成"""
        connection_ids = self.task_connections.get(task_id, set())
        
        data = {
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # 向所有相关连接发送完成通知
        for conn_id in list(connection_ids):
            if conn_id in self.connections:
                websocket = self.connections[conn_id]
                try:
                    await self._send_message(websocket, "task_complete", data)
                except Exception as e:
                    logger.error(f"发送完成通知失败: {e}")
                    self._cleanup_connection(conn_id)
        
        # 清理任务连接关系
        if task_id in self.task_connections:
            del self.task_connections[task_id]
    
    async def _send_message(self, websocket: WebSocketServerProtocol, msg_type: str, data: dict):
        """发送消息"""
        message = WebSocketMessage(type=msg_type, data=data)
        await websocket.send(message.model_dump_json())
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_type: str, message: str):
        """发送错误消息"""
        error = ErrorResponse(error=error_type, message=message)
        await self._send_message(websocket, "error", error.model_dump())
    
    def _cleanup_connection(self, connection_id: str):
        """清理连接"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        # 从任务连接中移除
        for task_id, conn_ids in list(self.task_connections.items()):
            if connection_id in conn_ids:
                conn_ids.remove(connection_id)
                if not conn_ids:
                    del self.task_connections[task_id]
        
        logger.debug(f"连接已清理: {connection_id}")
    
    async def _handle_chunked_upload_request(
        self,
        websocket: WebSocketServerProtocol,
        connection_id: str,
        data: dict,
        request: FileUploadRequest,
    ):
        """处理分片上传请求"""
        try:
            # 先清遗弃 session（机会式 sweep），再做硬数量上限准入控制，
            # 防恶意/坏客户端把 temp 文件堆满磁盘（codex: 保留 session 的磁盘压力面）
            self._sweep_upload_sessions()
            if len(self.upload_sessions) >= config.transcription.upload_session_max_count:
                await self._send_error(
                    websocket, "too_many_sessions",
                    f"并发上传会话已达上限({config.transcription.upload_session_max_count})，请稍后重试",
                )
                return

            # 生成任务ID
            task_id = str(uuid.uuid4())

            # 创建临时文件存储分片
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            
            # 创建上传会话
            session = {
                "task_id": task_id,
                "file_name": request.file_name,
                "file_size": request.file_size,
                "file_hash": request.file_hash,
                "chunk_size": data.get("chunk_size", 1024 * 1024),  # 默认1MB
                "total_chunks": data["total_chunks"],
                "received_chunks": 0,
                "temp_file_path": temp_file.name,
                "temp_file": temp_file,
                "chunks_received": set(),  # 记录已收到的分片索引
                "output_format": request.output_format,
                "force_refresh": request.force_refresh,
                "connection_id": connection_id,
                # PR1: 记录引擎选择，最终化时回填到 FileUploadRequest
                "engine": request.engine,
                # 词级时间戳: 记录 per-request 语言，最终化时回填
                "language": request.language,
                # diarize 开关: 记录 per-request 值，最终化时回填（缺省 True 向后兼容）
                "diarize": request.diarize,
                # word_align 开关: 记录 per-request 原始值（None=未指定），最终化时回填。
                # codex #2: 早返回缓存路径 (_finalize 内 create_task 之前) 须用此值解析 effective。
                "word_align": request.word_align,
                "terms": list(request.terms),
                "request": request,
                # 高负载止血: session 状态机 + TTL 时间戳
                # state: uploading → (收齐) ready → (提交成功) submitted；queue_full 时回到 ready 供重试
                "state": "uploading",
                "created_at": time.time(),
                # finalize 落地的最终文件路径（queue_full 重试时复用，避免重复落盘）
                "finalized_file_path": None,
            }
            
            self.upload_sessions[task_id] = session
            temp_file.close()
            
            # 关联任务和连接
            if task_id not in self.task_connections:
                self.task_connections[task_id] = set()
            self.task_connections[task_id].add(connection_id)
            
            logger.info(f"创建分片上传会话: {task_id}, 文件: {session['file_name']}, "
                       f"总大小: {session['file_size']/1024/1024:.2f}MB, "
                       f"分片数: {session['total_chunks']}")
            
            await self._send_message(websocket, "upload_ready", {
                "task_id": task_id,
                "message": "准备接收分片数据",
                "chunk_size": session["chunk_size"],
                "total_chunks": session["total_chunks"]
            })
            
        except Exception as e:
            logger.error(f"处理分片上传请求失败: {e}")
            await self._send_error(websocket, "chunked_upload_error", str(e))
    
    async def _handle_chunk_upload(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理分片数据上传"""
        try:
            task_id = data.get("task_id")
            chunk_index = data.get("chunk_index")
            
            if not task_id or chunk_index is None:
                await self._send_error(websocket, "missing_chunk_data", "缺少task_id或chunk_index")
                return
            
            if task_id not in self.upload_sessions:
                await self._send_error(websocket, "session_not_found", "上传会话不存在")
                return
            
            session = self.upload_sessions[task_id]
            
            # 检查分片是否重复
            if chunk_index in session["chunks_received"]:
                await self._send_message(websocket, "chunk_received", {
                    "task_id": task_id,
                    "chunk_index": chunk_index,
                    "status": "duplicate",
                    "progress": session["received_chunks"] / session["total_chunks"] * 100
                })
                return
            
            # 解码和验证分片数据
            chunk_data_base64 = data.get("chunk_data")
            if not chunk_data_base64:
                await self._send_error(websocket, "missing_chunk_data", "缺少分片数据")
                return
            
            chunk_data = base64.b64decode(chunk_data_base64)
            chunk_hash = hashlib.md5(chunk_data).hexdigest()
            
            # 验证分片哈希
            expected_hash = data.get("chunk_hash")
            if expected_hash and chunk_hash != expected_hash:
                await self._send_error(websocket, "chunk_hash_mismatch", 
                                     f"分片 {chunk_index} 哈希校验失败")
                return
            
            # 写入分片数据到临时文件
            with open(session["temp_file_path"], "r+b") as f:
                f.seek(chunk_index * session["chunk_size"])
                f.write(chunk_data)
            
            # 更新会话状态
            session["chunks_received"].add(chunk_index)
            session["received_chunks"] += 1
            
            progress = session["received_chunks"] / session["total_chunks"] * 100
            
            logger.debug(f"接收分片 {chunk_index}/{session['total_chunks']}, "
                        f"进度: {progress:.1f}%")
            
            # 发送分片接收确认
            await self._send_message(websocket, "chunk_received", {
                "task_id": task_id,
                "chunk_index": chunk_index,
                "progress": progress,
                "status": "received"
            })
            
            # 检查是否所有分片都已接收
            if session["received_chunks"] == session["total_chunks"]:
                await self._finalize_chunked_upload(websocket, task_id)
                
        except Exception as e:
            logger.error(f"处理分片上传失败: {e}")
            await self._send_error(websocket, "chunk_upload_error", str(e))
    
    async def _finalize_chunked_upload(self, websocket: WebSocketServerProtocol, task_id: str):
        """完成分片上传"""
        try:
            session = self.upload_sessions[task_id]
            
            logger.info(f"完成分片上传: {task_id}, 验证文件完整性...")
            
            # 验证完整文件哈希
            file_hash = self._calculate_file_hash(session["temp_file_path"])
            if file_hash != session["file_hash"]:
                await self._send_error(websocket, "file_hash_mismatch", "文件完整性校验失败")
                return
            
            # 检查缓存（如果不强制刷新）
            # cache key 含 engine（session 中已记录，无则回退 default_engine）;
            # word_align / diarize 折维收拢在 cache_params (D4, 此处无 task 对象走低层入口)
            if not session["force_refresh"]:
                from src.core.config import config as _config
                from src.models.schemas import TranscribeOptions, resolve_word_align
                _engine_for_cache = session.get("engine") or _config.transcription.default_engine
                _session_options = TranscribeOptions(
                    language=session.get("language"),
                    diarize=session.get("diarize", True),
                    terms=list(session.get("terms", [])),
                    # 决策 1A: 早返回缓存路径同样解析 effective word_align（请求 > config 兜底）
                    word_align=resolve_word_align(
                        session.get("word_align"), _config.qwen3.word_align_enabled
                    ),
                )
                from src.core.database import cache_allowed_for
                if cache_allowed_for(_session_options):
                    from src.core.database import db_manager, cache_params
                    _ce, _allow = cache_params(_engine_for_cache, _session_options)
                    cached_result = await db_manager.get_cached_result(
                        session["file_hash"], session["output_format"], engine=_ce, allow_cross_engine=_allow,
                        options=_session_options,
                    )
                else:
                    cached_result = None
                if cached_result:
                    logger.info(f"使用缓存结果（分片上传阶段）: {task_id}")

                    # E2: 早返回出口组装 effective options 回显 (session 回填值已在
                    # _session_options 收拢, 优先级 request > session 回填 > config)
                    # DRY(commit 1): projected 提取 + metadata 构建走共享纯函数 cache_hit_metadata
                    from src.core.result_projection import cache_hit_metadata
                    _md, _proj, _srt_ok = cache_hit_metadata(
                        cached_result, engine=_engine_for_cache, options=_session_options,
                        output_format=session["output_format"],
                    )

                    # 根据输出格式准备结果
                    if session["output_format"] == "srt":
                        if _srt_ok:
                            # cache_hit_metadata 用 get 不 pop, 此处排除 projected key 不泄漏给客户端
                            result_data = {k: v for k, v in cached_result.items() if k != "projected"}
                            result_data["metadata"] = _md
                        else:
                            result_data = None
                    else:
                        cached_result.metadata = _md
                        result_data = cached_result.model_dump()
                    
                    if result_data:
                        # 清理临时文件和会话
                        await self._cleanup_upload_session(task_id)
                        
                        # 直接返回缓存结果
                        await self._send_message(websocket, "task_complete", {
                            "task_id": task_id,
                            "result": result_data
                        })
                        return
            
            # 移动文件到最终位置（queue_full 重试时复用已落地文件，不重复落盘）
            from src.utils.file_utils import save_uploaded_file

            file_path = session.get("finalized_file_path")
            if not (file_path and os.path.exists(file_path)):
                with open(session["temp_file_path"], "rb") as f:
                    file_data = f.read()
                file_path, _ = await save_uploaded_file(file_data, session["file_name"])
                session["finalized_file_path"] = file_path

            request = session.get("request")
            if request is None:
                # 旧内存 session 没有 request 字段时只按旧字段兼容恢复；terms 缺失为空。
                from src.models.schemas import FileUploadRequest
                request = FileUploadRequest(
                    file_name=session["file_name"],
                    file_size=session["file_size"],
                    file_hash=session["file_hash"],
                    force_refresh=session["force_refresh"],
                    output_format=session["output_format"],
                    engine=session.get("engine"),
                    language=session.get("language"),
                    diarize=session.get("diarize", True),
                    word_align=session.get("word_align"),
                    terms=session.get("terms", []),
                )

            # 创建任务 + 提交队列
            from src.core.task_manager import task_manager, QueueFullError
            task = await task_manager.create_task(request, task_id=task_id)

            session["state"] = "finalizing"
            try:
                queue_info = await task_manager.submit_task(task_id, file_path)
            except QueueFullError as qe:
                # 准入控制拒绝：保留 session + 已落地文件，客户端据 retry_after 退避后
                # 发 finalize_upload 重试（不重传大文件）。self.tasks 回滚已在 task_manager 完成。
                session["state"] = "ready"
                logger.warning(f"分片任务入队被拒(queue_full): {task_id}, retry_after={qe.retry_after}s")
                await self._send_message(websocket, "queue_full", {
                    "task_id": task_id,
                    "retry_after": qe.retry_after,
                    "queue_size": qe.queue_size,
                    "max_queue_size": qe.max_queue_size,
                    # 兼容: 老客户端能识别的 error 形态 + 结构化字段（codex: 新消息类型别被旧客户端忽略）
                    "error": "queue_full",
                    "message": str(qe),
                })
                return

            # 提交成功：标记 submitted，清理 session（temp 删，finalized 文件交给 task 接管不删）
            session["state"] = "submitted"
            await self._cleanup_upload_session(task_id, delete_finalized=False)

            # 发送上传完成消息
            response_data = {
                "task_id": task_id,
                "message": "分片文件上传成功，开始处理"
            }

            # 如果启用了排队状态通知且任务在排队
            if config.transcription.queue_status_enabled and queue_info:
                if queue_info.get("queued", False):
                    response_data.update({
                        "queue_position": queue_info["position"],
                        "estimated_wait_seconds": queue_info["estimated_wait"],
                        "message": f"分片文件上传成功，排队位置: {queue_info['position']}"
                    })
                    await self._send_message(websocket, "task_queued", response_data)
                    return

            await self._send_message(websocket, "upload_complete", response_data)

            logger.info(f"分片上传完成: {task_id}, 文件: {session['file_name']}")

        except Exception as e:
            logger.error(f"完成分片上传失败: {e}")
            # 错误分级(codex): 提交成功后(submitted)绝不删最终文件——任务要用它；
            # 提交前的真错误 → 清 session(含 temp + 已落地文件)，防泄漏。
            submitted = self.upload_sessions.get(task_id, {}).get("state") == "submitted"
            if not submitted:
                await self._cleanup_upload_session(task_id, delete_finalized=True)
            await self._send_error(websocket, "finalize_error", str(e))

    async def _handle_finalize_upload(self, websocket: WebSocketServerProtocol, task_id: str):
        """显式重试 finalize（queue_full 后客户端重发，避免重传大文件）。

        - session 还在 → 走重试 finalize（复用已落地文件）
        - session 已清但 task 已提交 → 幂等返回当前状态，不重复入队（codex: 防 double-submit）
        - 都没有 → 会话不存在/已过期
        """
        if task_id in self.upload_sessions:
            await self._finalize_chunked_upload(websocket, task_id)
            return
        from src.core.task_manager import task_manager
        if task_manager.get_task(task_id):
            await self._send_task_status(websocket, task_id)
        else:
            await self._send_error(websocket, "session_not_found", "上传会话不存在或已过期")

    def _sweep_upload_sessions(self):
        """清理超 TTL 的遗弃上传 session（含 temp + 已落地文件），防磁盘泄漏。

        机会式调用（新建 session 时），无需独立后台循环；配合硬数量上限做准入控制。
        """
        ttl = config.transcription.upload_session_ttl_seconds
        now = time.time()
        for task_id in list(self.upload_sessions.keys()):
            sess = self.upload_sessions[task_id]
            if now - sess.get("created_at", now) > ttl:
                self._remove_session_files(sess, delete_finalized=True)
                del self.upload_sessions[task_id]
                logger.info(f"清理遗弃上传会话(TTL): {task_id}")
    
    def _calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """高效计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # 分块计算哈希，避免大文件内存占用
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _remove_session_files(self, session: dict, delete_finalized: bool = True):
        """删除 session 关联的磁盘文件。

        delete_finalized=False 时保留已落地的最终文件（提交成功后由 task 接管其生命周期，
        不能删，否则任务拿不到文件）。
        """
        keys = ["temp_file_path"]
        if delete_finalized:
            keys.append("finalized_file_path")
        for key in keys:
            path = session.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"清理文件({key}): {path}")
                except Exception as e:
                    logger.error(f"清理文件失败({key}): {e}")

    async def _cleanup_upload_session(self, task_id: str, delete_finalized: bool = True):
        """清理上传会话（含磁盘文件）。

        delete_finalized: 提交成功路径传 False（最终文件交给 task），
        其余（真错误/遗弃 sweep）传 True 全清防泄漏。
        """
        if task_id in self.upload_sessions:
            session = self.upload_sessions[task_id]
            self._remove_session_files(session, delete_finalized=delete_finalized)
            del self.upload_sessions[task_id]
            logger.debug(f"清理上传会话: {task_id}")


# 全局WebSocket处理器实例
ws_handler = WebSocketHandler()
