"""I1 真实 websockets.serve 入口：协议与 capability 绑定。"""
from contextlib import asynccontextmanager
import asyncio
import base64
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import websockets

from src.api import http_endpoints, websocket_handler
from src.api.http_endpoints import HttpEndpoints
from src.api.websocket_handler import WebSocketHandler
from src.models.schemas import TranscribeOptions, TranscriptionTask


@asynccontextmanager
async def _running_server(monkeypatch):
    from src.core.config import config

    monkeypatch.setattr(config.auth, "enabled", False)
    monkeypatch.setattr(config.observability, "metrics_enabled", True)
    monkeypatch.setattr(config.transcription, "default_engine", "funasr")
    monkeypatch.setattr(http_endpoints, "detect_runtime", lambda: SimpleNamespace(name="cpu"))
    monkeypatch.setattr(websocket_handler, "detect_runtime", lambda: SimpleNamespace(name="cpu"))
    handler = WebSocketHandler()
    tm = MagicMock(is_running=True)
    tm._maintenance_task = MagicMock()
    tm._maintenance_task.done.return_value = False
    endpoint = HttpEndpoints(tm, MagicMock(), config)
    server = await websockets.serve(
        handler.handle_connection, "127.0.0.1", 0, process_request=endpoint.process_request,
    )
    try:
        yield handler, server.sockets[0].getsockname()[1], config
    finally:
        server.close()
        await server.wait_closed()


async def _receive_type(ws, expected):
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
        if message["type"] == expected:
            return message


def _fake_task_manager():
    tasks, requests = {}, []
    tm = MagicMock()

    async def create_task(request, task_id=None):
        task_id = task_id or f"live-{len(tasks)}"
        requests.append(request)
        task = TranscriptionTask(
            task_id=task_id, file_name=request.file_name, file_path="",
            file_size=request.file_size, file_hash=request.file_hash, engine="funasr",
            output_format=request.output_format,
            options=TranscribeOptions(
                language=request.language, diarize=request.diarize,
                word_align=bool(request.word_align), terms=list(request.terms),
            ),
        )
        tasks[task_id] = task
        return task

    tm.create_task = AsyncMock(side_effect=create_task)
    tm.get_task = MagicMock(side_effect=tasks.get)
    tm.submit_task = AsyncMock(return_value=None)
    return tm, tasks, requests


async def _single_upload(ws, handler, terms):
    audio = b"fake-audio"
    request = {
        "file_name": "single.wav", "file_size": len(audio),
        "file_hash": hashlib.md5(audio).hexdigest(), "force_refresh": True,
    }
    if terms is not None:
        request["terms"] = terms
    await ws.send(json.dumps({"type": "upload_request", "data": request}))
    ready = await _receive_type(ws, "upload_ready")
    task_id = ready["data"]["task_id"]
    await ws.send(json.dumps({
        "type": "upload_data",
        "data": {"task_id": task_id, "file_data": base64.b64encode(audio).decode()},
    }))
    await _receive_type(ws, "upload_complete")
    await handler.notify_task_complete(task_id, {
        "metadata": {"engine": "funasr", "terms_count": len(terms or [])},
    })
    return await _receive_type(ws, "task_complete")


@pytest.mark.asyncio
@pytest.mark.parametrize("terms", [None, [], ["Alpha"]])
async def test_live_single_upload_terms_and_http_ws_capabilities(monkeypatch, tmp_path, terms):
    async with _running_server(monkeypatch) as (handler, port, _config):
        tm, _tasks, requests = _fake_task_manager()

        async def save_file(data, file_name):
            path = tmp_path / file_name
            path.write_bytes(data)
            return str(path), None

        with patch("src.core.task_manager.task_manager", tm), \
             patch("src.utils.file_utils.save_uploaded_file", new=AsyncMock(side_effect=save_file)):
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://127.0.0.1:{port}/capabilities?probe=ws")
            assert response.status_code == 200
            http_capabilities = response.json()
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                connected = await _receive_type(ws, "connected")
                assert connected["data"]["capabilities"] == http_capabilities
                completed = await _single_upload(ws, handler, terms)
                assert completed["data"]["result"]["metadata"]["terms_count"] == len(terms or [])

        assert requests[0].terms == (terms or [])


@pytest.mark.asyncio
async def test_live_invalid_terms_returns_protocol_error(monkeypatch):
    async with _running_server(monkeypatch) as (handler, port, _config):
        tm, _tasks, requests = _fake_task_manager()
        with patch("src.core.task_manager.task_manager", tm):
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                await _receive_type(ws, "connected")
                await ws.send(json.dumps({
                    "type": "upload_request",
                    "data": {"file_name": "bad.wav", "file_size": 1,
                             "file_hash": "bad", "terms": ["x"] * 101},
                }))
                error = await _receive_type(ws, "error")
                assert error["data"]["error"] == "invalid_terms"
                assert error["data"]["reason"] == "too_many_raw_items"
        assert requests == []


@pytest.mark.asyncio
async def test_live_chunked_upload_terms_reach_finalize(monkeypatch, tmp_path):
    async with _running_server(monkeypatch) as (handler, port, _config):
        tm, _tasks, requests = _fake_task_manager()
        audio = b"chunk-audio"

        async def save_file(data, file_name):
            path = tmp_path / "chunk-final.wav"
            path.write_bytes(data)
            return str(path), None

        with patch("src.core.task_manager.task_manager", tm), \
             patch("src.utils.file_utils.save_uploaded_file", new=AsyncMock(side_effect=save_file)):
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                await _receive_type(ws, "connected")
                await ws.send(json.dumps({"type": "upload_request", "data": {
                    "file_name": "chunk.wav", "file_size": len(audio),
                    "file_hash": hashlib.md5(audio).hexdigest(), "force_refresh": True,
                    "upload_mode": "chunked", "total_chunks": 1,
                    "chunk_size": len(audio), "terms": ["Alpha"],
                }}))
                ready = await _receive_type(ws, "upload_ready")
                task_id = ready["data"]["task_id"]
                await ws.send(json.dumps({"type": "upload_chunk", "data": {
                    "task_id": task_id, "chunk_index": 0,
                    "chunk_data": base64.b64encode(audio).decode(),
                    "chunk_hash": hashlib.md5(audio).hexdigest(),
                }}))
                await _receive_type(ws, "chunk_received")
                await _receive_type(ws, "upload_complete")
                await handler.notify_task_complete(task_id, {"metadata": {"terms_count": 1}})
                await _receive_type(ws, "task_complete")
        assert requests[0].terms == ["Alpha"]
        assert tm.submit_task.await_count == 1
        assert handler.upload_sessions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["json", "srt"])
async def test_live_reconnect_reads_original_task_result_after_disconnect(
    monkeypatch, tmp_path, output_format,
):
    """真实 socket 断开不取消任务，重连后 batch 查询取回对应格式的终态。"""
    from src.models.schemas import TaskStatus, TranscriptionResult, TranscriptionSegment

    async with _running_server(monkeypatch) as (handler, port, _config):
        tm, tasks, _requests = _fake_task_manager()
        audio = b"reconnect-audio"

        async def save_file(data, file_name):
            path = tmp_path / file_name
            path.write_bytes(data)
            return str(path), None

        with patch("src.core.task_manager.task_manager", tm), \
             patch("src.utils.file_utils.save_uploaded_file", new=AsyncMock(side_effect=save_file)):
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as first_ws:
                await _receive_type(first_ws, "connected")
                await first_ws.send(json.dumps({"type": "upload_request", "data": {
                    "file_name": "reconnect.wav", "file_size": len(audio),
                    "file_hash": hashlib.md5(audio).hexdigest(),
                    "force_refresh": True, "output_format": output_format,
                }}))
                ready = await _receive_type(first_ws, "upload_ready")
                task_id = ready["data"]["task_id"]
                await first_ws.send(json.dumps({"type": "upload_data", "data": {
                    "task_id": task_id,
                    "file_data": base64.b64encode(audio).decode(),
                }}))
                await _receive_type(first_ws, "upload_complete")

                # 模拟 worker 已接手任务后客户端掉线，任务对象仍处于 PROCESSING。
                tasks[task_id].status = TaskStatus.PROCESSING

            assert tasks[task_id].status == TaskStatus.PROCESSING
            assert tm.create_task.await_count == 1
            assert tm.submit_task.await_count == 1
            tm.cancel_task.assert_not_called()

            async with websockets.connect(f"ws://127.0.0.1:{port}/") as resumed_ws:
                await _receive_type(resumed_ws, "connected")
                query = json.dumps({"type": "task_status_batch", "data": {"task_ids": [task_id]}})
                await resumed_ws.send(query)
                processing_response = await _receive_type(resumed_ws, "task_status_batch")
                processing_item, = processing_response["data"]["items"]
                assert processing_item["task_id"] == task_id
                assert processing_item["status"] == "processing"
                assert processing_item["result"] is None
                assert processing_item["srt_content"] is None

                if output_format == "json":
                    tasks[task_id].result = TranscriptionResult(
                        task_id=task_id,
                        file_name="reconnect.wav",
                        file_hash=hashlib.md5(audio).hexdigest(),
                        duration=1.0,
                        segments=[TranscriptionSegment(
                            start_time=0.0, end_time=1.0, text="recovered",
                        )],
                        speakers=[],
                        processing_time=0.1,
                        metadata={"engine": "funasr"},
                    )
                    original_result = tasks[task_id].result.model_dump(mode="json")
                else:
                    original_result = "1\n00:00:00,000 --> 00:00:01,000\nrecovered\n"
                    tasks[task_id].srt_content = original_result
                tasks[task_id].progress = 100.0
                tasks[task_id].status = TaskStatus.COMPLETED

                await resumed_ws.send(query)
                response = await _receive_type(resumed_ws, "task_status_batch")

        item, = response["data"]["items"]
        assert item["task_id"] == task_id
        assert item["status"] == "completed"
        if output_format == "json":
            assert item["result"] == original_result
            assert item["srt_content"] is None
        else:
            assert item["srt_content"] == original_result
            assert item["result"] is None
        assert tm.create_task.await_count == 1
        assert tm.submit_task.await_count == 1
        tm.cancel_task.assert_not_called()
