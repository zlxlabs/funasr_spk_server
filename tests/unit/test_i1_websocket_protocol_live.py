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
