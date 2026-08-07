"""I1 terms 协议：规范化、authority 和 invalid_terms 映射。"""
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import ValidationError
from src.api.websocket_handler import WebSocketHandler
from src.models.schemas import FileUploadRequest
def _request(**values):
    return FileUploadRequest(file_name="terms.wav", file_size=1, file_hash="h-terms", **values)
def test_terms_normalize_nfkc_trim_whitespace_dedup_and_order():
    assert _request(terms=["  Ａ  B ", "A B", "", "  ", "C"]).terms == ["A B", "C"]
@pytest.mark.parametrize(("terms", "reason"), [
    (["x"] * 101, "too_many_raw_items"), (["x" * 257], "raw_item_too_long"),
    ([f"term-{i}" for i in range(51)], "too_many_terms"), (["x" * 65], "term_too_long"),
    ([f"{i:02d}" + "x" * 48 for i in range(21)], "total_too_long"),
])
def test_terms_limits_use_invalid_terms_reason(terms, reason):
    with pytest.raises(ValidationError) as caught:
        _request(terms=terms)
    error = caught.value.errors()[0]
    assert (error["type"], error["ctx"]["reason"]) == ("invalid_terms", reason)
def test_terms_missing_is_empty_and_task_manager_copies_effective_terms():
    assert _request().terms == []
    request = _request(terms=["  Alpha  ", "Alpha", "Beta"])
    from src.core.task_manager import TaskManager
    task = asyncio.run(TaskManager().create_task(request, task_id="terms-copy"))
    assert task.options.terms == ["Alpha", "Beta"] and task.options.terms is not request.terms
@pytest.mark.asyncio
async def test_chunked_session_keeps_validated_terms_without_raw_data_reparse():
    handler, websocket = WebSocketHandler(), MagicMock()
    websocket.send = AsyncMock()
    request = _request(terms=["  Alpha  ", "Alpha"])
    data = {**request.model_dump(), "total_chunks": 1, "upload_mode": "chunked"}
    await handler._handle_chunked_upload_request(websocket, "conn-terms", data, request=request)
    session = next(iter(handler.upload_sessions.values()))
    assert session["terms"] == ["Alpha"] and session["request"] is request


@pytest.mark.asyncio
async def test_upload_validation_maps_only_invalid_terms_to_protocol_error():
    handler, websocket = WebSocketHandler(), MagicMock()
    websocket.send = AsyncMock()
    data = {"file_name": "terms.wav", "file_size": 1, "file_hash": "h", "terms": ["x"] * 101}
    await handler._handle_upload_request(websocket, "conn-terms", data)
    sent = websocket.send.await_args.args[0]
    assert all(token in sent for token in ('"type":"error"', '"error":"invalid_terms"', '"reason":"too_many_raw_items"'))


@pytest.mark.asyncio
async def test_finalize_terms_field_is_rejected_without_changing_session():
    handler, websocket = WebSocketHandler(), MagicMock()
    websocket.send = AsyncMock()
    handler.upload_sessions["task-terms"] = {"state": "ready", "terms": ["Alpha"]}
    await handler._handle_message(websocket, "conn-terms", {
        "type": "finalize_upload", "data": {"task_id": "task-terms", "terms": ["Other"]},
    })
    assert handler.upload_sessions["task-terms"]["terms"] == ["Alpha"]
    assert '"error":"protocol_error"' in websocket.send.await_args.args[0]
