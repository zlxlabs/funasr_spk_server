"""代理字符 terms 的用户错误文案准确且保持不可重试分类。"""
from __future__ import annotations

import pytest

from src.core.file_based_process_pool import FileBasedProcessPool
from src.core.task_manager import ErrorKind, classify_error


@pytest.mark.asyncio
async def test_unencodable_terms_have_accurate_non_retryable_error(tmp_path):
    """公开进程池派发入口在启动 worker 前拒绝代理字符 terms。"""
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    pool = object.__new__(FileBasedProcessPool)

    with pytest.raises(ValueError, match="terms 必须是 UTF-8 可编码文本") as raised:
        await pool.generate_with_pool(
            str(audio),
            extra_task_fields={"options": {"terms": ["valid", "\ud800"]}},
        )

    message = str(raised.value)
    assert "不支持的文件格式" not in message
    assert classify_error(raised.value) is ErrorKind.NON_RETRYABLE_INPUT
