"""缓存 TTL cutoff 必须使用 SQLite CURRENT_TIMESTAMP 的 UTC 时间基准。"""
from __future__ import annotations

import os
import time
from pathlib import Path

import aiosqlite
import pytest

from src.core.config import config
from src.core.database import DatabaseManager


@pytest.mark.asyncio
async def test_ttl_cutoff_uses_utc_in_non_utc_timezone(tmp_path: Path, monkeypatch):
    """非 UTC 本地时区下，刚过 TTL 的 SQLite UTC 行应删除，未过期行应保留。"""
    if not hasattr(time, "tzset"):
        pytest.skip("该回归测试需要支持 time.tzset 的运行环境")

    old_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        assert time.localtime().tm_gmtoff != 0, "测试前提要求本地时区不是 UTC"
        monkeypatch.setattr(config.database, "max_cache_days", 30)
        monkeypatch.setattr(config.database, "max_cache_count", 0)
        db_path = str(tmp_path / "cache-utc.db")
        manager = DatabaseManager(db_path=db_path)
        await manager.init_db()

        async with aiosqlite.connect(db_path) as db:
            for file_hash, ttl_offset in (
                ("just-expired", "-5 minutes"),
                ("not-expired", "+5 minutes"),
            ):
                await db.execute(
                    """INSERT INTO transcription_cache
                       (file_hash, file_name, result, raw_result, duration,
                        processing_time, engine, created_at, accessed_at)
                       VALUES (?, ?, '{}', '{}', 1, 0.1, 'funasr',
                               datetime(CURRENT_TIMESTAMP, '-30 days', ?), CURRENT_TIMESTAMP)""",
                    (file_hash, f"{file_hash}.wav", ttl_offset),
                )
            await db.commit()

        await manager.clean_old_cache()
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT file_hash FROM transcription_cache ORDER BY file_hash"
            )
            hashes = {row[0] for row in await cursor.fetchall()}
        assert "just-expired" not in hashes
        assert "not-expired" in hashes
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        time.tzset()
