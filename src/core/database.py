"""
数据库管理模块

PR1 变更：
- DatabaseManager 接受可选 db_path（提升可测性）
- transcription_cache 表新增 engine 列
- UNIQUE 约束从 file_hash 单列改为 (file_hash, engine)
- init_db 对旧 schema 自动迁移（加列 + 重建索引）
- get_cached_result / save_result 增加 engine 参数（默认 funasr 保持向后兼容）
"""
import aiosqlite
import json
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
from loguru import logger
from pydantic import ValidationError
from src.core.config import config
from src.core.result_projection import (
    merge_segments_view,
    project_result_nospk,
    segments_to_srt_text,
)
from src.models.schemas import TranscribeOptions, TranscriptionResult


# 数据库 schema 当前期望的引擎列默认值（迁移和旧调用兼容时使用）
DEFAULT_ENGINE = "funasr"


def compute_cache_engine(
    engine: str,
    *,
    word_align_enabled: bool,
    language: Optional[str],
    word_align_language: str,
    diarize: bool = True,
    output_format: str = "json",
) -> str:
    """把 word_align / diarize 状态折进缓存用的 engine tag (D9 字符串折维).

    word_align 给 segment 加 words、diarize=false 抹 speaker, 都是契约变化:
    不同形态的结果不能互相命中. 所以缓存 key (file_hash, engine) 的 engine 维
    按固定顺序折状态 (缺省不写):

    - qwen3                  (word_align 关 + diarize 开, 老 key 不变)
    - qwen3+wa:<lang>        (word_align 开; lang = per-request language strip 后
                              非空则用之, 否则 config word_align_language 兜底)
    - qwen3+nospk            (diarize 关)
    - qwen3+wa:<lang>+nospk  (两者叠加, 顺序固定 +wa 在前)
    - 非 qwen3 → 原样 (funasr 免折维, D4: 存一行 diarized, serve 层按需投影)

    ⚠️ 决策 2A (2026-06-16): word_align 是 JSON-only (SRT 不挂词, transcribe :699 写死).
    SRT 请求即使 word_align=true, 产出的行实际无词, 故 **缓存维度有效 word_align =
    word_align_enabled AND output_format=='json'**: SRT 降回纯 tag (无 +wa), 避免无词行
    被未来 JSON +wa 请求误命中 (codex #5 同向). output_format 默认 json 向后兼容老调用.

    ⚠️ 触发条件写死: 折维维度 >3 时升级结构化 variant 列, 不再加字符串后缀.
    """
    if engine != "qwen3":
        return engine
    tag = engine
    # 决策 2A: SRT 行无词, 缓存维度上强制视 word_align 为关
    effective_wa = word_align_enabled and output_format == "json"
    if effective_wa:
        # language 规范化 (T-D #8): strip() 后非空才生效, 否则 config 兜底
        lang = (language or "").strip() or word_align_language
        tag += f"+wa:{lang}"
    if not diarize:
        tag += "+nospk"
    return tag


def cache_lookup_params(
    engine: str,
    *,
    word_align_enabled: bool,
    language: Optional[str],
    word_align_language: str,
    diarize: bool = True,
    output_format: str = "json",
):
    """返回 (cache_engine, allow_cross_engine) 给 get_cached_result.

    带折维 tag (+wa / +nospk) 时强制 strict (allow_cross_engine=False): 否则
    跨引擎按 file_hash 回退可能命中形态不符的行 (无词 / 带 speaker), 把请求
    降级成错误形态结果 (T-D #7).
    """
    cache_engine = compute_cache_engine(
        engine,
        word_align_enabled=word_align_enabled,
        language=language,
        word_align_language=word_align_language,
        diarize=diarize,
        output_format=output_format,
    )
    allow_cross = False if cache_engine != engine else None
    return cache_engine, allow_cross


def cache_params(engine: str, options, output_format: str = "json") -> tuple:
    """(cache_engine_tag, allow_cross_engine) — 查询与写入共用的统一入口 (D4 收拢).

    决策 1A (2026-06-16): word_align 改读 per-request options.word_align (已在
    create_task / 分片 session 由 resolve_word_align 解析成 effective bool), 不再读
    全局 config. language / diarize 同样从 options 读. 写入只用第一个元素 (tag), 查询两个都用.

    output_format 透传给 compute_cache_engine 做 2A 的 SRT 降 +wa.

    Args:
        engine: ASR 引擎名 (task.engine 或 session 回退 default_engine 后的值).
        options: TranscribeOptions (language / diarize / word_align).
        output_format: json / srt (SRT 时 +wa 折维降级, 决策 2A).
    """
    return cache_lookup_params(
        engine,
        word_align_enabled=options.word_align,
        language=options.language,
        word_align_language=config.qwen3.word_align_language,
        diarize=options.diarize,
        output_format=output_format,
    )


def cache_params_for(task) -> tuple:
    """cache_params 的 TranscriptionTask 便捷入口 — 消灭各处手写折维参数."""
    return cache_params(task.engine, task.options, task.output_format)


def cache_allowed_for(options: TranscribeOptions) -> bool:
    """有效 terms 不得命中或写入普通缓存；空 terms 保持旧缓存语义。"""
    return not options.terms


def cache_save_engine_for(task, has_words: bool) -> str:
    """写入用 cache tag — 决策 B (codex #5): 请求 word_align 但实际无词 (CUDA+CPU 都失败)
    时降 +wa 维, 存 base/其它维 tag, 避免无词结果 exact-hit 永久毒化该文件的 +wa 缓存.

    其它折维 (+nospk 等) 不受影响. 非 +wa 请求原样.
    """
    tag = cache_params_for(task)[0]
    if not has_words and "+wa:" in tag:
        demoted = task.options.model_copy(update={"word_align": False})
        tag = cache_params(task.engine, demoted, task.output_format)[0]
    return tag


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: 可选的数据库文件路径。不传则使用 config.database.path（保持现有单例语义）。
                     传入路径用于测试隔离或多实例场景。
        """
        self.db_path = db_path if db_path is not None else config.database.path
        # 可观测性 (stats 三维度之一): 投影 serve 命中计数 (进程内, 不持久化)
        self.projected_serves = 0
        self._ensure_db_dir()

    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self):
        """初始化数据库（含旧 schema 自动迁移）"""
        async with aiosqlite.connect(self.db_path) as db:
            # 1. 建基础表（已存在则跳过）
            #    注：fresh 创建时这里包含 engine 列；
            #    旧库（无 engine 列）后续会通过 ALTER 补齐。
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transcription_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    result TEXT NOT NULL,
                    raw_result TEXT,
                    duration REAL,
                    processing_time REAL,
                    engine TEXT NOT NULL DEFAULT 'funasr',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. 列升级（兼容旧库）
            cursor = await db.execute("PRAGMA table_info(transcription_cache)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

            if 'raw_result' not in column_names:
                await db.execute("ALTER TABLE transcription_cache ADD COLUMN raw_result TEXT")
                logger.info("数据库迁移：新增 raw_result 列")

            if 'duration' not in column_names:
                await db.execute("ALTER TABLE transcription_cache ADD COLUMN duration REAL")
                logger.info("数据库迁移：新增 duration 列")

            if 'processing_time' not in column_names:
                await db.execute("ALTER TABLE transcription_cache ADD COLUMN processing_time REAL")
                logger.info("数据库迁移：新增 processing_time 列")

            if 'engine' not in column_names:
                # 旧库迁移：加 engine 列 + 已有行回填为 funasr
                await db.execute(
                    "ALTER TABLE transcription_cache ADD COLUMN engine TEXT NOT NULL DEFAULT 'funasr'"
                )
                logger.info("数据库迁移：新增 engine 列，旧数据默认 engine=funasr")

            # 3. 重建索引：丢掉旧的 file_hash 单列 UNIQUE，建 (file_hash, engine) 复合 UNIQUE
            #    SQLite 的 CREATE TABLE IF NOT EXISTS 不会重建 UNIQUE 约束，需要靠索引层处理。
            #    PRAGMA index_list 检查
            cursor = await db.execute("PRAGMA index_list(transcription_cache)")
            indexes = await cursor.fetchall()
            for idx in indexes:
                # idx: (seq, name, unique, origin, partial)
                name = idx[1]
                is_unique = idx[2] == 1
                if not is_unique:
                    continue
                cursor2 = await db.execute(f"PRAGMA index_info({name})")
                idx_cols = [row[2] for row in await cursor2.fetchall()]
                # 旧的单列 file_hash UNIQUE（自动命名 sqlite_autoindex_... 或我们的旧索引）
                if idx_cols == ["file_hash"]:
                    # SQLite 不允许 DROP INDEX 由 UNIQUE 约束自动生成的 sqlite_autoindex_*
                    # 这种情况下需要重建表
                    if name.startswith("sqlite_autoindex_"):
                        await self._rebuild_table_for_unique_migration(db)
                    else:
                        await db.execute(f"DROP INDEX IF EXISTS {name}")
                        logger.info(f"数据库迁移：删除旧 UNIQUE 索引 {name}")

            # 4. 创建新的复合 UNIQUE 索引（已存在则跳过）
            await db.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_cache_hash_engine
                    ON transcription_cache(file_hash, engine)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON transcription_cache(file_hash)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON transcription_cache(created_at)
            """)
            # P4 A1: size-cap 按 accessed_at LRU 挤, 加索引避免 ORDER BY 全表扫
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at ON transcription_cache(accessed_at)
            """)

            await db.commit()
            logger.info("数据库初始化完成")

    async def _rebuild_table_for_unique_migration(self, db: aiosqlite.Connection):
        """
        旧 schema 的 file_hash UNIQUE 是 CREATE TABLE 内联约束，
        对应 sqlite_autoindex_* 不允许 DROP INDEX。
        策略：拷贝数据到新表，drop 旧表，rename。
        """
        logger.warning("数据库迁移：检测到旧 file_hash UNIQUE 内联约束，开始重建表")
        await db.execute("""
            CREATE TABLE transcription_cache_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                file_name TEXT NOT NULL,
                result TEXT NOT NULL,
                raw_result TEXT,
                duration REAL,
                processing_time REAL,
                engine TEXT NOT NULL DEFAULT 'funasr',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 拷贝时回填 engine='funasr'
        await db.execute("""
            INSERT INTO transcription_cache_new
                (id, file_hash, file_name, result, raw_result, duration, processing_time, engine, created_at, accessed_at)
            SELECT id, file_hash, file_name, result, raw_result, duration, processing_time,
                   COALESCE(engine, 'funasr'), created_at, accessed_at
            FROM transcription_cache
        """)
        await db.execute("DROP TABLE transcription_cache")
        await db.execute("ALTER TABLE transcription_cache_new RENAME TO transcription_cache")
        logger.info("数据库迁移：旧表重建完成，原有数据已保留并填充 engine=funasr")

    async def get_cached_result(
        self,
        file_hash: str,
        output_format: str = "json",
        engine: str = DEFAULT_ENGINE,
        allow_cross_engine: Optional[bool] = None,
        options=None,
    ) -> Optional[TranscriptionResult]:
        """
        获取缓存的转录结果

        Args:
            file_hash: 文件 MD5
            output_format: 返回格式(json / srt), 影响返回结构
            engine: 折维后的缓存 tag (cache_params_for 输出, 可能带 +wa/+nospk 后缀)
            allow_cross_engine: 跨引擎共享开关. None → 走 config.transcription.cache_cross_engine
                True: 精确 engine miss 后回退按 file_hash 查任意 engine 最新行
                False: strict, 仅按 (file_hash, engine) 查
            options: per-request TranscribeOptions. diarize=False 时本出口做 nospk
                投影 (D3 双出口之一): exact +nospk tag miss → 回退同引擎同 wa-tag
                diarized 行现场投影 (标 projected, **不回写**, E1+T2); funasr 免折维
                行 (本身 diarized) 直接出口投影.

        跨引擎 SRT 命中时从 TranscriptionResult.segments 重建(不依赖 raw_result 引擎特定结构).
        """
        if not config.transcription.cache_enabled:
            return None

        nospk = options is not None and options.diarize is False
        effective_cross = (
            config.transcription.cache_cross_engine
            if allow_cross_engine is None
            else allow_cross_engine
        )

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 1) 先精确匹配 (file_hash, engine)
                cursor = await db.execute(
                    "SELECT result, raw_result, file_name, duration, engine "
                    "FROM transcription_cache WHERE file_hash = ? AND engine = ?",
                    (file_hash, engine),
                )
                row = await cursor.fetchone()

                # 1.5) nospk 投影回退 (E1): exact +nospk miss → 显式只查同引擎同
                #      wa-tag 的 diarized 行 (strip "+nospk" 后缀), 禁 cross-engine.
                #      命中后在出口投影返回, 不回写 (T2: 缓存里永远只有真算结果).
                if row is None and nospk and engine.endswith("+nospk"):
                    diarized_tag = engine[: -len("+nospk")]
                    cursor = await db.execute(
                        "SELECT result, raw_result, file_name, duration, engine "
                        "FROM transcription_cache WHERE file_hash = ? AND engine = ?",
                        (file_hash, diarized_tag),
                    )
                    row = await cursor.fetchone()
                    if row is not None:
                        logger.info(
                            f"nospk 投影回退命中: file_hash={file_hash} "
                            f"diarized_tag={diarized_tag} (现场投影, 不回写)"
                        )

                # 2) miss 时按 file_hash 回退到任意 engine 最新一行(若开启跨引擎共享)
                #    决策 C (codex #4): 只回退到"裸 engine"行 (engine NOT LIKE '%+%'),
                #    排除 +wa / +nospk 折维行. 否则 base 请求 (无 tag, allow_cross=None
                #    跟随 config) 可能跨引擎命中 +wa 行 → 没要词却返回带词 (反向污染).
                #    折维形态的 strict 隔离由此双向对称.
                if row is None and effective_cross:
                    cursor = await db.execute(
                        "SELECT result, raw_result, file_name, duration, engine "
                        "FROM transcription_cache WHERE file_hash = ? "
                        "AND engine NOT LIKE '%+%' "
                        "ORDER BY created_at DESC LIMIT 1",
                        (file_hash,),
                    )
                    row = await cursor.fetchone()

                if row is None:
                    return None

                result_json, raw_result_json, file_name, duration, cached_engine = row
                cross_hit = cached_engine != engine
                if cross_hit:
                    logger.info(
                        f"跨引擎缓存命中: file_hash={file_hash} "
                        f"cached_engine={cached_engine} requested_engine={engine}"
                    )

                # 更新 accessed_at(用 cached_engine 定位实际那行)
                await db.execute(
                    "UPDATE transcription_cache SET accessed_at = CURRENT_TIMESTAMP "
                    "WHERE file_hash = ? AND engine = ?",
                    (file_hash, cached_engine),
                )
                await db.commit()

                if output_format == "json":
                    result_data = json.loads(result_json)
                    result = TranscriptionResult(**result_data)
                    # D3: funasr 行 JSON 出口先 merge 视图 (句级→合并, 带 span cap),
                    # 再 nospk。顺序铁律: merge 需要 speaker; 反序会把 null speaker 全并成一条.
                    # 以行的 engine 列判定, 不看请求; qwen3 家族 (含折维 tag) 一律不过.
                    merge_applied = False
                    if cached_engine == "funasr":
                        result.segments = merge_segments_view(
                            result.segments,
                            gap_sec=config.transcription.segment_merge_gap_sec,
                            max_span_sec=config.transcription.segment_merge_max_span_sec,
                        )
                        merge_applied = True
                    was_projected = False
                    if nospk:
                        # 出口投影 (D8): 投影回退行 / funasr diarized 行 → 抹 speaker.
                        # projected=True 表示"由 diarized 行投影而来"; exact nospk 行
                        # (真算的) 为 False. metadata 是请求级属性, 不入库.
                        was_projected = bool(result.speakers)
                        result = project_result_nospk(result)
                        if was_projected:
                            self.projected_serves += 1
                    # serve 层通道 (与 projected 同形, get 不 pop; 不入库):
                    # segment_merge_applied 让 cache_hit_metadata 按事实回显 cap 键,
                    # 不跟请求 engine 推断 (跨引擎回退时请求 engine ≠ 行 engine).
                    channel = {}
                    if nospk:
                        channel["projected"] = was_projected
                    if merge_applied:
                        channel["segment_merge_applied"] = True
                    result.metadata = channel or None
                    return result
                if output_format == "srt":
                    if nospk:
                        # nospk SRT: 旁路 raw 路径 (funasr raw 会渲染出 SpeakerN: 前缀),
                        # 从投影 segments 重渲染无前缀 (D8 + T-D #4 渲染点).
                        # **不过 merge 视图** — SRT 语义是原始分割; 句级直接渲染
                        # (canonical 改句级后从粗变细是预期修正).
                        result_data = json.loads(result_json)
                        tr = TranscriptionResult(**result_data)
                        was_projected = bool(tr.speakers)
                        tr = project_result_nospk(tr)
                        if was_projected:
                            self.projected_serves += 1
                        return {
                            "format": "srt",
                            "content": self._segments_to_srt(tr.segments),
                            "file_name": file_name,
                            "file_hash": file_hash,
                            "duration": duration,
                            "projected": was_projected,
                        }
                    # SRT 重建分流按 raw 结构而非引擎名 (T-B):
                    # - 同引擎 + raw 是 funasr 私有 sentence_info 结构 → 原 raw 重建路径
                    # - 其余 (跨引擎 / 无 raw / qwen3 raw 无 sentence_info) → schema 中立
                    #   segments 路径. 旧逻辑对 qwen3 行误走 raw 路径, 命中返回空 content.
                    srt_content = None
                    if not cross_hit and raw_result_json:
                        raw_result = json.loads(raw_result_json)
                        raw_data = (
                            raw_result[0]
                            if isinstance(raw_result, list) and raw_result
                            else raw_result
                        )
                        if isinstance(raw_data, dict) and "sentence_info" in raw_data:
                            srt_content = self._generate_srt_from_raw_result(raw_result)
                    if srt_content is None:
                        result_data = json.loads(result_json)
                        tr = TranscriptionResult(**result_data)
                        srt_content = self._segments_to_srt(tr.segments)
                    return {
                        "format": "srt",
                        "content": srt_content,
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "duration": duration,
                    }

                return None
        except (aiosqlite.Error, json.JSONDecodeError, ValidationError) as e:
            # 具名异常 (T-B): DB 不可用 / 坏缓存行 (非法 JSON / schema 不符) → 当 miss
            # 处理但必须留日志; 其余异常 (编程错误) 冒泡, 禁止 catch-all 静默吞.
            logger.error(
                f"获取缓存结果失败, 当 cache miss 处理: "
                f"file_hash={file_hash} engine={engine}: {type(e).__name__}: {e}"
            )
            return None

    async def save_result(
        self,
        result: TranscriptionResult,
        raw_result: dict = None,
        engine: str = DEFAULT_ENGINE,
    ):
        """
        保存转录结果到缓存

        Args:
            result: TranscriptionResult 对象
            raw_result: FunASR 原始输出（用于格式转换；其它引擎可能传 None）
            engine: ASR 引擎名，作为缓存 key 的一部分
        """
        if not config.transcription.cache_enabled:
            return

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # metadata 是请求级属性 (projected 等), 禁止入库 (T-D #9):
                # 否则缓存读出会继承上次请求的 projected, 污染后续请求.
                result_json = result.model_dump_json(exclude={"metadata"})
                raw_result_json = json.dumps(raw_result) if raw_result else None

                await db.execute(
                    """
                    INSERT OR REPLACE INTO transcription_cache
                    (file_hash, file_name, result, raw_result, duration, processing_time, engine)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.file_hash,
                        result.file_name,
                        result_json,
                        raw_result_json,
                        result.duration,
                        result.processing_time,
                        engine,
                    ),
                )
                await db.commit()
                logger.debug(f"转录结果已缓存: {result.file_hash} (engine={engine})")
        except Exception as e:
            logger.error(f"保存缓存结果失败: {e}")

    async def clean_old_cache(self):
        """清理缓存:① TTL 过期清理 ② size-cap 数量上限挤出(LRU)。

        每小时随 main._periodic_cleanup 调一次, 是防无限增长的兜底, 非实时硬限。

        ⚠️ 时间戳格式:DB 的 created_at/accessed_at 由 SQLite CURRENT_TIMESTAMP 写入,
        格式是空格分隔(`2026-06-17 12:00:00`)。比较 cutoff 必须用同格式
        (`strftime("%Y-%m-%d %H:%M:%S")`), 不能用 Python isoformat() 的 `T` 分隔——
        字符串比较下 `'T'(0x54) > ' '(0x20)` 会把 cutoff 同日、时间更晚(应保留)的行误删。
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=config.database.max_cache_days)
            # 与 DB CURRENT_TIMESTAMP 同格式(空格分隔), 见上方 docstring
            cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            max_count = config.database.max_cache_count

            async with aiosqlite.connect(self.db_path) as db:
                # ① TTL 清理
                cursor = await db.execute(
                    "DELETE FROM transcription_cache WHERE created_at < ?",
                    (cutoff_str,),
                )
                ttl_deleted = cursor.rowcount
                await db.commit()
                if ttl_deleted > 0:
                    logger.info(f"清理了 {ttl_deleted} 条过期缓存(TTL>{config.database.max_cache_days}天)")

                # ② size-cap:仍超数量上限则按 accessed_at LRU 挤(max_count<=0 关闭)
                size_deleted = 0
                if max_count and max_count > 0:
                    cur = await db.execute("SELECT COUNT(*) FROM transcription_cache")
                    total = (await cur.fetchone())[0]
                    overflow = total - max_count
                    if overflow > 0:
                        # accessed_at 读缓存时更新(true LRU), id ASC 做确定性 tie-breaker
                        cursor = await db.execute(
                            """DELETE FROM transcription_cache WHERE id IN (
                                   SELECT id FROM transcription_cache
                                   ORDER BY accessed_at ASC, id ASC
                                   LIMIT ?
                               )""",
                            (overflow,),
                        )
                        size_deleted = cursor.rowcount
                        await db.commit()
                        logger.info(
                            f"缓存数量超上限({total}>{max_count}), 按 LRU 挤出 {size_deleted} 条"
                        )
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")

    async def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM transcription_cache")
                total_count = (await cursor.fetchone())[0]

                today = datetime.now().date()
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM transcription_cache WHERE DATE(created_at) = ?",
                    (today.isoformat(),),
                )
                today_count = (await cursor.fetchone())[0]

                cursor = await db.execute(
                    "SELECT SUM(LENGTH(result)) FROM transcription_cache"
                )
                cache_size = (await cursor.fetchone())[0] or 0

                return {
                    "total_count": total_count,
                    "today_count": today_count,
                    "cache_size_mb": round(cache_size / 1024 / 1024, 2),
                    "projected_serves": self.projected_serves,
                }
        except Exception as e:
            logger.error(f"获取缓存统计信息失败: {e}")
            return {
                "total_count": 0,
                "today_count": 0,
                "cache_size_mb": 0,
                "projected_serves": self.projected_serves,
            }

    def _segments_to_srt(self, segments) -> str:
        """从 TranscriptionResult.segments 重建 SRT 字符串(引擎中立).

        用于跨引擎缓存命中: funasr 缓存被 qwen3 请求(或反之)时, raw_result 是另一引擎的
        私有结构(funasr 是 sentence_info, qwen3 是 asr_text/turns), 无法直接转 SRT.
        本函数走 schema 中立的 TranscriptionResult.segments 重建, 字节级与 FunASR
        _generate_srt_from_raw_result 输出对齐(Speaker{N}:文本 无空格).

        speaker=None (diarize=false 的 nospk 行, D8): 该段无说话人区分,
        渲染纯文本行 (无 "SpeakerN:" 前缀).

        实现委托给 result_projection.segments_to_srt_text (渲染点收拢).
        """
        return segments_to_srt_text(segments)

    def _generate_srt_from_raw_result(self, raw_result: dict) -> str:
        """从原始 FunASR 结果生成 SRT 格式字符串"""
        srt_lines = []

        if isinstance(raw_result, list) and len(raw_result) > 0:
            result_data = raw_result[0]
        elif isinstance(raw_result, dict):
            result_data = raw_result
        else:
            logger.warning(f"未知的结果格式: {type(raw_result)}")
            return ""

        if 'sentence_info' not in result_data:
            logger.warning("结果中没有sentence_info字段，可能转录失败")
            return ""

        sentences = result_data.get('sentence_info', [])

        for idx, sentence in enumerate(sentences, 1):
            start_ms = sentence.get('start', 0)
            end_ms = sentence.get('end', 0)
            start_time = self._ms_to_srt_time(start_ms)
            end_time = self._ms_to_srt_time(end_ms)
            text = sentence.get('text', '').strip()
            speaker_id = sentence.get('spk', 0)
            if isinstance(speaker_id, int):
                speaker = f"Speaker{speaker_id + 1}"
            else:
                speaker = "Speaker1"

            if text:
                srt_lines.append(f"{idx}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(f"{speaker}:{text}")
                srt_lines.append("")

        return "\n".join(srt_lines)

    def _ms_to_srt_time(self, milliseconds: int) -> str:
        """将毫秒转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
        seconds = milliseconds / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# 全局数据库管理器实例
db_manager = DatabaseManager()
