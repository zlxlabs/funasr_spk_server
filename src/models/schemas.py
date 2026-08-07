"""
数据模型定义
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
import re
import unicodedata

from pydantic import BaseModel, Field, field_validator, model_serializer
from pydantic_core import PydanticCustomError
from enum import Enum


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # 看门狗终态化：超 max_processing_seconds 仍卡在 PROCESSING 的任务被强制标记，
    # 使其进入终态从而能被内存清理 + 释放并发名额（高负载队列止血，2026-06-16）
    TIMED_OUT = "timed_out"


class WordTimestamp(BaseModel):
    """词级时间戳(MMS-300M CTC-FA 对齐输出, 绝对秒)

    增量挂进 TranscriptionSegment.words, 不替换段边界. 见
    docs/开发/2026-06-09-qwen3-词级时间戳-PoC计划.md.
    """
    text: str = Field(..., description="词文本")
    start: float = Field(..., description="开始时间（秒，绝对）")
    end: float = Field(..., description="结束时间（秒，绝对）")
    confidence: Optional[float] = Field(None, description="对齐置信度（可空）")

    model_config = {"protected_namespaces": ()}


class TranscriptionSegment(BaseModel):
    """转录片段"""
    start_time: float = Field(..., description="开始时间（秒）")
    end_time: float = Field(..., description="结束时间（秒）")
    text: str = Field(..., description="转录文本")
    # Optional (D8): null = 本次请求未做说话人区分 (diarize=false), 与"真只有一人
    # (Speaker1)"语义可区分. 内部 Segment(speaker:int) 永不为 None, null 只在出口转换层出现.
    speaker: Optional[str] = Field(None, description="说话人标识；diarize=false 时为 null（未区分）")
    # 词级时间戳(可选): word_align 开启且对齐成功时挂上, 否则 None(向后兼容).
    words: Optional[List[WordTimestamp]] = Field(None, description="词级时间戳列表")

    model_config = {"protected_namespaces": ()}


class TranscriptionResult(BaseModel):
    """转录结果"""
    task_id: str = Field(..., description="任务ID")
    file_name: str = Field(..., description="文件名")
    file_hash: str = Field(..., description="文件哈希值")
    duration: float = Field(..., description="音频时长（秒）")
    segments: List[TranscriptionSegment] = Field(..., description="转录片段列表")
    speakers: List[str] = Field(..., description="说话人列表")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    processing_time: float = Field(..., description="处理时长（秒）")
    error: Optional[str] = Field(None, description="错误信息")
    # E2 effective options 回显: serve 层组装 (engine/diarize/word_align/language/
    # projected), **不随缓存内容存取** — projected 是请求级属性, save_result 写库时
    # exclude 本字段, 防止缓存污染 (T-D #9).
    metadata: Optional[Dict[str, Any]] = Field(None, description="effective options 回显（serve 层组装，不入库）")

    model_config = {"protected_namespaces": ()}


class TranscribeOptions(BaseModel):
    """per-request 转录选项（E3 收拢）

    language（后续 diarize）收进一个结构, 整体穿透:
    schema → websocket_handler（含分片 session 回填）→ task_manager → 两套 pool
    → worker → transcribe.

    跨 file-based pool 进程边界时用 model_dump() 序列化成 dict 写任务文件,
    worker 端读 dict 并对缺失字段用本模型默认值兜底（老任务文件兼容）.
    """
    language: Optional[str] = Field(default=None, description="识别语言 ISO 码（chi/eng/jpn/kor…），驱动 word_align 词级时间戳语言；None 走 config 兜底")
    # diarize 开关（设计定案 D2）: 对外契约是 diarize=false ⇒ 响应不含说话人区分;
    # 各引擎自行决定达成方式（qwen3 真跳过 diarize+speaker 后处理省算力, funasr 出口投影）
    diarize: bool = Field(default=True, description="是否输出说话人区分；False 时 segments.speaker=null、SRT 无 SpeakerN: 前缀")
    # word_align 开关（2026-06-16 显存落地，决策 1A）: 已解析的 effective 值（确定 bool，非 Optional）。
    # 优先级链（请求 > config 兜底）在构造 options 前由 resolve_word_align 算一次，下游
    # transcribe/cache/metadata 全读此字段，不再各自读 config。JSON-only（SRT 不挂词，见 cache 2A）。
    word_align: bool = Field(default=False, description="是否输出词级时间戳（segment.words）；已解析的 effective 值")
    terms: List[str] = Field(default_factory=list, description="已验证的 ASR 专名术语")

    model_config = {"protected_namespaces": ()}

    @model_serializer(mode="wrap")
    def dump_transcription_options(self, handler):
        data = handler(self)
        if not self.terms:
            data.pop("terms", None)
        return data


def resolve_word_align(request_value: Optional[bool], config_default: bool) -> bool:
    """effective word_align 优先级单一事实源（决策 1A）。

    请求级 word_align 是 Optional[bool]：非 None 表示客户端显式指定（True/False 都压过
    config），None 表示未指定 → 跟随 server config 兜底。**所有构造 TranscribeOptions 的点
    （task_manager.create_task / handler 分片 session）都调本函数**，保证优先级规则不在多处 drift。

    ⚠️ 这里只解析"请求想不想要词"，不掺 output_format / 失败状态：
    - SRT 强制降级在 cache 层（compute_cache_engine AND output_format=='json'，决策 2A）
    - 对齐失败的 delivered=false 在 metadata 层（build_result_metadata）
    职责分离，本函数保持纯粹。
    """
    return request_value if request_value is not None else config_default


class TranscriptionTask(BaseModel):
    """转录任务"""
    task_id: str = Field(..., description="任务ID")
    file_name: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小（字节）")
    file_hash: str = Field(..., description="文件哈希值")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    progress: float = Field(default=0.0, description="进度（0-100）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    retry_count: int = Field(default=0, description="重试次数")
    error: Optional[str] = Field(None, description="错误信息")
    result: Optional[TranscriptionResult] = Field(None, description="转录结果")
    force_refresh: bool = Field(default=False, description="强制刷新缓存")
    output_format: str = Field(default="json", description="输出格式: json 或 srt")
    srt_content: Optional[str] = Field(None, description="SRT格式内容")
    engine: str = Field(default="funasr", description="ASR 引擎名（由 task_manager 根据 request.engine 或 default_engine 解析后填入）")
    # per-request 选项嵌套结构（D1）: 平铺 language 已删, options 是唯一 source of truth
    options: TranscribeOptions = Field(default_factory=TranscribeOptions, description="per-request 转录选项（language 等），整体穿透到 transcribe")

    model_config = {"protected_namespaces": ()}


class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    type: str = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    
    model_config = {"protected_namespaces": ()}


class AuthRequest(BaseModel):
    """认证请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    
    model_config = {"protected_namespaces": ()}


class AuthResponse(BaseModel):
    """认证响应"""
    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间（秒）")
    
    model_config = {"protected_namespaces": ()}


class FileUploadRequest(BaseModel):
    """文件上传请求"""
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小（字节）")
    file_hash: str = Field(..., description="文件哈希值")
    force_refresh: bool = Field(default=False, description="强制刷新缓存")
    output_format: str = Field(default="json", description="输出格式: json 或 srt")
    engine: Optional[str] = Field(default=None, description="ASR 引擎名（None 表示走 config.transcription.default_engine）")
    language: Optional[str] = Field(default=None, description="识别语言 ISO 码（chi/eng/jpn/kor…），驱动 word_align 词级时间戳语言；None 走 config 兜底")
    # diarize=false ⇒ 响应不含说话人区分（JSON speaker=null + SRT 无前缀, D8）.
    # 默认 true 完全向后兼容; 老 server Pydantic 忽略未知字段（部署顺序: server 先升级）.
    diarize: bool = Field(default=True, description="是否输出说话人区分（默认 True 向后兼容）")
    # word_align 请求级开关（2026-06-16 显存落地）: Optional[bool], None=未指定（跟随 config 兜底），
    # True/False=显式. 默认 None → 老客户端不传此字段行为零变化（resolve_word_align 走 config，
    # 默认关）. 老 server Pydantic 忽略未知字段（部署顺序: server 先升级、客户端后启用）.
    word_align: Optional[bool] = Field(default=None, description="是否输出词级时间戳；None=跟随 server 默认（默认关）")
    terms: List[str] = Field(default_factory=list, description="结构化 ASR 专名术语；服务端负责规范化")

    @field_validator("terms", mode="before")
    @classmethod
    def validate_transcription_terms(cls, raw_terms):
        """唯一入口：规范化术语并把限额失败编码为 invalid_terms。"""
        if raw_terms is None:
            raw_terms = []
        if not isinstance(raw_terms, (list, tuple)) or not all(
            isinstance(term, str) for term in raw_terms
        ):
            return raw_terms
        try:
            return normalize_transcription_terms(list(raw_terms))
        except TranscriptionTermsError as error:
            raise PydanticCustomError(
                "invalid_terms",
                "Invalid transcription terms: {reason}",
                {"reason": error.reason},
            ) from error

    model_config = {"protected_namespaces": ()}


class TranscriptionTermsError(ValueError):
    """术语规范化失败，reason 是稳定的协议错误枚举。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def normalize_transcription_terms(raw: Optional[List[str]]) -> List[str]:
    """规范化 ASR 术语：先检查 raw 限额，再 NFKC、trim、折叠空白、稳定去重。"""
    if raw is None:
        return []
    if len(raw) > 100:
        raise TranscriptionTermsError("too_many_raw_items")
    if any(len(term) > 256 for term in raw):
        raise TranscriptionTermsError("raw_item_too_long")

    effective: List[str] = []
    seen = set()
    for term in raw:
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", term).strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        effective.append(normalized)
    if len(effective) > 50:
        raise TranscriptionTermsError("too_many_terms")
    if any(len(term) > 64 for term in effective):
        raise TranscriptionTermsError("term_too_long")
    if sum(len(term) for term in effective) > 1024:
        raise TranscriptionTermsError("total_too_long")
    return effective


class FileUploadResponse(BaseModel):
    """文件上传响应"""
    task_id: str = Field(..., description="任务ID")
    upload_url: Optional[str] = Field(None, description="上传URL")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    
    model_config = {"protected_namespaces": ()}


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: float = Field(..., description="进度（0-100）")
    result: Optional[TranscriptionResult] = Field(None, description="转录结果")
    error: Optional[str] = Field(None, description="错误信息")
    
    model_config = {"protected_namespaces": ()}


class TaskStatusBatchItem(BaseModel):
    """批量状态查询的逐 id 项（异步轮询契约 TODOS #20）。

    一项对应一个被查询的 task_id，与单查 TaskStatusResponse 的关键差异：
    - COMPLETED 内联结果：JSON 走 result，**SRT 走 srt_content**
      （codex #11：TaskStatusResponse.result=TranscriptionResult 装不下 SRT 文本）。
    - 终态全集：completed/failed/timed_out/cancelled 都返终态（含 error），
      让客户端据此停轮询（codex #10：只盯 completed 会让失败任务无限轮询）。
    - PENDING/PROCESSING → result/srt_content/error 全 None（小帧）。
    - expired/not_found：status=None + error 标 "task_expired"/"task_not_found"
      （不整批失败；与单查 _send_task_status 的 expired vs not_found 语义对齐）。
    """
    task_id: str = Field(..., description="任务ID")
    status: Optional[TaskStatus] = Field(None, description="任务状态；expired/not_found 时为 None")
    progress: float = Field(default=0.0, description="进度（0-100）")
    result: Optional[TranscriptionResult] = Field(None, description="JSON 转录结果（仅 COMPLETED 且 output_format=json）")
    srt_content: Optional[str] = Field(None, description="SRT 文本（仅 COMPLETED 且 output_format=srt）")
    error: Optional[str] = Field(None, description="错误信息 / task_expired / task_not_found")

    model_config = {"protected_namespaces": ()}


class TaskStatusBatchResponse(BaseModel):
    """批量状态查询响应（异步轮询契约 TODOS #20）。

    防 N+1 拉取风暴 + TTL race：客户端单条 task_status_batch 一次性拿全集状态，
    完成项 result 随响应内联，不再逐 id 单查。task_ids 上限 50（服务端截断 + warn）。
    """
    items: List[TaskStatusBatchItem] = Field(..., description="逐 id 状态项")

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    task_id: Optional[str] = Field(None, description="关联任务 ID；连接级错误为 None")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
    
    model_config = {"protected_namespaces": ()}
