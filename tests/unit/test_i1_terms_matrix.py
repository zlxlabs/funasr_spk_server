"""I1 terms × diarize × word_align 的 FunASR fresh 出口矩阵。"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.schemas import (
    FileUploadRequest,
    TaskStatus,
    TranscriptionResult,
    TranscriptionSegment,
)


def _fresh_result(task_id):
    segment = TranscriptionSegment(
        start_time=0, end_time=1, text="hello", speaker="Speaker1",
    )
    return TranscriptionResult(
        task_id=task_id, file_name="matrix.wav", file_hash="matrix-hash",
        duration=1, segments=[segment], speakers=["Speaker1"], processing_time=0.1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terms", "diarize", "word_align"),
    [
        (terms, diarize, word_align)
        for terms in ([], ["Alpha"])
        for diarize in (True, False)
        for word_align in (True, False)
    ],
)
async def test_funasr_fresh_terms_matrix_effective_options_and_output(
    tmp_path, terms, diarize, word_align,
):
    from src.core.task_manager import TaskManager

    audio = tmp_path / "matrix.wav"
    audio.write_bytes(b"audio")
    request = FileUploadRequest(
        file_name=audio.name, file_size=5, file_hash="matrix-hash",
        engine="funasr", diarize=diarize, word_align=word_align, terms=terms,
    )
    manager = TaskManager()
    task = await manager.create_task(request, task_id=f"matrix-{diarize}-{word_align}-{bool(terms)}")
    task.file_path = str(audio)
    fake = MagicMock(transcribe=AsyncMock(return_value=(_fresh_result(task.task_id), [])))

    with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake), \
         patch("src.core.task_manager.db_manager") as db, \
         patch.object(manager, "_notify_task_progress", new=AsyncMock()), \
         patch.object(manager, "_notify_task_complete", new=AsyncMock()), \
         patch.object(manager, "_maybe_delete_task_file", new=AsyncMock()):
        db.get_cached_result = AsyncMock(return_value=None)
        db.save_result = AsyncMock()
        await manager._process_task(task.task_id)

    assert task.status is TaskStatus.COMPLETED
    options = fake.transcribe.await_args.kwargs["options"]
    assert options.terms == request.terms
    assert options.diarize is diarize
    assert options.word_align is word_align

    metadata = task.result.metadata
    assert metadata["word_align"] is False  # FunASR 不交付词级时间戳
    assert metadata["diarize"] is diarize
    if terms:
        assert metadata["context_applied"] is True
        assert metadata["terms_count"] == 1
    else:
        assert "context_applied" not in metadata
        assert "terms_count" not in metadata

    segment = task.result.segments[0]
    assert segment.words is None
    if diarize:
        assert task.result.speakers == ["Speaker1"]
        assert segment.speaker == "Speaker1"
    else:
        assert task.result.speakers == []
        assert segment.speaker is None
