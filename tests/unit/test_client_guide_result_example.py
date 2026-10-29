"""客户端指南的 JSON 示例必须与真实结果序列化契约一致。"""
import json
from pathlib import Path

from src.models.schemas import TranscriptionResult, TranscriptionSegment


def test_json_result_example_matches_producer():
    """按小节标题抽取示例，对比生产者实际输出的键及关键类型。"""
    guide = (Path(__file__).resolve().parents[2] / "docs/使用/客户端交互指南.md").read_text()
    section = guide.split("### JSON格式输出", 1)[1].split("### SRT格式输出", 1)[0]
    example = json.loads(section.split("```json\n", 1)[1].split("```", 1)[0])
    payload = TranscriptionResult(
        task_id="contract-task",
        file_name="example.wav",
        file_hash="example-hash",
        duration=5.2,
        segments=[TranscriptionSegment(
            start_time=0.0, end_time=5.2, text="分段文本", speaker="Speaker1",
        )],
        speakers=["Speaker1"],
        processing_time=1.0,
    ).model_dump(mode="json")

    assert set(example) == set(payload)
    assert example["segments"], "示例必须包含至少一个 segment"
    for segment in example["segments"]:
        assert set(segment) == set(payload["segments"][0])
        assert isinstance(segment["start_time"], (int, float))
        assert isinstance(segment["end_time"], (int, float))
        assert isinstance(segment["text"], str)
        assert isinstance(segment["speaker"], str)
    assert isinstance(example["speakers"], type(payload["speakers"]))
    assert example["speakers"], "示例必须包含至少一个说话人"
    assert all(isinstance(speaker, type(payload["speakers"][0])) for speaker in example["speakers"])
