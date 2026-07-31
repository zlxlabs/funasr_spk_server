#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""演示 FunASR 段合并的两层结构（issue #1 之后）。

1. 引擎层：``_parse_and_merge_segments`` 产出**句级** canonical（不再合并）
2. serve 投影层：``merge_segments_view`` 做同说话人相邻句合并 + max_span cap

本脚本不加载 FunASR 模型，仅用 mock sentence_info 走解析 + 投影路径。
"""

import json
import sys
from pathlib import Path

# tests/manual/core/ → 项目根
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.funasr_transcriber import FunASRTranscriber
from src.core.result_projection import merge_segments_view


def create_mock_funasr_result():
    """创建模拟的 FunASR 结果 - 基于真实日志数据"""
    return [{
        'key': 'spk_extract',
        'text': '好，欢迎收听本期的创业内幕。我是主持人lily.本期我们请到的嘉宾来自于国内首家自主研发云CAD公司卡伦特的技术合伙人李荣璐lif李总跟大家打个招呼吧。大家好，我叫李荣录。复旦计算机系极其专业的博士，曾经在这个auto desk担任首位的华人首席工程师，也在这个SAP blackboard等这些公司呢担任首席架程师，然后呢也有将近二十年的软件开发和管理经验。嗯，您当年为什么选择加入卡伦特，这是一个很有意思的问题啊，当年我在复旦读博的时候，确实赶上了人工智能的一个。',
        'timestamp': [[50, 290], [310, 410], [410, 590]],  # 简化版
        'sentence_info': [
            # Speaker1 的连续句子
            {'text': '好，', 'start': 50, 'end': 290, 'timestamp': [[50, 290]], 'spk': 0},
            {'text': '欢迎收听本期的创业内幕。', 'start': 310, 'end': 1710, 'timestamp': [[310, 410]], 'spk': 0},
            {'text': '我是主持人 lily。', 'start': 1710, 'end': 2670, 'timestamp': [[1710, 1830]], 'spk': 0},
            {'text': '本期我们请到的嘉宾来自于国内首家自主研发云 CAD 公司卡伦特的技术合伙人李荣璐 lif 李总跟大家打个招呼吧。', 'start': 2950, 'end': 12490, 'timestamp': [[2950, 3090]], 'spk': 0},

            # Speaker2 的连续句子
            {'text': '大家好，', 'start': 12770, 'end': 13330, 'timestamp': [[12770, 12910]], 'spk': 1},
            {'text': '我叫李荣录。', 'start': 13550, 'end': 14410, 'timestamp': [[13550, 13690]], 'spk': 1},
            {'text': '复旦计算机系极其专业的博士，', 'start': 14550, 'end': 16970, 'timestamp': [[14550, 14710]], 'spk': 1},
            {'text': '曾经在这个 auto desk 担任首位的华人首席工程师，', 'start': 17330, 'end': 21710, 'timestamp': [[17330, 17450]], 'spk': 1},
            {'text': '也在这个 SAP blackboard 等这些公司呢担任首席架程师，', 'start': 21950, 'end': 26790, 'timestamp': [[21950, 22130]], 'spk': 1},
            {'text': '然后呢也有将近二十年的软件开发和管理经验。', 'start': 27270, 'end': 31050, 'timestamp': [[27270, 27390]], 'spk': 1},
            {'text': '嗯，', 'start': 31430, 'end': 31670, 'timestamp': [[31430, 31670]], 'spk': 1},

            # Speaker1 再次发言
            {'text': '您当年为什么选择加入卡伦特，', 'start': 32110, 'end': 34010, 'timestamp': [[32110, 32330]], 'spk': 0},

            # Speaker2 继续回答
            {'text': '这是一个很有意思的问题啊，', 'start': 34010, 'end': 36150, 'timestamp': [[34010, 34250]], 'spk': 1},
            {'text': '当年我在复旦读博的时候，', 'start': 36150, 'end': 38070, 'timestamp': [[36150, 36390]], 'spk': 1},
            {'text': '确实赶上了人工智能的一个，', 'start': 38070, 'end': 39960, 'timestamp': [[38070, 38230]], 'spk': 1},
        ]
    }]


def _max_span(segments) -> float:
    if not segments:
        return 0.0
    return max(seg.end_time - seg.start_time for seg in segments)


def test_merge_function():
    """演示引擎句级 canonical + serve 投影层 merge。"""
    print("=== 引擎层句级 canonical + serve 层 merge 投影 ===\n")

    # 不 initialize：只调解析方法，不加载模型
    transcriber = FunASRTranscriber()
    mock_result = create_mock_funasr_result()

    # 第一层：引擎解析 → 句级段（canonical，不再合并）
    print("1. 引擎层 _parse_and_merge_segments → 句级 canonical:")
    sentence_segments = transcriber._parse_and_merge_segments(mock_result)
    print(f"   句级段数: {len(sentence_segments)}")
    for i, seg in enumerate(sentence_segments[:5]):
        print(
            f"   {i + 1}. [{seg.speaker}] {seg.start_time}s-{seg.end_time}s: "
            f"{seg.text[:30]}..."
        )
    if len(sentence_segments) > 5:
        print(f"   ... 还有 {len(sentence_segments) - 5} 个片段")

    # 第二层：serve 投影合并（gap + max_span cap）
    print("\n2. serve 投影 merge_segments_view(gap=3.0, max_span=120):")
    merged_capped = merge_segments_view(
        sentence_segments, gap_sec=3.0, max_span_sec=120.0
    )
    print(f"   合并后段数: {len(merged_capped)}")
    print(f"   最长段时长: {_max_span(merged_capped):.2f}s")
    for i, seg in enumerate(merged_capped):
        dur = seg.end_time - seg.start_time
        print(
            f"   {i + 1}. [{seg.speaker}] {seg.start_time}s-{seg.end_time}s "
            f"({dur:.1f}s): {seg.text[:50]}..."
        )

    # 对比：max_span_sec=0 表示无上限
    print("\n3. 对比 cap 作用: max_span_sec=0（无上限） vs 120:")
    merged_uncapped = merge_segments_view(
        sentence_segments, gap_sec=3.0, max_span_sec=0
    )
    print(
        f"   无上限: 段数={len(merged_uncapped)}, "
        f"最长={_max_span(merged_uncapped):.2f}s"
    )
    print(
        f"   cap=120: 段数={len(merged_capped)}, "
        f"最长={_max_span(merged_capped):.2f}s"
    )
    if len(merged_uncapped) != len(merged_capped):
        print("   → cap 截断了过长合并，段数更多 / 最长段更短（或相等）")
    else:
        print("   → 本 mock 数据未触发 120s cap（最长段本身 < 120s）")

    # 保存合并后的结果（cap=120 视图）
    output_data = {
        "merge_test_result": {
            "sentence_level_segments": len(sentence_segments),
            "merged_segments_cap120": len(merged_capped),
            "merged_segments_no_cap": len(merged_uncapped),
            "max_span_cap120": round(_max_span(merged_capped), 2),
            "max_span_no_cap": round(_max_span(merged_uncapped), 2),
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "duration": round(seg.end_time - seg.start_time, 2),
                }
                for seg in merged_capped
            ],
        }
    }

    output_path = project_root / "tests" / "output" / "merge_test_result.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 合并测试结果已保存到: {output_path}")

    # 说话人维度的合并效果
    print("\n4. 合并效果验证:")
    s1_sent = [s for s in sentence_segments if s.speaker == "Speaker1"]
    s2_sent = [s for s in sentence_segments if s.speaker == "Speaker2"]
    s1_merged = [s for s in merged_capped if s.speaker == "Speaker1"]
    s2_merged = [s for s in merged_capped if s.speaker == "Speaker2"]
    print(f"   Speaker1: {len(s1_sent)} 句级 → {len(s1_merged)} 合并后")
    print(f"   Speaker2: {len(s2_sent)} 句级 → {len(s2_merged)} 合并后")

    return merged_capped


def main():
    print("=" * 60)
    print("FunASR 两层段结构演示（句级 canonical + serve merge）")
    print("=" * 60)

    try:
        result = test_merge_function()
        print(f"\n测试完成！serve 投影后共 {len(result)} 个片段。")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
