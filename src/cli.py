"""一次性 CPU 转录入口：不加载服务配置，不启动池或服务。"""
import argparse
import asyncio
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys


def model_cache() -> str:
    """模型与虚拟环境分离，尊重调用者指定的 ModelScope 缓存。"""
    return os.environ.setdefault(
        "MODELSCOPE_CACHE", str(Path.home() / ".cache/funasr-cli/modelscope")
    )


async def transcribe(audio: Path | None):
    """预取与转录共享同一次初始化，直接复用内核 JSON 结果。"""
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    import torch
    from src.core.funasr_transcriber import FunASRTranscriber

    torch.set_num_threads(2)
    transcriber = FunASRTranscriber(config={
        "funasr": {
            "model": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "model_revision": "v2.0.4",
            "vad_model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "vad_model_revision": "v2.0.4",
            "punc_model": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            "punc_model_revision": "v2.0.4",
            "spk_model": "iic/speech_campplus_sv_zh-cn_16k-common",
            "spk_model_revision": "v2.0.2",
            "model_dir": model_cache(), "device": "cpu", "ncpu": 2,
            "device_priority": ["cpu"], "batch_size_s": 300,
            "disable_update": True, "disable_pbar": True,
            "vad_kwargs": {"ncpu": 2}, "punc_kwargs": {"ncpu": 2},
            "spk_kwargs": {"ncpu": 2},
        },
        "transcription": {"concurrency_mode": "lock"},
    })
    await transcriber.initialize()
    if audio is not None:
        result, _raw = await transcriber.transcribe(str(audio), task_id="cli")
        return result.model_dump(mode="json")


def main():
    """先验证参数与输入；第三方导入、加载、推理的 stdout 全部送 stderr。"""
    parser = argparse.ArgumentParser(description="单文件 CPU 转录，stdout 仅输出 JSON")
    parser.add_argument("input", nargs="?", type=Path, help="输入音频路径")
    parser.add_argument("--download-models", action="store_true", help="仅预取并加载模型")
    args = parser.parse_args()
    if (args.input is None) == (not args.download_models):
        parser.error("请提供 INPUT 或单独使用 --download-models")
    if args.input is not None and not args.input.is_file():
        raise FileNotFoundError(f"输入音频不存在或不是文件: {args.input}")
    model_cache()
    with redirect_stdout(sys.stderr):
        result = asyncio.run(transcribe(args.input))
    if not args.download_models:
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
