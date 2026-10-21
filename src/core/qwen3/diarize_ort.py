"""ORT 直 wrap diarize backend — pyannote-seg + TitaNet + Python clustering.

替代 sherpa-onnx 的 OfflineSpeakerDiarization, 拆组件用 Python onnxruntime 直跑,
pipeline 结构 1:1 移植 sherpa-onnx C++ 实现
(sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h):

1. pyannote-segmentation-3.0 sliding window, **每 chunk 独立 argmax** 出 multi-label
   (不跨 chunk 平均 logits — pyannote 的 speaker slot 是 chunk 局部的, 跨 chunk
   平均会把不同说话人混进同一 slot, 短音频下导致 under-detect, 见 2026-06-10 根因调查)
2. per-(chunk, local_speaker) 提取 TitaNet embedding (剔重叠帧, <10 活跃帧跳过)
3. FastClustering (cosine distance + complete linkage, scipy 复刻 sherpa C++)
4. cluster 重标 chunk 帧 → 全局 frame 网格 per-frame top-k → 段重建

为什么不走 sherpa-onnx GPU build:
sherpa-onnx CUDA build 的 C++ wrapper 跟 llama.cpp CUDA 撞 segfault (见 docs/开发/gpu加速/
2026-05-21-3060-CUDA移植与优化.md). ORT Python API 跟 llama.cpp CUDA 共存稳定 (PoC 通过).
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

# 延迟 import: librosa/scipy 只在 mel 计算时用, 不污染 module import 路径
def _scipy_hann_window(size: int) -> np.ndarray:
    from scipy.signal.windows import hann as _hann

    return _hann(size, sym=False).astype(np.float32)


def _librosa_mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    import librosa

    return librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    ).astype(np.float32)

# pyannote-segmentation-3.0 模型常量 (固定, 与模型 ONNX metadata 一致)
PYANNOTE_CHUNK_SAMPLES = 160000  # 10s @ 16k (window_size)
PYANNOTE_CHUNK_FRAMES = 589  # 每 chunk 输出 frame 数
PYANNOTE_NUM_SPEAKERS = 3  # max speakers per chunk
PYANNOTE_NUM_POWERSET_CLASSES = 7  # C(3,0)+C(3,1)+C(3,2) = 1+3+3
PYANNOTE_STEP_SAMPLES = 16000  # window_shift (sherpa 默认 = 0.1 * window_size)
PYANNOTE_RECEPTIVE_FIELD_SHIFT = 270  # 每 frame 对应的样点步长 (model metadata)
PYANNOTE_RECEPTIVE_FIELD_SIZE = 991  # 单 frame 感受野样点数 (model metadata)

# powerset class id → multi-label binary (3-speaker)
# class 0 = silence; 1/2/3 = 单 speaker; 4/5/6 = 两 speaker 同时活动
_POWERSET_MAP = np.array(
    [
        [0, 0, 0],  # silence
        [1, 0, 0],  # spk0
        [0, 1, 0],  # spk1
        [0, 0, 1],  # spk2
        [1, 1, 0],  # spk0+1
        [1, 0, 1],  # spk0+2
        [0, 1, 1],  # spk1+2
    ],
    dtype=np.int8,
)


def _iter_audio_chunks(
    audio: np.ndarray,
    chunk_samples: int = PYANNOTE_CHUNK_SAMPLES,
    step_samples: int = PYANNOTE_STEP_SAMPLES,
) -> Iterator[Tuple[int, np.ndarray]]:
    """切 audio 为重叠 chunks, yield (start_sample, chunk).

    保证 chunk shape == (chunk_samples,) (末尾不足时 zero pad). audio 为空时
    不 yield. audio 短于 chunk_samples 时 yield 一次 padded chunk.

    切法与 sherpa RunSpeakerSegmentationModel 等价: 整 chunk 步进 step_samples,
    余数不足一个 step 时补一个 zero-pad 的尾 chunk.
    """
    n = int(audio.shape[0])
    if n == 0:
        return
    start = 0
    while True:
        chunk = audio[start : start + chunk_samples]
        if chunk.shape[0] < chunk_samples:
            pad = np.zeros(chunk_samples - chunk.shape[0], dtype=audio.dtype)
            chunk = np.concatenate([chunk, pad])
        yield start, chunk
        if start + chunk_samples >= n:
            break
        start += step_samples


def _powerset_to_multilabel(
    powerset_logits: np.ndarray, num_speakers: int = PYANNOTE_NUM_SPEAKERS
) -> np.ndarray:
    """(..., K) powerset logits / probs → (..., S) multi-label binary.

    每 frame 取 argmax class id, 查表 _POWERSET_MAP 得到 multi-label.
    K=7, S=3 for pyannote-segmentation-3.0.
    """
    cls = np.argmax(powerset_logits, axis=-1)  # (...,)
    return _POWERSET_MAP[cls]  # (..., S)


def run_segmentation_chunks(
    audio: np.ndarray,
    ort_session,
    input_name: Optional[str] = None,
    chunk_samples: int = PYANNOTE_CHUNK_SAMPLES,
    step_samples: int = PYANNOTE_STEP_SAMPLES,
) -> Tuple[list[int], list[np.ndarray]]:
    """逐 chunk 跑 pyannote 分段模型, 每 chunk **独立** argmax 出 multi-label.

    sherpa parity 关键: 绝不跨 chunk 平均 logits — speaker slot 是 chunk 局部的,
    跨 chunk 融合属于错误用法 (会混叠不同说话人的活动).

    Returns:
        (starts, labels): starts[i] 是 chunk i 的起始样点;
        labels[i] 是 (chunk_frames, 3) int8 multi-label.
    """
    if input_name is None:
        input_name = ort_session.get_inputs()[0].name

    starts: list[int] = []
    labels: list[np.ndarray] = []
    for start, chunk in _iter_audio_chunks(audio, chunk_samples, step_samples):
        x = chunk.reshape(1, 1, -1).astype(np.float32)
        raw = ort_session.run(None, {input_name: x})[0]
        # raw shape (1, frames, K); 去 batch dim, per-chunk argmax
        labels.append(_powerset_to_multilabel(raw[0]).astype(np.int8))
        starts.append(start)
    return starts, labels


def compute_speakers_per_frame(
    chunk_labels: list[np.ndarray],
    window_size: int = PYANNOTE_CHUNK_SAMPLES,
    window_shift: int = PYANNOTE_STEP_SAMPLES,
    receptive_field_shift: int = PYANNOTE_RECEPTIVE_FIELD_SHIFT,
) -> np.ndarray:
    """全局 frame 网格上的 per-frame 说话人数 (sherpa ComputeSpeakersPerFrame).

    每 chunk 在自己覆盖的 frame 区间贡献 rowwise speaker 数, 重叠区按覆盖
    chunk 数取平均后 +0.5 四舍五入. 用 **原始** multi-label (不剔重叠帧).

    Returns:
        (num_frames,) int32, num_frames = (window + (n_chunks-1)*shift) // rf_shift + 1.
    """
    num_chunks = len(chunk_labels)
    num_frames = (
        window_size + (num_chunks - 1) * window_shift
    ) // receptive_field_shift + 1

    count = np.zeros(num_frames, dtype=np.float64)
    weight = np.zeros(num_frames, dtype=np.float64)
    for i, label in enumerate(chunk_labels):
        start = int(i * window_shift / receptive_field_shift + 0.5)
        end = min(start + label.shape[0], num_frames)
        if end <= start:
            continue
        count[start:end] += label[: end - start].sum(axis=1)
        weight[start:end] += 1.0

    return (count / (weight + 1e-12) + 0.5).astype(np.int32)


def exclude_overlap(label: np.ndarray) -> np.ndarray:
    """同帧 ≥2 speaker 活跃的 frame 整帧清零 (sherpa ExcludeOverlap).

    重叠帧语音是混合信号, 不能用来提 speaker embedding. 返回副本, 不改原数组.
    """
    out = label.copy()
    out[label.sum(axis=1) >= 2] = 0
    return out


def get_chunk_speaker_sample_indexes(
    chunk_labels: list[np.ndarray],
    window_size: int = PYANNOTE_CHUNK_SAMPLES,
    window_shift: int = PYANNOTE_STEP_SAMPLES,
    min_active_frames: int = 10,
) -> Tuple[list[Tuple[int, int]], list[list[Tuple[int, int]]]]:
    """每个 (chunk, local_speaker) 的活跃采样区间 (sherpa GetChunkSpeakerSampleIndexes).

    先剔重叠帧, 再按 speaker slot 扫连续活跃 frame 段, 帧 index 按
    frame/num_frames*window_size + chunk_offset 映射回样点. 总活跃帧
    < min_active_frames 的 (chunk, speaker) 整对跳过 (sherpa 固定 10 帧).

    Returns:
        (pairs, sample_ranges): pairs[i] = (chunk_idx, local_speaker);
        sample_ranges[i] = [(start_sample, end_sample), ...] 该对的全部活跃区间.
    """
    pairs: list[Tuple[int, int]] = []
    sample_ranges: list[list[Tuple[int, int]]] = []

    for ci, label in enumerate(chunk_labels):
        lab = exclude_overlap(label)
        num_frames = lab.shape[0]
        sample_offset = ci * window_shift
        for spk in range(lab.shape[1]):
            idx = np.flatnonzero(lab[:, spk])
            if idx.size < min_active_frames:
                continue
            # 连续活跃 run 边界 (end 为 exclusive 帧 index)
            breaks = np.flatnonzero(np.diff(idx) > 1)
            run_starts = np.concatenate([[idx[0]], idx[breaks + 1]])
            run_ends = np.concatenate([idx[breaks] + 1, [idx[-1] + 1]])

            segs: list[Tuple[int, int]] = []
            for a, b in zip(run_starts, run_ends):
                # sherpa: 活跃到 chunk 末尾时 end 用 num_frames-1 映射
                b_eff = int(b) if b < num_frames else num_frames - 1
                segs.append(
                    (
                        int(int(a) / num_frames * window_size) + sample_offset,
                        int(b_eff / num_frames * window_size) + sample_offset,
                    )
                )
            pairs.append((ci, spk))
            sample_ranges.append(segs)

    return pairs, sample_ranges


def relabel_chunks(
    chunk_labels: list[np.ndarray],
    pair_to_cluster: dict[Tuple[int, int], int],
    num_clusters: int,
) -> list[np.ndarray]:
    """按聚类结果把 chunk 局部 slot 重标成全局 cluster 列 (sherpa ReLabel).

    用 **原始** multi-label (含重叠帧). 不在映射里的 (chunk, slot)
    (embedding 阶段被跳过/过滤) 直接丢弃.

    Returns:
        list of (chunk_frames, num_clusters) int8.
    """
    out: list[np.ndarray] = []
    for ci, label in enumerate(chunk_labels):
        new = np.zeros((label.shape[0], num_clusters), dtype=np.int8)
        for spk in range(label.shape[1]):
            cluster = pair_to_cluster.get((ci, spk))
            if cluster is None or not (0 <= cluster < num_clusters):
                continue
            new[label[:, spk] == 1, cluster] = 1
        out.append(new)
    return out


def compute_speaker_count_grid(
    relabeled: list[np.ndarray],
    num_samples: int,
    window_size: int = PYANNOTE_CHUNK_SAMPLES,
    window_shift: int = PYANNOTE_STEP_SAMPLES,
    receptive_field_shift: int = PYANNOTE_RECEPTIVE_FIELD_SHIFT,
) -> np.ndarray:
    """重标后的 chunk 帧投到全局 frame 网格累加 (sherpa ComputeSpeakerCount).

    末尾存在 zero-pad 尾 chunk 时截断到音频真实 frame 数.

    Returns:
        (num_frames, num_clusters) int32 — 每帧每 cluster 被多少个 chunk 标活跃.
    """
    num_chunks = len(relabeled)
    num_frames = (
        window_size + (num_chunks - 1) * window_shift
    ) // receptive_field_shift + 1
    num_clusters = relabeled[0].shape[1]

    count = np.zeros((num_frames, num_clusters), dtype=np.int32)
    for i, lab in enumerate(relabeled):
        start = int(i * window_shift / receptive_field_shift + 0.5)
        end = min(start + lab.shape[0], num_frames)
        if end <= start:
            continue
        count[start:end] += lab[: end - start]

    # C++ 负数取模语义: num_samples <= window_size 时恒为 False
    has_last_chunk = (
        num_samples > window_size
        and ((num_samples - window_size) % window_shift) > 0
    )
    if has_last_chunk:
        last_frame = min(num_samples // receptive_field_shift, num_frames - 1)
        count = count[: last_frame + 1]
    return count


def finalize_labels(
    count: np.ndarray, speakers_per_frame: np.ndarray
) -> np.ndarray:
    """每帧按 count 取 top-k cluster (k = speakers_per_frame) (sherpa FinalizeLabels).

    Returns:
        (num_frames, num_clusters) int8 binary.
    """
    num_frames, num_clusters = count.shape
    k_per_frame = np.clip(
        speakers_per_frame[:num_frames].astype(np.int32), 0, num_clusters
    )

    # 每行按 count 降序的排名 (稳定排序, 平局按列 index 升序 — deterministic)
    order = np.argsort(-count, axis=1, kind="stable")  # (T, K) 列 index 按值降序
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order, np.broadcast_to(np.arange(num_clusters), order.shape).copy(), axis=1
    )
    return (ranks < k_per_frame[:, None]).astype(np.int8)


def labels_to_turns(
    final_labels: np.ndarray,
    receptive_field_shift: int = PYANNOTE_RECEPTIVE_FIELD_SHIFT,
    receptive_field_size: int = PYANNOTE_RECEPTIVE_FIELD_SIZE,
    sample_rate: int = 16000,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
) -> list[dict]:
    """(num_frames, num_clusters) binary → turn list (sherpa ComputeResult + MergeSegments).

    帧→秒映射: t = frame * rf_shift/sr + 0.5*rf_size/sr. 每 speaker 独立扫
    连续活跃段; 同 speaker 相邻段 gap ≤ min_duration_off 合并; 时长
    ≤ min_duration_on 的段丢弃 (sherpa 是严格 >).

    Returns:
        按 start 排序的 [{"start", "end", "speaker"}, ...].
    """
    scale = receptive_field_shift / sample_rate
    scale_offset = 0.5 * receptive_field_size / sample_rate

    num_frames, num_clusters = final_labels.shape
    turns: list[dict] = []
    for spk in range(num_clusters):
        idx = np.flatnonzero(final_labels[:, spk])
        if idx.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        run_starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        run_ends = np.concatenate([idx[breaks] + 1, [idx[-1] + 1]])

        segs: list[list[float]] = []
        for a, b in zip(run_starts, run_ends):
            # sherpa: 活跃到最后一帧时 end 用 num_frames-1 映射
            b_eff = int(b) if b < num_frames else num_frames - 1
            segs.append(
                [float(a) * scale + scale_offset, float(b_eff) * scale + scale_offset]
            )

        # 同 speaker 相邻段 gap ∈ (0, min_duration_off] 合并 (sherpa Segment::Merge)
        merged: list[list[float]] = []
        for s in segs:
            if merged and 0 < s[0] - merged[-1][1] <= min_duration_off:
                merged[-1][1] = s[1]
            else:
                merged.append(s)

        for s in merged:
            if s[1] - s[0] > min_duration_on:
                turns.append({"start": s[0], "end": s[1], "speaker": spk})

    turns.sort(key=lambda x: (x["start"], x["end"]))
    return turns


def compute_titanet_log_mel(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_window_size: int = 400,
    n_window_stride: int = 160,
    n_fft: int = 512,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    preemph: float = 0.97,
    log_zero_guard: float = 2**-24,
    norm_eps: float = 1e-5,
) -> np.ndarray:
    """NeMo TitaNet 风格 80-band log-mel preprocessing (numpy + librosa/scipy).

    步骤跟 NeMo AudioToMelSpectrogramPreprocessor 对齐:
    1. preemphasis: y[t] = x[t] - preemph * x[t-1]  (preemph=0 跳过)
    2. STFT: n_fft=512, hann window n_window_size=400 zero-pad 到 n_fft, hop=160,
       center=True (reflect pad)
    3. power spectrum (|spec|^2)
    4. mel filterbank: librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax=sr/2)
    5. log(mel + log_zero_guard)
    6. per_feature normalize: 沿 time axis 做 z-score, 每个 mel band 独立

    输入: audio (T,) float32 @ sample_rate
    输出: (n_mels, T_mel) float32. ONNX 期望 (B, n_mels, T_mel), 调用方加 batch dim.
    """
    if fmax is None:
        fmax = sample_rate / 2

    audio = np.asarray(audio, dtype=np.float32)

    # 1. preemphasis
    if preemph and preemph != 0.0:
        audio = np.concatenate(
            [[audio[0]], audio[1:] - preemph * audio[:-1]]
        ).astype(np.float32)

    # 2. center=True reflect pad
    pad = n_fft // 2
    if audio.shape[0] < n_fft:
        audio = np.concatenate(
            [audio, np.zeros(n_fft - audio.shape[0], dtype=np.float32)]
        )
    audio_padded = np.pad(audio, pad, mode="reflect")

    # 切 STFT frames
    n_frames = 1 + (audio_padded.shape[0] - n_fft) // n_window_stride
    if n_frames < 1:
        n_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        audio_padded,
        shape=(n_frames, n_fft),
        strides=(audio_padded.strides[0] * n_window_stride, audio_padded.strides[0]),
    ).copy()

    # hann window n_window_size, zero-pad 到 n_fft
    win = _scipy_hann_window(n_window_size)
    if n_window_size < n_fft:
        win = np.pad(win, (0, n_fft - n_window_size)).astype(np.float32)
    windowed = frames * win  # (n_frames, n_fft)

    # 3. STFT power
    spec = np.fft.rfft(windowed, n=n_fft, axis=-1)
    mag_power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    # 4. mel filterbank
    mel_basis = _librosa_mel_filterbank(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )  # (n_mels, n_fft//2 + 1)
    mel_spec = mag_power @ mel_basis.T  # (n_frames, n_mels)

    # 5. log
    log_mel = np.log(mel_spec + log_zero_guard).astype(np.float32)

    # 6. per_feature normalize (沿 time axis z-score, 每个 band 独立)
    mean = log_mel.mean(axis=0, keepdims=True)
    std = log_mel.std(axis=0, keepdims=True)
    log_mel = (log_mel - mean) / (std + norm_eps)

    # 转 (n_mels, n_frames) 跟 NeMo / sherpa 期望对齐
    return log_mel.T.astype(np.float32)


def compute_titanet_embedding(
    audio: np.ndarray,
    ort_session,
    input_name: Optional[str] = None,
    length_input_name: Optional[str] = None,
) -> np.ndarray:
    """跑 TitaNet ONNX embedding 单 audio segment.

    Args:
        audio: (T,) float32 16kHz mono. 期望 >= 0.3s 才有意义 embedding.
        ort_session: 已加载的 onnxruntime InferenceSession 或 mock.
        input_name: mel 输入名; None 则用 session.get_inputs()[0].name.
        length_input_name: T_mel length 输入名 (sherpa nemo-titanet 有 2 inputs);
            None 则按 inputs[1].name 自动取.

    Returns:
        (192,) float32 L2-normalized embedding. 全 0 audio 时 norm=0, 返回原始
        embedding 不再 normalize (避免除零 NaN).
    """
    mel = compute_titanet_log_mel(audio)  # (80, T_mel)
    mel_in = mel[np.newaxis, :, :].astype(np.float32)  # (1, 80, T_mel)

    inputs = ort_session.get_inputs()
    in_name = input_name or inputs[0].name
    feed: dict = {in_name: mel_in}
    if len(inputs) >= 2:
        ln_name = length_input_name or inputs[1].name
        feed[ln_name] = np.array([mel_in.shape[-1]], dtype=np.int64)

    # NeMo TitaNet ONNX 输出 [logits (B,16681), embs (B,192)] — 必须显式取 embs.
    # 取 outputs[0] 会拿到训练集分类 logits, 聚类离散度大 (主说话人被劈卫星簇).
    output_names = [o.name for o in ort_session.get_outputs()]
    if "embs" in output_names:
        out = ort_session.run(["embs"], feed)[0]
    else:
        out = ort_session.run(None, feed)[0]
    emb = np.asarray(out).flatten().astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if norm > 1e-12:
        emb = emb / norm
    return emb


def fast_clustering(
    embeddings: np.ndarray,
    num_clusters: Optional[int] = None,
    threshold: float = 0.9,
    method: str = "complete",
) -> np.ndarray:
    """sherpa-onnx FastClustering 兼容: cosine distance + agglomerative linkage.

    Args:
        embeddings: (N, D) float32, 建议 L2-normalized.
        num_clusters: 固定簇数; None 时按 threshold 截断.
        threshold: distance > threshold 时停止合并. cosine distance ∈ [0, 2].
        method: scipy.cluster.hierarchy.linkage method.
            - "complete" (默认): 最远距离合并, 与 sherpa HCLUST_METHOD_COMPLETE 一致
            - "average" / "single": 仅实验用

    Returns:
        (N,) int32 labels in [0, K) — 0-based 连续.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = int(embeddings.shape[0])
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.array([0], dtype=np.int32)

    dist = pdist(embeddings.astype(np.float64), metric="cosine")
    Z = linkage(dist, method=method)
    if num_clusters is not None:
        labels = fcluster(Z, t=num_clusters, criterion="maxclust")
    else:
        labels = fcluster(Z, t=threshold, criterion="distance")
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq)}
    return np.array([remap[v] for v in labels], dtype=np.int32)


# ==================== ORT session cache ====================

_PYANNOTE_SESSION_CACHE: dict[str, object] = {}
_TITANET_SESSION_CACHE: dict[str, object] = {}


def _default_providers() -> list:
    """优先 CUDA → CoreML(Mac) → CPU. ORT 自动 fallback 到第一个可用."""
    return ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]


def _get_pyannote_session(model_path: str, providers: Optional[list] = None):
    """加载 / cache pyannote-seg ONNX session (按 model_path key)."""
    if model_path in _PYANNOTE_SESSION_CACHE:
        return _PYANNOTE_SESSION_CACHE[model_path]
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=providers or _default_providers())
    _PYANNOTE_SESSION_CACHE[model_path] = sess
    return sess


def _get_titanet_session(model_path: str, providers: Optional[list] = None):
    if model_path in _TITANET_SESSION_CACHE:
        return _TITANET_SESSION_CACHE[model_path]
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=providers or _default_providers())
    _TITANET_SESSION_CACHE[model_path] = sess
    return sess


def reset_session_cache() -> None:
    """测试 helper / 进程结束清理用."""
    _PYANNOTE_SESSION_CACHE.clear()
    _TITANET_SESSION_CACHE.clear()


def run_diarization_ort_cuda(
    audio_path: str,
    segmentation_model: str,
    embedding_model: str,
    num_speakers: Optional[int] = None,
    cluster_threshold: float = 0.9,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
    providers: Optional[list] = None,
    min_embedding_audio_sec: float = 0.3,
) -> list[dict]:
    """ORT 直 wrap diarize 端到端 (sherpa pipeline 忠实移植).

    跟 src.core.qwen3.diarize.run_diarization (sherpa-onnx) 行为 parity, 输出 schema
    一致: [{"start": float, "end": float, "speaker": int}, ...]

    Mac/Linux 通用 — providers 没传时按 CUDA → CoreML → CPU fallback. CUDA 平台
    走 GPU, mac 上 CoreML/CPU 也能跑 (慢但功能正确), 给 dev 阶段 debug 方便.
    """
    # 复用 sherpa 的 audio loading 跟 fallback 链 (libsndfile + librosa)
    from src.core.qwen3.diarize import _load_audio_mono_16k

    audio, _ = _load_audio_mono_16k(audio_path)
    n = int(audio.shape[0])

    seg_sess = _get_pyannote_session(segmentation_model, providers=providers)
    emb_sess = _get_titanet_session(embedding_model, providers=providers)

    # 1. 逐 chunk 分段 (每 chunk 独立 argmax — 不跨 chunk 平均)
    _starts, chunk_labels = run_segmentation_chunks(audio, seg_sess)
    if not chunk_labels:
        return []

    # 单 chunk 特例 (音频 ≤ 10s 窗): 不聚类, chunk 局部 slot 直接当输出 speaker
    # (sherpa HandleOneChunkSpecialCase; n <= window 时 C++ 端恒不截断)
    if len(chunk_labels) == 1:
        return labels_to_turns(
            chunk_labels[0],
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

    # 2. 全局 per-frame 说话人数 (用原始 multi-label)
    speakers_per_frame = compute_speakers_per_frame(chunk_labels)
    if int(speakers_per_frame.max()) == 0:
        return []

    # 3. per-(chunk, local_speaker) 提取 TitaNet embedding
    sample_rate = 16000
    min_samples = int(min_embedding_audio_sec * sample_rate)
    pairs, sample_ranges = get_chunk_speaker_sample_indexes(chunk_labels)

    embeddings: list[np.ndarray] = []
    valid_pairs: list[Tuple[int, int]] = []
    for pair, segs in zip(pairs, sample_ranges):
        pieces = [audio[a : min(b, n)] for a, b in segs if min(b, n) > a]
        if not pieces:
            continue
        seg_audio = np.concatenate(pieces)
        if seg_audio.shape[0] < min_samples:
            continue
        emb = compute_titanet_embedding(seg_audio, emb_sess)
        if np.isnan(emb).any():
            # sherpa: NaN embedding 的 (chunk, speaker) 对直接剔除
            continue
        embeddings.append(emb)
        valid_pairs.append(pair)

    if not embeddings:
        return []

    # 4. FastClustering 跨 chunk 重组 global speaker。sherpa 约定 <= 0 表示自动聚类，
    # 不能将 0 / 负数传给 scipy 作为 maxclust 的固定簇数。
    emb_matrix = np.stack(embeddings).astype(np.float32)
    if num_speakers is not None and num_speakers <= 0:
        num_speakers = None
    cluster_labels = fast_clustering(
        emb_matrix,
        num_clusters=num_speakers,
        threshold=cluster_threshold,
    )
    num_clusters = int(cluster_labels.max()) + 1
    pair_to_cluster = {
        pair: int(c) for pair, c in zip(valid_pairs, cluster_labels)
    }

    # 5. cluster 重标 chunk 帧 → 全局网格 top-k → 段重建
    relabeled = relabel_chunks(chunk_labels, pair_to_cluster, num_clusters)
    count = compute_speaker_count_grid(relabeled, num_samples=n)
    final = finalize_labels(count, speakers_per_frame)
    return labels_to_turns(
        final,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
