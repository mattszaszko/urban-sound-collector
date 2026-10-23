"""DASM window preprocessing: HPF, peak-norm, 32 kHz resample, log-mel (+ optional bn0)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from core.audio_constants import CAPTURE_SAMPLE_RATE
from core.dasm_onnx import DEFAULT_DASM_DIR, dasm_model_paths
from core.yamnet_preprocess import (
    DEFAULT_HPF_HZ,
    DEFAULT_HPF_ORDER,
    apply_butter_hpf,
)

# Match HTS-AT CLAPAudioCfp / Detect Any Sound text_query feature config.
DASM_SAMPLE_RATE = 32_000
DASM_AUDIO_SECONDS = 10.0
DASM_N_FFT = 1024
DASM_HOP = 320
DASM_N_MELS = 64
DASM_FMIN = 50.0
DASM_FMAX = 14_000.0
# center=True STFT on 10 s @ 32 kHz / hop 320 → 1001 frames (spike export shape).
DASM_MEL_FRAMES = 1001


def _resample(pcm: np.ndarray, *, src_rate: int, dst_rate: int) -> np.ndarray:
    mono = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if mono.size == 0 or src_rate == dst_rate:
        return mono
    gcd = math.gcd(src_rate, dst_rate)
    return resample_poly(mono, dst_rate // gcd, src_rate // gcd).astype(np.float32, copy=False)


def pad_or_truncate(pcm: np.ndarray, target_samples: int) -> np.ndarray:
    mono = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if mono.size == target_samples:
        return mono
    if mono.size > target_samples:
        return mono[:target_samples].copy()
    out = np.zeros(target_samples, dtype=np.float32)
    out[: mono.size] = mono
    return out


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """HTK-style mel filterbank (matches torchlibrosa LogmelFilterBank defaults)."""
    n_freqs = n_fft // 2 + 1
    m_min = _hz_to_mel(fmin)
    m_max = _hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < n_freqs and center != left:
                fb[i - 1, j] = (j - left) / float(center - left)
        for j in range(center, right):
            if 0 <= j < n_freqs and right != center:
                fb[i - 1, j] = (right - j) / float(right - center)
    # Normalize filters (slaney-ish energy norm used by many SED stacks).
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    fb *= enorm[:, np.newaxis].astype(np.float32)
    return fb


def _stft_magnitude(pcm: np.ndarray, *, n_fft: int, hop: int) -> np.ndarray:
    """Centered reflect-pad STFT magnitude (power=1 path into log-mel)."""
    mono = np.asarray(pcm, dtype=np.float64).reshape(-1)
    pad = n_fft // 2
    padded = np.pad(mono, (pad, pad), mode="reflect")
    window = np.hanning(n_fft).astype(np.float64)
    n_frames = 1 + (padded.size - n_fft) // hop
    if n_frames < 1:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop, padded.strides[0]),
        writeable=False,
    )
    windowed = frames * window[np.newaxis, :]
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)
    return np.abs(spec).astype(np.float32)


def log_mel_spectrogram(
    pcm_32k: np.ndarray,
    *,
    sample_rate: int = DASM_SAMPLE_RATE,
    n_fft: int = DASM_N_FFT,
    hop: int = DASM_HOP,
    n_mels: int = DASM_N_MELS,
    fmin: float = DASM_FMIN,
    fmax: float = DASM_FMAX,
    target_frames: int = DASM_MEL_FRAMES,
) -> np.ndarray:
    """Return log-mel as float32 [1, 1, T, n_mels] for DASM ONNX."""
    mag = _stft_magnitude(pcm_32k, n_fft=n_fft, hop=hop)  # [T, F]
    fb = _mel_filterbank(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    mel = mag @ fb.T  # [T, n_mels]
    mel = np.maximum(mel, 1e-10)
    log_mel = np.log10(mel).astype(np.float32)
    if log_mel.shape[0] > target_frames:
        log_mel = log_mel[:target_frames]
    elif log_mel.shape[0] < target_frames:
        pad = np.zeros((target_frames - log_mel.shape[0], n_mels), dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=0)
    # [1, 1, T, mel]
    return log_mel[np.newaxis, np.newaxis, :, :]


def load_bn0(path: Path | None = None, dasm_dir: Path = DEFAULT_DASM_DIR) -> dict[str, np.ndarray] | None:
    bn_path = path or dasm_model_paths(dasm_dir)["bn0"]
    if not bn_path.exists():
        return None
    data = np.load(bn_path)
    return {k: np.asarray(data[k], dtype=np.float32) for k in data.files}


def apply_bn0(mel: np.ndarray, bn0: dict[str, np.ndarray]) -> np.ndarray:
    """Apply HTS-AT bn0 (BatchNorm2d over mel bins). mel: [B,1,T,M]."""
    x = np.asarray(mel, dtype=np.float32)
    # wav2mel does transpose to [B,M,T,1] before bn0.
    # y = (x - mean) / sqrt(var+eps) * weight + bias, broadcast on mel axis.
    mean = bn0["running_mean"].reshape(1, -1, 1, 1)
    var = bn0["running_var"].reshape(1, -1, 1, 1)
    weight = bn0["weight"].reshape(1, -1, 1, 1)
    bias = bn0["bias"].reshape(1, -1, 1, 1)
    eps_raw = bn0.get("eps", np.float32(1e-5))
    eps = float(np.asarray(eps_raw).reshape(-1)[0])
    # x is [B,1,T,M] → treat as channels-last mel; convert to [B,M,T,1]
    x_bmt = np.transpose(x, (0, 3, 2, 1))
    y = (x_bmt - mean) / np.sqrt(var + eps) * weight + bias
    return np.transpose(y, (0, 3, 2, 1)).astype(np.float32)


def prepare_dasm_mel(
    pcm_48k: np.ndarray,
    *,
    capture_rate: int = CAPTURE_SAMPLE_RATE,
    hpf_hz: float = DEFAULT_HPF_HZ,
    hpf_order: int = DEFAULT_HPF_ORDER,
    bn0: dict[str, np.ndarray] | None = None,
    dasm_dir: Path = DEFAULT_DASM_DIR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """HPF + peak-norm @ capture rate → 32 kHz 10 s → log-mel (+ optional bn0)."""
    mono = np.asarray(pcm_48k, dtype=np.float32).reshape(-1)
    mono = apply_butter_hpf(
        mono, sample_rate=float(capture_rate), hpf_hz=hpf_hz, hpf_order=hpf_order
    )
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1e-8:
        mono = mono / peak

    pcm_32k = _resample(mono, src_rate=int(capture_rate), dst_rate=DASM_SAMPLE_RATE)
    target = int(round(DASM_AUDIO_SECONDS * DASM_SAMPLE_RATE))
    pcm_32k = pad_or_truncate(pcm_32k, target)

    mel = log_mel_spectrogram(pcm_32k)
    used_bn0 = False
    bn = bn0
    if bn is None:
        bn = load_bn0(dasm_dir=dasm_dir)
    if bn is not None:
        mel = apply_bn0(mel, bn)
        used_bn0 = True

    meta = {
        "sample_rate": DASM_SAMPLE_RATE,
        "audio_seconds": DASM_AUDIO_SECONDS,
        "mel_shape": list(mel.shape),
        "bn0": used_bn0,
        "peak": peak,
    }
    return mel, meta
