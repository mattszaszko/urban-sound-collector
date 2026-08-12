"""Resample capture-rate PCM to YAMNet's 16 kHz window."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly

from core.audio_constants import (
    CAPTURE_SAMPLE_RATE,
    YAMNET_CHUNK_SAMPLES,
    YAMNET_SAMPLE_RATE,
)


def to_yamnet_waveform(
    pcm_48k: np.ndarray,
    *,
    capture_rate: int = CAPTURE_SAMPLE_RATE,
    target_rate: int = YAMNET_SAMPLE_RATE,
    target_samples: int = YAMNET_CHUNK_SAMPLES,
    gain: float = 1.0,
) -> np.ndarray:
    """Polyphase-resample a capture buffer to a fixed YAMNet-length waveform.

    For the default 48 kHz → 16 kHz path this is an exact 3:1 downsample.
    Optional ``gain`` is applied only for classifier input (not loudness) and
    hard-clipped to [-1, 1] after scaling.
    """
    mono = np.asarray(pcm_48k, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if capture_rate == target_rate:
        resampled = mono
    else:
        # Reduce fraction capture_rate:target_rate for resample_poly(up, down).
        gcd = math.gcd(capture_rate, target_rate)
        up = target_rate // gcd
        down = capture_rate // gcd
        resampled = resample_poly(mono, up, down).astype(np.float32, copy=False)

    if resampled.size == target_samples:
        waveform = resampled
    elif resampled.size > target_samples:
        waveform = resampled[:target_samples].copy()
    else:
        waveform = np.zeros(target_samples, dtype=np.float32)
        waveform[: resampled.size] = resampled

    if gain != 1.0:
        waveform = np.clip(waveform * np.float32(gain), -1.0, 1.0)

    return waveform
