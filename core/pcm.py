"""INMP441 PCM alignment: ALSA S32_LE int32 frames to normalized float32."""

from __future__ import annotations

import numpy as np

# INMP441 delivers 24-bit samples left-aligned in a 32-bit I2S word.
PCM_24BIT_SCALE = float(2**23)


def int32_frames_to_float32(raw_frames: np.ndarray) -> np.ndarray:
    """Align 32-bit I2S frames and normalize to [-1.0, +1.0].

    ALSA ``S32_LE`` capture returns 32-bit containers holding 24-bit PCM.
    Right-shift by 8 bits to recover the signed 24-bit value, then divide
    by 2^23 for unit-scale float samples.

    Args:
        raw_frames: One-dimensional int32 (or int64-safe) sample array.

    Returns:
        float32 array in nominal range [-1.0, +1.0].
    """
    frames = np.asarray(raw_frames).reshape(-1)
    if frames.size == 0:
        return np.array([], dtype=np.float32)

    # Promote before shift to avoid overflow on int32 edge cases.
    aligned = frames.astype(np.int64) >> 8
    return (aligned / PCM_24BIT_SCALE).astype(np.float32)


def naive_int32_to_float32(raw_frames: np.ndarray) -> np.ndarray:
    """Naive normalization without 24-bit alignment (for comparison only)."""
    frames = np.asarray(raw_frames, dtype=np.float64).reshape(-1)
    return (frames / float(2**31)).astype(np.float32)


def chunk_stats(samples: np.ndarray) -> dict[str, float]:
    """Return peak and RMS for a float32 mono buffer."""
    if samples.size == 0:
        return {"peak": 0.0, "rms": 0.0}

    x = samples.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    return {"peak": peak, "rms": rms}
