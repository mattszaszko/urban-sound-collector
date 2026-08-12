"""Audio intensity metrics."""

from __future__ import annotations

import numpy as np


def calculate_rms(audio_chunk: np.ndarray) -> float:
    """Compute the Root Mean Square (RMS) amplitude of an audio chunk.

    RMS is a simple proxy for perceived sound intensity / loudness.

    Args:
        audio_chunk: One-dimensional float audio array.

    Returns:
        RMS amplitude as a Python float. Returns ``0.0`` for empty input.

    Raises:
        TypeError: If ``audio_chunk`` is not a NumPy array.
        ValueError: If ``audio_chunk`` is not one-dimensional.
    """
    if not isinstance(audio_chunk, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, got {type(audio_chunk).__name__}"
        )

    if audio_chunk.ndim != 1:
        raise ValueError(
            f"Expected a 1-D audio array, got shape {audio_chunk.shape}"
        )

    if audio_chunk.size == 0:
        return 0.0

    samples = audio_chunk.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(samples))))
