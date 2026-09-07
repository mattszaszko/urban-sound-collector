"""Fixed-capacity mono PCM ring buffer for CLAP context windows."""

from __future__ import annotations

import numpy as np


class AudioRingBuffer:
    """Append-only float32 mono ring with a fixed sample capacity."""

    def __init__(self, maxlen_samples: int) -> None:
        if maxlen_samples < 1:
            raise ValueError(f"maxlen_samples must be >= 1 (got {maxlen_samples})")
        self.maxlen_samples = int(maxlen_samples)
        self._buf = np.zeros(self.maxlen_samples, dtype=np.float32)
        self._size = 0
        self._write = 0

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size >= self.maxlen_samples

    def seconds(self, sample_rate: float) -> float:
        return float(self._size) / float(sample_rate)

    def append(self, samples: np.ndarray) -> None:
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        if mono.size >= self.maxlen_samples:
            self._buf[:] = mono[-self.maxlen_samples :]
            self._size = self.maxlen_samples
            self._write = 0
            return

        first = min(mono.size, self.maxlen_samples - self._write)
        self._buf[self._write : self._write + first] = mono[:first]
        second = mono.size - first
        if second:
            self._buf[0:second] = mono[first:]
        self._write = (self._write + mono.size) % self.maxlen_samples
        self._size = min(self.maxlen_samples, self._size + mono.size)

    def snapshot(self) -> np.ndarray:
        """Return oldest→newest samples currently stored."""
        if self._size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._size < self.maxlen_samples:
            return self._buf[: self._size].copy()
        return np.concatenate(
            (self._buf[self._write :], self._buf[: self._write])
        ).astype(np.float32, copy=False)
