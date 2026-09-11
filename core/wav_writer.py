"""Optional mono WAV recorder for ungained capture PCM."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from core.audio_constants import CAPTURE_SAMPLE_RATE


class WavWriter:
    """Write float32 mono PCM [-1, 1] as 16-bit PCM WAV."""

    def __init__(
        self,
        path: Path,
        *,
        sample_rate: int = CAPTURE_SAMPLE_RATE,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wave = wave.open(str(self.path), "wb")
        self._wave.setnchannels(1)
        self._wave.setsampwidth(2)
        self._wave.setframerate(self.sample_rate)
        self._closed = False

    def write_float32(self, samples: np.ndarray) -> None:
        """Append mono float32 samples clipped to int16 PCM."""
        if self._closed:
            raise RuntimeError(f"WavWriter already closed: {self.path}")
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        clipped = np.clip(mono, -1.0, 1.0)
        # Symmetric int16 mapping; keep 32767 reachable from +1.0.
        pcm16 = (clipped * 32767.0).astype(np.int16)
        self._wave.writeframes(pcm16.tobytes())

    def close(self) -> None:
        """Finalize RIFF sizes. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._wave.close()
        except Exception:  # noqa: BLE001 — best-effort finalize on stop
            pass
