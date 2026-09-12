"""Hybrid CLAP capture: pre-roll from ring + post-roll after YAMNet wake."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.clap_trigger import CLAP_POST_ROLL_SECONDS, CLAP_PRE_ROLL_SECONDS


@dataclass
class HybridClapCapture:
    """Accumulate a 7 s pre-roll + 3 s post-roll waveform for one CLAP inference."""

    sample_rate: int
    pre_roll_seconds: float = CLAP_PRE_ROLL_SECONDS
    post_roll_seconds: float = CLAP_POST_ROLL_SECONDS
    _parts: list[np.ndarray] = field(default_factory=list)
    _size: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def pre_samples(self) -> int:
        return max(1, int(round(self.pre_roll_seconds * self.sample_rate)))

    @property
    def post_samples(self) -> int:
        return max(1, int(round(self.post_roll_seconds * self.sample_rate)))

    @property
    def target_samples(self) -> int:
        return self.pre_samples + self.post_samples

    @property
    def active(self) -> bool:
        return bool(self._parts) or self._size > 0

    @property
    def ready(self) -> bool:
        return self._size >= self.target_samples

    @property
    def captured_samples(self) -> int:
        return int(self._size)

    def start(self, pre_roll: np.ndarray, meta: dict[str, Any]) -> None:
        mono = np.asarray(pre_roll, dtype=np.float32).reshape(-1)
        if mono.size > self.pre_samples:
            mono = mono[-self.pre_samples :]
        elif mono.size < self.pre_samples:
            # Should not happen if preroll_ready; pad left with zeros.
            pad = np.zeros(self.pre_samples - mono.size, dtype=np.float32)
            mono = np.concatenate([pad, mono])
        self._parts = [mono]
        self._size = int(mono.size)
        self.meta = dict(meta)

    def append_post(self, samples: np.ndarray) -> None:
        if not self.active:
            raise RuntimeError("HybridClapCapture.append_post called before start")
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        remaining = self.target_samples - self._size
        if remaining <= 0:
            return
        if mono.size > remaining:
            mono = mono[:remaining]
        self._parts.append(mono)
        self._size += int(mono.size)

    def waveform(self) -> np.ndarray:
        if not self._parts:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(self._parts).astype(np.float32, copy=False)
        if out.size > self.target_samples:
            out = out[: self.target_samples]
        return out

    def clear(self) -> None:
        self._parts = []
        self._size = 0
        self.meta = {}
