"""PCM + per-chunk telemetry lookback for dynamic CLAP event slicing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(frozen=True)
class ChunkTelemetry:
    """Per-chunk gate / level snapshot aligned with a PCM chunk."""

    gate_open: bool
    dba_spl: float
    raw_rms_dbfs: float | None = None
    ambient_noise_floor_dbfs: float | None = None
    chunk_index: int | None = None


@dataclass
class BufferedChunk:
    pcm: np.ndarray
    telem: ChunkTelemetry


class ClapEventBuffer:
    """Rolling lookback of mono PCM chunks with aligned telemetry."""

    def __init__(self, *, sample_rate: int, lookback_seconds: float) -> None:
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1 (got {sample_rate})")
        if lookback_seconds <= 0:
            raise ValueError(f"lookback_seconds must be > 0 (got {lookback_seconds})")
        self.sample_rate = int(sample_rate)
        self.lookback_seconds = float(lookback_seconds)
        self.lookback_samples = max(1, int(round(self.lookback_seconds * self.sample_rate)))
        self._chunks: Deque[BufferedChunk] = deque()
        self._total_samples = 0

    def __len__(self) -> int:
        return self._total_samples

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def warm(self) -> bool:
        """True once at least one chunk is buffered (onset search can run)."""
        return self.chunk_count >= 1

    def seconds(self) -> float:
        return float(self._total_samples) / float(self.sample_rate)

    def clear(self) -> None:
        self._chunks.clear()
        self._total_samples = 0

    def append(self, pcm: np.ndarray, telem: ChunkTelemetry) -> None:
        mono = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        self._chunks.append(BufferedChunk(pcm=mono.copy(), telem=telem))
        self._total_samples += int(mono.size)
        self._trim()

    def _trim(self) -> None:
        while self._chunks and self._total_samples > self.lookback_samples:
            # Drop whole oldest chunks until within capacity (keep newest lookback).
            if (
                self._total_samples - self._chunks[0].pcm.size >= self.lookback_samples
                or len(self._chunks) == 1
            ):
                dropped = self._chunks.popleft()
                self._total_samples -= int(dropped.pcm.size)
                continue
            # Oldest chunk straddles the lookback edge — crop from the left.
            keep = self.lookback_samples - (self._total_samples - self._chunks[0].pcm.size)
            old = self._chunks[0]
            cropped = old.pcm[-keep:]
            self._total_samples -= int(old.pcm.size - cropped.size)
            self._chunks[0] = BufferedChunk(pcm=cropped, telem=old.telem)
            break

    def chunks(self) -> list[BufferedChunk]:
        return list(self._chunks)

    def pcm_from_offset(self, start_sample: int) -> np.ndarray:
        """Concatenate PCM from ``start_sample`` (inclusive) to the end of the buffer."""
        start = max(0, int(start_sample))
        if not self._chunks or start >= self._total_samples:
            return np.zeros(0, dtype=np.float32)
        parts: list[np.ndarray] = []
        cursor = 0
        for chunk in self._chunks:
            end = cursor + int(chunk.pcm.size)
            if end <= start:
                cursor = end
                continue
            local = max(0, start - cursor)
            parts.append(chunk.pcm[local:])
            cursor = end
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)


def energy_elevated(telem: ChunkTelemetry, margin_db: float) -> bool:
    """True when chunk RMS sits above the ambient floor by ``margin_db``."""
    if telem.ambient_noise_floor_dbfs is None or telem.raw_rms_dbfs is None:
        return False
    return float(telem.raw_rms_dbfs) > float(telem.ambient_noise_floor_dbfs) + float(margin_db)


def chunk_settled(telem: ChunkTelemetry, margin_db: float) -> bool:
    """Gate closed, or RMS back near ambient (same units as the gate floor)."""
    if not telem.gate_open:
        return True
    if telem.ambient_noise_floor_dbfs is None or telem.raw_rms_dbfs is None:
        return False
    return float(telem.raw_rms_dbfs) <= float(telem.ambient_noise_floor_dbfs) + float(margin_db)


def resolve_onset_sample(
    chunks: list[BufferedChunk],
    *,
    margin_db: float,
    pre_onset_pad_samples: int,
) -> tuple[int, str]:
    """
    Find sample offset of event onset within ``chunks`` (oldest → newest).

    Prefers the start of the contiguous ``gate_open`` run ending at the newest
    chunk; falls back to an elevated-RMS run; else the start of the newest chunk.
    Applies a pre-onset pad (clamped to buffer start).
    """
    if not chunks:
        return 0, "empty"

    n = len(chunks)
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + int(chunks[i].pcm.size))

    onset_idx: int
    reason: str

    if chunks[-1].telem.gate_open:
        i = n - 1
        while i > 0 and chunks[i - 1].telem.gate_open:
            i -= 1
        onset_idx = i
        reason = "gate_open"
    elif energy_elevated(chunks[-1].telem, margin_db):
        i = n - 1
        while i > 0 and energy_elevated(chunks[i - 1].telem, margin_db):
            i -= 1
        onset_idx = i
        reason = "energy_rise"
    else:
        onset_idx = n - 1
        reason = "trigger_chunk"

    start = starts[onset_idx]
    start = max(0, start - max(0, int(pre_onset_pad_samples)))
    return start, reason
