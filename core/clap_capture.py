"""Dynamic CLAP capture: YAMNet arm + energy/gate event slicing."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from core.clap_event_buffer import (
    BufferedChunk,
    ChunkTelemetry,
    ClapEventBuffer,
    resolve_onset_sample,
)

# Defaults matching the Trigger-on-Label / Slice-on-Energy plan.
DEFAULT_LOOKBACK_SECONDS = 4.0
DEFAULT_PRE_ONSET_PAD_MS = 150.0
DEFAULT_END_SETTLE_CHUNKS = 2
DEFAULT_MAX_EVENT_SECONDS = 5.0
DEFAULT_ONSET_DBA_MARGIN_DB = 3.0
DEFAULT_PEAK_DECAY_DB = 5.0


class CaptureState(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    READY = "ready"


class DynamicClapCapture:
    """
    Lookback ring + state machine:

    IDLE → (YAMNet arm) retroactive onset → CAPTURING → READY
    (peak decay, gate close, or max cap).
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS,
        pre_onset_pad_ms: float = DEFAULT_PRE_ONSET_PAD_MS,
        end_settle_chunks: int = DEFAULT_END_SETTLE_CHUNKS,
        max_event_seconds: float = DEFAULT_MAX_EVENT_SECONDS,
        onset_dba_margin_db: float = DEFAULT_ONSET_DBA_MARGIN_DB,
        peak_decay_db: float = DEFAULT_PEAK_DECAY_DB,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.lookback_seconds = float(lookback_seconds)
        self.pre_onset_pad_ms = float(pre_onset_pad_ms)
        self.end_settle_chunks = max(1, int(end_settle_chunks))
        self.max_event_seconds = float(max_event_seconds)
        self.onset_dba_margin_db = float(onset_dba_margin_db)
        self.peak_decay_db = float(peak_decay_db)

        self.buffer = ClapEventBuffer(
            sample_rate=self.sample_rate,
            lookback_seconds=self.lookback_seconds,
        )
        self._parts: list[np.ndarray] = []
        self._size = 0
        self._state = CaptureState.IDLE
        self._gate_settle_streak = 0
        self._peak_decay_streak = 0
        self._dba_peak: float | None = None
        self.meta: dict[str, Any] = {}

    @property
    def pre_onset_pad_samples(self) -> int:
        return max(0, int(round(self.pre_onset_pad_ms * 0.001 * self.sample_rate)))

    @property
    def max_event_samples(self) -> int:
        return max(1, int(round(self.max_event_seconds * self.sample_rate)))

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state in {CaptureState.CAPTURING, CaptureState.READY}

    @property
    def ready(self) -> bool:
        return self._state == CaptureState.READY

    @property
    def captured_samples(self) -> int:
        return int(self._size)

    @property
    def warm(self) -> bool:
        return self.buffer.warm

    def feed(self, pcm: np.ndarray, telem: ChunkTelemetry) -> None:
        """Always update lookback; while capturing, extend the event window."""
        self.buffer.append(pcm, telem)
        if self._state == CaptureState.CAPTURING:
            self._extend_event(pcm, telem)

    def arm(self, meta: dict[str, Any]) -> None:
        """YAMNet trigger: resolve T_start from lookback and begin capturing."""
        if self._state != CaptureState.IDLE:
            raise RuntimeError(f"DynamicClapCapture.arm called in state {self._state}")
        chunks = self.buffer.chunks()
        onset, reason = resolve_onset_sample(
            chunks,
            margin_db=self.onset_dba_margin_db,
            pre_onset_pad_samples=self.pre_onset_pad_samples,
        )
        seed = self.buffer.pcm_from_offset(onset)
        if seed.size > self.max_event_samples:
            seed = seed[: self.max_event_samples]
            capped = True
        else:
            capped = False

        self._parts = [seed] if seed.size else []
        self._size = int(seed.size)
        self._gate_settle_streak = 0
        self._peak_decay_streak = 0
        self._dba_peak = None
        self._seed_peak_from_chunks(chunks, onset)

        self.meta = {
            **dict(meta),
            "window": "dynamic_energy",
            "t_start_reason": reason,
            "pre_onset_pad_ms": self.pre_onset_pad_ms,
            "lookback_seconds": self.lookback_seconds,
            "max_event_seconds": self.max_event_seconds,
            "peak_decay_db": self.peak_decay_db,
            "capped": capped,
            "event_seconds": round(self._size / float(self.sample_rate), 3),
            "dba_peak": self._dba_peak,
        }
        if capped or self._size >= self.max_event_samples:
            self._finish(capped=True, reason="capped")
            return

        # If the trigger chunk itself already meets an end condition, start streaks.
        if chunks:
            self._update_end_streaks(chunks[-1].telem)
            if self._try_finish_from_streaks():
                return

        self._state = CaptureState.CAPTURING

    def _seed_peak_from_chunks(
        self, chunks: list[BufferedChunk], onset_sample: int
    ) -> None:
        """Track peak dBA over lookback chunks that contribute to the seeded event."""
        cursor = 0
        for chunk in chunks:
            end = cursor + int(chunk.pcm.size)
            if end > onset_sample:
                self._note_peak(chunk.telem.dba_spl)
            cursor = end

    def _note_peak(self, dba_spl: float) -> None:
        try:
            val = float(dba_spl)
        except (TypeError, ValueError):
            return
        if not np.isfinite(val):
            return
        if self._dba_peak is None or val > self._dba_peak:
            self._dba_peak = val

    def _update_end_streaks(self, telem: ChunkTelemetry) -> None:
        if not telem.gate_open:
            self._gate_settle_streak += 1
        else:
            self._gate_settle_streak = 0

        self._note_peak(telem.dba_spl)
        if (
            self._dba_peak is not None
            and float(telem.dba_spl) <= float(self._dba_peak) - self.peak_decay_db
        ):
            self._peak_decay_streak += 1
        else:
            self._peak_decay_streak = 0

    def _try_finish_from_streaks(self) -> bool:
        if self._gate_settle_streak >= self.end_settle_chunks:
            self._finish(capped=False, reason="gate_close")
            return True
        if self._peak_decay_streak >= self.end_settle_chunks:
            self._finish(capped=False, reason="peak_decay")
            return True
        return False

    def _extend_event(self, pcm: np.ndarray, telem: ChunkTelemetry) -> None:
        mono = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        remaining = self.max_event_samples - self._size
        if remaining <= 0:
            self._finish(capped=True, reason="capped")
            return
        if mono.size > remaining:
            mono = mono[:remaining]
            self._parts.append(mono)
            self._size += int(mono.size)
            self._note_peak(telem.dba_spl)
            self._finish(capped=True, reason="capped")
            return

        self._parts.append(mono)
        self._size += int(mono.size)

        self._update_end_streaks(telem)

        if self._size >= self.max_event_samples:
            self._finish(capped=True, reason="capped")
            return
        self._try_finish_from_streaks()

    def _finish(self, *, capped: bool, reason: str) -> None:
        self._state = CaptureState.READY
        self.meta["capped"] = bool(capped)
        self.meta["t_end_reason"] = reason
        self.meta["dba_peak"] = self._dba_peak
        self.meta["event_seconds"] = round(self._size / float(self.sample_rate), 3)

    def waveform(self) -> np.ndarray:
        if not self._parts:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(self._parts).astype(np.float32, copy=False)
        if out.size > self.max_event_samples:
            out = out[: self.max_event_samples]
        return out

    def clear(self) -> None:
        self._parts = []
        self._size = 0
        self._gate_settle_streak = 0
        self._peak_decay_streak = 0
        self._dba_peak = None
        self._state = CaptureState.IDLE
        self.meta = {}
        # Keep lookback buffer across events so the next arm has history.


# Back-compat alias during migration of imports/tests.
HybridClapCapture = DynamicClapCapture
