"""Fixed ±5 s DASM capture window around a YAMNet arm instant."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from core.clap_event_buffer import ChunkTelemetry, ClapEventBuffer
from core.dasm_prompts import DEFAULT_POST_ROLL_S, DEFAULT_PRE_ROLL_S


class CaptureState(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    READY = "ready"


class FixedDasmCapture:
    """Lookback ring + fixed pre/post roll around arm time.

    On arm: snapshot the last ``pre_roll`` seconds from the lookback buffer,
    then collect the next ``post_roll`` seconds of live PCM. Emit a contiguous
    float32 mono window of length ``pre_roll + post_roll`` (zero-padded if the
    stream started mid-window).
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        pre_roll_s: float = DEFAULT_PRE_ROLL_S,
        post_roll_s: float = DEFAULT_POST_ROLL_S,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.pre_roll_s = float(pre_roll_s)
        self.post_roll_s = float(post_roll_s)
        self.pre_roll_samples = max(1, int(round(self.pre_roll_s * self.sample_rate)))
        self.post_roll_samples = max(1, int(round(self.post_roll_s * self.sample_rate)))
        self.buffer = ClapEventBuffer(
            sample_rate=self.sample_rate,
            lookback_seconds=self.pre_roll_s + 1.0,
        )
        self._state = CaptureState.IDLE
        self._pre_pcm = np.zeros(0, dtype=np.float32)
        self._post_parts: list[np.ndarray] = []
        self._post_collected = 0
        self.meta: dict[str, Any] = {}

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
    def warm(self) -> bool:
        return self.buffer.warm

    @property
    def event_seconds(self) -> float:
        return float(self.pre_roll_s + self.post_roll_s)

    @property
    def post_collected_samples(self) -> int:
        return int(self._post_collected)

    def feed(self, pcm: np.ndarray, telem: ChunkTelemetry) -> None:
        mono = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self.buffer.append(mono, telem)
        if self._state != CaptureState.CAPTURING:
            return
        if mono.size and self._post_collected < self.post_roll_samples:
            remain = self.post_roll_samples - self._post_collected
            take = mono[:remain]
            self._post_parts.append(take.copy())
            self._post_collected += int(take.size)
        if self._post_collected >= self.post_roll_samples:
            self._state = CaptureState.READY
            self.meta["window_complete"] = True
            self.meta["post_samples"] = self._post_collected

    def arm(self, meta: dict[str, Any]) -> None:
        if self._state != CaptureState.IDLE:
            raise RuntimeError(f"FixedDasmCapture.arm called in state {self._state}")
        self._pre_pcm = self._pcm_last(self.pre_roll_samples)
        self._post_parts = []
        self._post_collected = 0
        self._state = CaptureState.CAPTURING
        self.meta = {
            **dict(meta),
            "window": "fixed_pm",
            "pre_roll_s": self.pre_roll_s,
            "post_roll_s": self.post_roll_s,
            "event_seconds": self.event_seconds,
            "t_start_reason": "arm_minus_pre_roll",
            "pre_samples": int(self._pre_pcm.size),
            "post_target_samples": self.post_roll_samples,
        }

    def _pcm_last(self, n: int) -> np.ndarray:
        n = max(0, int(n))
        if n <= 0 or self.buffer._total_samples <= 0:
            return np.zeros(0, dtype=np.float32)
        start = max(0, self.buffer._total_samples - n)
        return self.buffer.pcm_from_offset(start)

    def waveform(self) -> np.ndarray:
        pre = self._pre_pcm
        post = (
            np.concatenate(self._post_parts).astype(np.float32, copy=False)
            if self._post_parts
            else np.zeros(0, dtype=np.float32)
        )
        if post.size < self.post_roll_samples:
            post = np.concatenate(
                [post, np.zeros(self.post_roll_samples - post.size, dtype=np.float32)]
            )
        elif post.size > self.post_roll_samples:
            post = post[: self.post_roll_samples]
        if pre.size < self.pre_roll_samples:
            pre = np.concatenate(
                [np.zeros(self.pre_roll_samples - pre.size, dtype=np.float32), pre]
            )
        elif pre.size > self.pre_roll_samples:
            pre = pre[-self.pre_roll_samples :]
        return np.concatenate([pre, post]).astype(np.float32, copy=False)

    def clear(self) -> None:
        self._state = CaptureState.IDLE
        self._pre_pcm = np.zeros(0, dtype=np.float32)
        self._post_parts = []
        self._post_collected = 0
        self.meta = {}
