"""Dynamic YAMNet input preprocessing (Branch B only)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from core.audio_constants import CAPTURE_SAMPLE_RATE
from core.pcm import chunk_stats
from core.resampler import to_yamnet_waveform

DEFAULT_HPF_HZ = 175.0
DEFAULT_TARGET_DBFS = -23.0
DEFAULT_PEAK_CEILING = 0.9
DEFAULT_GAIN_SMOOTH_CHUNKS = 5

# L90 dynamic gate defaults (~5 min window @ 0.975 s/chunk).
DEFAULT_AMBIENT_WINDOW_CHUNKS = 300
DEFAULT_GATE_SENSITIVITY_DB = 8.0
DEFAULT_GATE_HYSTERESIS_DB = 3.0
DEFAULT_AMBIENT_PERCENTILE = 10.0  # L90: level exceeded 90% of the time

GATE_MODE_DYNAMIC_L90 = "dynamic_l90"

_RMS_EPSILON = 1e-12


def linear_to_dbfs(value: float) -> float:
    """Convert a linear amplitude (0–1) to dBFS."""
    return 20.0 * math.log10(max(value, _RMS_EPSILON))


def dbfs_to_linear(dbfs: float) -> float:
    """Convert dBFS to linear amplitude."""
    return 10.0 ** (dbfs / 20.0)


def silence_predictions(top_k: int = 3) -> list[dict[str, Any]]:
    """Synthetic YAMNet-style predictions when the silence gate fires."""
    predictions = [{"label": "Silence", "confidence": 1.0}]
    while len(predictions) < top_k:
        predictions.append({"label": "Silence", "confidence": 0.0})
    return predictions


@dataclass
class YamnetPrepResult:
    """Output of one Branch B preprocessing pass."""

    waveform_16k: np.ndarray | None
    gated: bool
    applied_gain: float
    raw_rms_dbfs: float
    smoothed_rms_dbfs: float
    metadata: dict[str, Any]


@dataclass
class YamnetPreprocessor:
    """HPF, L90 dynamic silence gate, smoothed RMS normalization, resample."""

    sample_rate: float = float(CAPTURE_SAMPLE_RATE)
    hpf_hz: float = DEFAULT_HPF_HZ
    target_dbfs: float = DEFAULT_TARGET_DBFS
    peak_ceiling: float = DEFAULT_PEAK_CEILING
    gain_smooth_chunks: int = DEFAULT_GAIN_SMOOTH_CHUNKS
    ambient_window_chunks: int = DEFAULT_AMBIENT_WINDOW_CHUNKS
    gate_sensitivity_db: float = DEFAULT_GATE_SENSITIVITY_DB
    gate_hysteresis_db: float = DEFAULT_GATE_HYSTERESIS_DB
    ambient_percentile: float = DEFAULT_AMBIENT_PERCENTILE
    hpf_sos: np.ndarray = field(init=False)
    _hpf_zi: np.ndarray | None = field(init=False, default=None)
    _rms_history: Deque[float] = field(init=False)
    _ambient_rms_dbfs: Deque[float] = field(init=False)
    _gate_open: bool = field(init=False, default=False)
    _last_ambient_floor_dbfs: float = field(init=False, default=0.0)
    _last_gate_open_dbfs: float = field(init=False, default=0.0)
    _last_gate_close_dbfs: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.gate_hysteresis_db <= 0:
            raise ValueError(
                f"gate_hysteresis_db must be positive (got {self.gate_hysteresis_db})"
            )
        if self.ambient_window_chunks < 1:
            raise ValueError(
                f"ambient_window_chunks must be >= 1 (got {self.ambient_window_chunks})"
            )
        nyquist = self.sample_rate / 2.0
        cutoff = min(max(self.hpf_hz, 1.0), nyquist * 0.99)
        self.hpf_sos = butter(2, cutoff, btype="high", fs=self.sample_rate, output="sos")
        self._rms_history = deque(maxlen=max(1, self.gain_smooth_chunks))
        self._ambient_rms_dbfs = deque(maxlen=self.ambient_window_chunks)

    def _highpass(self, samples: np.ndarray) -> np.ndarray:
        mono = np.asarray(samples, dtype=np.float64).reshape(-1)
        if mono.size == 0:
            return mono.astype(np.float32)

        if self._hpf_zi is None:
            self._hpf_zi = sosfilt_zi(self.hpf_sos) * float(mono[0])

        filtered, self._hpf_zi = sosfilt(self.hpf_sos, mono, zi=self._hpf_zi)
        return filtered.astype(np.float32, copy=False)

    def _ambient_noise_floor(self, raw_rms_dbfs: float) -> float:
        """L90 background floor: 10th percentile of the rolling ambient buffer."""
        if not self._ambient_rms_dbfs:
            return raw_rms_dbfs
        return float(
            np.percentile(np.asarray(self._ambient_rms_dbfs, dtype=np.float64), self.ambient_percentile)
        )

    def _compute_thresholds(
        self, raw_rms_dbfs: float
    ) -> tuple[float, float, float]:
        """Return (ambient_floor, open_dbfs, close_dbfs) for the current chunk."""
        floor = self._ambient_noise_floor(raw_rms_dbfs)
        open_dbfs = floor + self.gate_sensitivity_db
        close_dbfs = open_dbfs - self.gate_hysteresis_db
        self._last_ambient_floor_dbfs = floor
        self._last_gate_open_dbfs = open_dbfs
        self._last_gate_close_dbfs = close_dbfs
        return floor, open_dbfs, close_dbfs

    def _apply_gate(self, raw_rms_dbfs: float, open_dbfs: float, close_dbfs: float) -> bool:
        """Update hysteresis gate; return True when chunk is gated (silence)."""
        if self._gate_open:
            if raw_rms_dbfs < close_dbfs:
                self._gate_open = False
        elif raw_rms_dbfs >= open_dbfs:
            self._gate_open = True
        return not self._gate_open

    def _metadata(self, *, gate_open: bool, gated: bool) -> dict[str, Any]:
        return {
            "gate_mode": GATE_MODE_DYNAMIC_L90,
            "hpf_hz": self.hpf_hz,
            "target_dbfs": self.target_dbfs,
            "ambient_noise_floor_dbfs": round(self._last_ambient_floor_dbfs, 1),
            "ambient_sample_count": len(self._ambient_rms_dbfs),
            "ambient_window_chunks": self.ambient_window_chunks,
            "ambient_percentile": self.ambient_percentile,
            "gate_sensitivity_db": self.gate_sensitivity_db,
            "gate_hysteresis_db": self.gate_hysteresis_db,
            "silence_gate_open_dbfs": round(self._last_gate_open_dbfs, 1),
            "silence_gate_close_dbfs": round(self._last_gate_close_dbfs, 1),
            "gate_open": gate_open,
            "gated": gated,
            "gain_smooth_chunks": self.gain_smooth_chunks,
        }

    def prepare(self, pcm_48k: np.ndarray) -> YamnetPrepResult:
        """Prepare one capture chunk for YAMNet inference."""
        filtered = self._highpass(pcm_48k)
        stats = chunk_stats(filtered)
        raw_rms = stats["rms"]
        peak = stats["peak"]
        raw_rms_dbfs = linear_to_dbfs(raw_rms)

        _, open_dbfs, close_dbfs = self._compute_thresholds(raw_rms_dbfs)
        gated = self._apply_gate(raw_rms_dbfs, open_dbfs, close_dbfs)
        self._ambient_rms_dbfs.append(raw_rms_dbfs)

        if gated:
            return YamnetPrepResult(
                waveform_16k=None,
                gated=True,
                applied_gain=0.0,
                raw_rms_dbfs=round(raw_rms_dbfs, 1),
                smoothed_rms_dbfs=round(raw_rms_dbfs, 1),
                metadata={
                    **self._metadata(gate_open=False, gated=True),
                    "applied_gain": 0.0,
                    "raw_rms_dbfs": round(raw_rms_dbfs, 1),
                    "smoothed_rms_dbfs": round(raw_rms_dbfs, 1),
                },
            )

        self._rms_history.append(raw_rms)
        smoothed_rms = float(sum(self._rms_history) / len(self._rms_history))
        smoothed_rms_dbfs = linear_to_dbfs(smoothed_rms)

        target_rms = dbfs_to_linear(self.target_dbfs)
        gain = target_rms / max(smoothed_rms, _RMS_EPSILON)

        if peak > _RMS_EPSILON:
            peak_limited_gain = self.peak_ceiling / peak
            gain = min(gain, peak_limited_gain)

        normalized = np.clip(filtered * np.float32(gain), -1.0, 1.0)
        waveform_16k = to_yamnet_waveform(normalized, gain=1.0)

        return YamnetPrepResult(
            waveform_16k=waveform_16k,
            gated=False,
            applied_gain=round(float(gain), 4),
            raw_rms_dbfs=round(raw_rms_dbfs, 1),
            smoothed_rms_dbfs=round(smoothed_rms_dbfs, 1),
            metadata={
                **self._metadata(gate_open=True, gated=False),
                "applied_gain": round(float(gain), 4),
                "raw_rms_dbfs": round(raw_rms_dbfs, 1),
                "smoothed_rms_dbfs": round(smoothed_rms_dbfs, 1),
            },
        )
