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
DEFAULT_HPF_ORDER = 4
DEFAULT_TARGET_DBFS = -23.0
DEFAULT_PEAK_CEILING = 0.9
DEFAULT_GAIN_SMOOTH_CHUNKS = 5

# L90-tied dynamic AGC cap: max_gain = clamp(target - floor - margin, min, ceiling).
DEFAULT_AMBIENT_GAIN_MARGIN_DB = 18.0
DEFAULT_EFFECTIVE_SLACK_DB = 12.0
DEFAULT_MAX_GAIN_MIN_DB = 12.0
DEFAULT_MAX_GAIN_CEILING_DB = 40.0

# L90 dynamic gate defaults (~5 min window @ 0.975 s/chunk).
DEFAULT_AMBIENT_WINDOW_CHUNKS = 300
DEFAULT_GATE_SENSITIVITY_DB = 5.0
DEFAULT_AMBIENT_PERCENTILE = 10.0  # L90: level exceeded 90% of the time
DEFAULT_GATE_DELTA_DB = 4.0
DEFAULT_GATE_DELTA_MIN_DBFS = -55.0
DEFAULT_GATE_SUBWINDOW_MS = 200.0
DEFAULT_GATE_RELEASE_CHUNKS = 2

GATE_MODE_DYNAMIC_L90 = "dynamic_l90"

GATE_REASON_CLOSED = "closed"
GATE_REASON_L90 = "l90"
GATE_REASON_DELTA = "delta"
GATE_REASON_HOLD = "hold"

_RMS_EPSILON = 1e-12


def linear_to_dbfs(value: float) -> float:
    """Convert a linear amplitude (0–1) to dBFS."""
    return 20.0 * math.log10(max(value, _RMS_EPSILON))


def dbfs_to_linear(dbfs: float) -> float:
    """Convert dBFS to linear amplitude."""
    return 10.0 ** (dbfs / 20.0)


def silence_predictions(top_k: int = 3) -> list[dict[str, Any]]:
    """Synthetic predictions when the silence gate skips YAMNet."""
    predictions = [{"label": "gated", "confidence": 1.0}]
    while len(predictions) < top_k:
        predictions.append({"label": "gated", "confidence": 0.0})
    return predictions


def peak_subwindow_rms_dbfs(
    samples: np.ndarray,
    *,
    sample_rate: float,
    subwindow_ms: float,
) -> float:
    """Return max RMS (dBFS) over non-overlapping sub-windows.

    Trailing remainder shorter than half a window is dropped. If
    ``subwindow_ms <= 0`` or the buffer is shorter than one window, falls
    back to full-buffer RMS.
    """
    mono = np.asarray(samples, dtype=np.float64).reshape(-1)
    if mono.size == 0:
        return linear_to_dbfs(0.0)

    if subwindow_ms <= 0:
        return linear_to_dbfs(float(np.sqrt(np.mean(np.square(mono)))))

    window_samples = max(1, int(round(sample_rate * (subwindow_ms / 1000.0))))
    if mono.size < window_samples:
        return linear_to_dbfs(float(np.sqrt(np.mean(np.square(mono)))))

    min_tail = max(1, window_samples // 2)
    peak_rms = 0.0
    offset = 0
    while offset + window_samples <= mono.size:
        segment = mono[offset : offset + window_samples]
        peak_rms = max(peak_rms, float(np.sqrt(np.mean(np.square(segment)))))
        offset += window_samples

    remainder = mono.size - offset
    if remainder >= min_tail:
        segment = mono[offset:]
        peak_rms = max(peak_rms, float(np.sqrt(np.mean(np.square(segment)))))

    return linear_to_dbfs(peak_rms)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


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
    hpf_order: int = DEFAULT_HPF_ORDER
    target_dbfs: float = DEFAULT_TARGET_DBFS
    peak_ceiling: float = DEFAULT_PEAK_CEILING
    gain_smooth_chunks: int = DEFAULT_GAIN_SMOOTH_CHUNKS
    ambient_gain_margin_db: float = DEFAULT_AMBIENT_GAIN_MARGIN_DB
    effective_slack_db: float = DEFAULT_EFFECTIVE_SLACK_DB
    max_gain_min_db: float = DEFAULT_MAX_GAIN_MIN_DB
    max_gain_ceiling_db: float = DEFAULT_MAX_GAIN_CEILING_DB
    ambient_window_chunks: int = DEFAULT_AMBIENT_WINDOW_CHUNKS
    gate_sensitivity_db: float = DEFAULT_GATE_SENSITIVITY_DB
    ambient_percentile: float = DEFAULT_AMBIENT_PERCENTILE
    gate_delta_db: float = DEFAULT_GATE_DELTA_DB
    gate_delta_min_dbfs: float = DEFAULT_GATE_DELTA_MIN_DBFS
    gate_subwindow_ms: float = DEFAULT_GATE_SUBWINDOW_MS
    gate_release_chunks: int = DEFAULT_GATE_RELEASE_CHUNKS
    hpf_sos: np.ndarray = field(init=False)
    _hpf_zi: np.ndarray | None = field(init=False, default=None)
    _rms_history: Deque[float] = field(init=False)
    _ambient_rms_dbfs: Deque[float] = field(init=False)
    _gate_open: bool = field(init=False, default=False)
    _below_open_streak: int = field(init=False, default=0)
    _prev_gate_level_dbfs: float | None = field(init=False, default=None)
    _last_ambient_floor_dbfs: float = field(init=False, default=0.0)
    _last_gate_open_dbfs: float = field(init=False, default=0.0)
    _last_gate_level_dbfs: float = field(init=False, default=0.0)
    _last_delta_rms_dbfs: float | None = field(init=False, default=None)
    _last_gate_open_reason: str = field(init=False, default=GATE_REASON_CLOSED)
    _last_max_gain_db: float = field(init=False, default=DEFAULT_MAX_GAIN_MIN_DB)
    _last_effective_level_dbfs: float = field(init=False, default=0.0)
    _last_min_effective_dbfs: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.ambient_window_chunks < 1:
            raise ValueError(
                f"ambient_window_chunks must be >= 1 (got {self.ambient_window_chunks})"
            )
        if self.gate_delta_db < 0:
            raise ValueError(
                f"gate_delta_db must be >= 0 (got {self.gate_delta_db})"
            )
        if self.gate_subwindow_ms < 0:
            raise ValueError(
                f"gate_subwindow_ms must be >= 0 (got {self.gate_subwindow_ms})"
            )
        if self.gate_release_chunks < 1:
            raise ValueError(
                f"gate_release_chunks must be >= 1 (got {self.gate_release_chunks})"
            )
        if self.ambient_gain_margin_db < 0:
            raise ValueError(
                f"ambient_gain_margin_db must be >= 0 (got {self.ambient_gain_margin_db})"
            )
        if self.effective_slack_db < 0:
            raise ValueError(
                f"effective_slack_db must be >= 0 (got {self.effective_slack_db})"
            )
        if self.max_gain_min_db < 0:
            raise ValueError(
                f"max_gain_min_db must be >= 0 (got {self.max_gain_min_db})"
            )
        if self.max_gain_ceiling_db < self.max_gain_min_db:
            raise ValueError(
                "max_gain_ceiling_db must be >= max_gain_min_db "
                f"(got ceiling={self.max_gain_ceiling_db}, min={self.max_gain_min_db})"
            )
        if self.hpf_order < 1:
            raise ValueError(f"hpf_order must be >= 1 (got {self.hpf_order})")
        nyquist = self.sample_rate / 2.0
        cutoff = min(max(self.hpf_hz, 1.0), nyquist * 0.99)
        self.hpf_sos = butter(
            self.hpf_order, cutoff, btype="high", fs=self.sample_rate, output="sos"
        )
        self._rms_history = deque(maxlen=max(1, self.gain_smooth_chunks))
        self._ambient_rms_dbfs = deque(maxlen=self.ambient_window_chunks)
        self._last_min_effective_dbfs = self.target_dbfs - self.effective_slack_db

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
            np.percentile(
                np.asarray(self._ambient_rms_dbfs, dtype=np.float64),
                self.ambient_percentile,
            )
        )

    def _effective_max_gain_db(self, ambient_floor_dbfs: float) -> float:
        """L90-tied AGC cap: enough to lift floor to target-margin, clamped."""
        raw = self.target_dbfs - ambient_floor_dbfs - self.ambient_gain_margin_db
        return clamp(raw, self.max_gain_min_db, self.max_gain_ceiling_db)

    def _min_effective_dbfs(self) -> float:
        """Lowest post-cap level accepted for YAMNet (target - slack)."""
        return self.target_dbfs - self.effective_slack_db

    def _is_warm(self, gate_level_dbfs: float, max_gain_db: float) -> bool:
        """True if gate_level + max_gain can reach the min effective level."""
        effective = gate_level_dbfs + max_gain_db
        self._last_effective_level_dbfs = effective
        return effective >= self._min_effective_dbfs()

    def _compute_thresholds(self, raw_rms_dbfs: float) -> tuple[float, float]:
        """Return (ambient_floor, open_dbfs) for the current chunk."""
        floor = self._ambient_noise_floor(raw_rms_dbfs)
        open_dbfs = floor + self.gate_sensitivity_db
        self._last_ambient_floor_dbfs = floor
        self._last_gate_open_dbfs = open_dbfs
        self._last_max_gain_db = self._effective_max_gain_db(floor)
        self._last_min_effective_dbfs = self._min_effective_dbfs()
        return floor, open_dbfs

    def _apply_gate(
        self,
        gate_level_dbfs: float,
        open_dbfs: float,
        *,
        warm: bool,
    ) -> str:
        """Open on L90 + warmth; close after release_chunks below open or cold."""
        if self._gate_open:
            if gate_level_dbfs >= open_dbfs and warm:
                self._below_open_streak = 0
                return GATE_REASON_HOLD
            # Below open, or too cold even at max gain: count toward release.
            self._below_open_streak += 1
            if self._below_open_streak >= self.gate_release_chunks:
                self._gate_open = False
                self._below_open_streak = 0
                return GATE_REASON_CLOSED
            return GATE_REASON_HOLD
        self._below_open_streak = 0
        if gate_level_dbfs >= open_dbfs and warm:
            self._gate_open = True
            return GATE_REASON_L90
        return GATE_REASON_CLOSED

    def _maybe_force_delta_open(
        self,
        gate_level_dbfs: float,
        delta_rms_dbfs: float | None,
        *,
        warm: bool,
    ) -> str | None:
        """Force gate open on ΔRMS jump above an absolute floor; return reason."""
        if self._gate_open:
            return None
        if not warm:
            return None
        if self.gate_delta_db <= 0 or delta_rms_dbfs is None:
            return None
        if gate_level_dbfs < self.gate_delta_min_dbfs:
            return None
        if delta_rms_dbfs >= self.gate_delta_db:
            self._gate_open = True
            self._below_open_streak = 0
            return GATE_REASON_DELTA
        return None

    def _metadata(self, *, gate_open: bool, gated: bool) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "gate_mode": GATE_MODE_DYNAMIC_L90,
            "hpf_hz": self.hpf_hz,
            "hpf_order": self.hpf_order,
            "target_dbfs": self.target_dbfs,
            "max_gain_db": round(self._last_max_gain_db, 1),
            "ambient_gain_margin_db": self.ambient_gain_margin_db,
            "effective_slack_db": self.effective_slack_db,
            "max_gain_min_db": self.max_gain_min_db,
            "max_gain_ceiling_db": self.max_gain_ceiling_db,
            "min_effective_dbfs": round(self._last_min_effective_dbfs, 1),
            "effective_level_dbfs": round(self._last_effective_level_dbfs, 1),
            "ambient_noise_floor_dbfs": round(self._last_ambient_floor_dbfs, 1),
            "ambient_sample_count": len(self._ambient_rms_dbfs),
            "ambient_window_chunks": self.ambient_window_chunks,
            "ambient_percentile": self.ambient_percentile,
            "gate_sensitivity_db": self.gate_sensitivity_db,
            "gate_delta_db": self.gate_delta_db,
            "gate_delta_min_dbfs": self.gate_delta_min_dbfs,
            "gate_subwindow_ms": self.gate_subwindow_ms,
            "gate_release_chunks": self.gate_release_chunks,
            "below_open_streak": self._below_open_streak,
            "silence_gate_open_dbfs": round(self._last_gate_open_dbfs, 1),
            "gate_level_dbfs": round(self._last_gate_level_dbfs, 1),
            "gate_open_reason": self._last_gate_open_reason,
            "gate_open": gate_open,
            "gated": gated,
            "gain_smooth_chunks": self.gain_smooth_chunks,
        }
        if self._last_delta_rms_dbfs is None:
            meta["delta_rms_dbfs"] = None
        else:
            meta["delta_rms_dbfs"] = round(self._last_delta_rms_dbfs, 1)
        return meta

    def prepare(self, pcm_48k: np.ndarray) -> YamnetPrepResult:
        """Prepare one capture chunk for YAMNet inference."""
        filtered = self._highpass(pcm_48k)
        stats = chunk_stats(filtered)
        raw_rms = stats["rms"]
        peak = stats["peak"]
        raw_rms_dbfs = linear_to_dbfs(raw_rms)
        gate_level_dbfs = peak_subwindow_rms_dbfs(
            filtered,
            sample_rate=self.sample_rate,
            subwindow_ms=self.gate_subwindow_ms,
        )
        self._last_gate_level_dbfs = gate_level_dbfs

        delta_rms_dbfs: float | None = None
        if self._prev_gate_level_dbfs is not None:
            delta_rms_dbfs = gate_level_dbfs - self._prev_gate_level_dbfs
        self._last_delta_rms_dbfs = delta_rms_dbfs

        _, open_dbfs = self._compute_thresholds(raw_rms_dbfs)
        max_gain_db = self._last_max_gain_db
        warm = self._is_warm(gate_level_dbfs, max_gain_db)

        reason = self._apply_gate(gate_level_dbfs, open_dbfs, warm=warm)
        delta_reason = self._maybe_force_delta_open(
            gate_level_dbfs, delta_rms_dbfs, warm=warm
        )
        if delta_reason is not None:
            reason = delta_reason
        self._last_gate_open_reason = reason

        self._ambient_rms_dbfs.append(raw_rms_dbfs)
        self._prev_gate_level_dbfs = gate_level_dbfs

        gated = not self._gate_open
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

        max_gain_linear = dbfs_to_linear(max_gain_db)
        gain = min(gain, max_gain_linear)

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
