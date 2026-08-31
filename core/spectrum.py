"""A- and Z-weighted spectral summaries per capture chunk (Branch C)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import sosfreqz, welch

from core.loudness import design_a_weighting_sos

# IEC / ANSI standard 1/3-octave center frequencies (Hz), 31.5 Hz – 16 kHz.
THIRD_OCTAVE_CENTERS_HZ: tuple[float, ...] = (
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
)

# Analysis band limits (INMP441 roll-off above ~10 kHz limits high-band accuracy).
ANALYSIS_MIN_HZ = 31.5
ANALYSIS_MAX_HZ = 16_000.0
PEAK_MIN_HZ = 25.0
PEAK_MAX_HZ = 16_000.0
PEAK_COUNT = 3
ROLLOFF_FRACTION = 0.85

# Low / mid / high energy regions (Hz).
LOW_BAND = (31.5, 500.0)
MID_BAND = (500.0, 2000.0)
HIGH_BAND = (2000.0, 16_000.0)

WELCH_NPERSEG = 8192
_POWER_EPSILON = 1e-20
_HALF_THIRD_OCTAVE = 2.0 ** (1.0 / 12.0) - 2.0 ** (-1.0 / 12.0)
_THIRD_OCTAVE_EDGE = 2.0 ** (1.0 / 6.0)


def third_octave_edges(center_hz: float) -> tuple[float, float]:
    """Return lower and upper band edges for a 1/3-octave center frequency."""
    return center_hz / _THIRD_OCTAVE_EDGE, center_hz * _THIRD_OCTAVE_EDGE


def _power_to_db(power: float) -> float:
    return 10.0 * math.log10(max(power, _POWER_EPSILON))


def _integrate_psd(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def _band_levels_db(freqs: np.ndarray, psd: np.ndarray) -> list[float]:
    levels: list[float] = []
    for center in THIRD_OCTAVE_CENTERS_HZ:
        f_low, f_high = third_octave_edges(center)
        power = _integrate_psd(freqs, psd, f_low, f_high)
        levels.append(round(_power_to_db(power), 1))
    return levels


def _smooth_psd(psd: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or psd.size < window:
        return psd
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(psd, kernel, mode="same")


def _find_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    count: int,
) -> list[dict[str, float]]:
    mask = (freqs >= PEAK_MIN_HZ) & (freqs <= PEAK_MAX_HZ)
    f = freqs[mask]
    p = psd[mask]
    if f.size < 3:
        return []

    smoothed = _smooth_psd(p, window=5)
    candidates: list[tuple[float, float]] = []
    for idx in range(1, len(smoothed) - 1):
        if smoothed[idx] >= smoothed[idx - 1] and smoothed[idx] > smoothed[idx + 1]:
            candidates.append((float(f[idx]), float(smoothed[idx])))

    candidates.sort(key=lambda item: item[1], reverse=True)

    merged: list[tuple[float, float]] = []
    for hz, power in candidates:
        too_close = False
        for kept_hz, kept_power in merged:
            min_hz = min(hz, kept_hz)
            if abs(hz - kept_hz) < min_hz * _HALF_THIRD_OCTAVE:
                too_close = True
                break
        if too_close:
            continue
        merged.append((hz, power))
        if len(merged) >= count:
            break

    return [
        {"hz": round(hz, 1), "db": round(_power_to_db(power), 1)}
        for hz, power in merged
    ]


def _centroid_hz(freqs: np.ndarray, psd: np.ndarray) -> float:
    mask = (freqs >= ANALYSIS_MIN_HZ) & (freqs <= ANALYSIS_MAX_HZ)
    f = freqs[mask]
    p = psd[mask]
    total = float(np.sum(p))
    if total <= _POWER_EPSILON:
        return 0.0
    return round(float(np.sum(f * p) / total), 1)


def _rolloff_hz(freqs: np.ndarray, psd: np.ndarray, fraction: float = ROLLOFF_FRACTION) -> float:
    mask = (freqs >= ANALYSIS_MIN_HZ) & (freqs <= ANALYSIS_MAX_HZ)
    f = freqs[mask]
    p = psd[mask]
    if f.size == 0:
        return 0.0
    cumulative = np.cumsum(p)
    total = cumulative[-1]
    if total <= _POWER_EPSILON:
        return 0.0
    target = fraction * total
    idx = int(np.searchsorted(cumulative, target))
    idx = min(idx, len(f) - 1)
    return round(float(f[idx]), 1)


def _energy_pct(freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    e_low = _integrate_psd(freqs, psd, LOW_BAND[0], LOW_BAND[1])
    e_mid = _integrate_psd(freqs, psd, MID_BAND[0], MID_BAND[1])
    e_high = _integrate_psd(freqs, psd, HIGH_BAND[0], HIGH_BAND[1])
    total = e_low + e_mid + e_high
    if total <= _POWER_EPSILON:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    return {
        "low": round(100.0 * e_low / total, 1),
        "mid": round(100.0 * e_mid / total, 1),
        "high": round(100.0 * e_high / total, 1),
    }


def _metrics_from_psd(freqs: np.ndarray, psd: np.ndarray) -> dict[str, Any]:
    return {
        "levels_db": _band_levels_db(freqs, psd),
        "peaks": _find_peaks(freqs, psd, count=PEAK_COUNT),
        "centroid_hz": _centroid_hz(freqs, psd),
        "rolloff_85_hz": _rolloff_hz(freqs, psd),
        "energy_pct": _energy_pct(freqs, psd),
    }


@dataclass
class SpectrumEngine:
    """Stateless Z- and A-weighted spectral analyser for one PCM chunk."""

    sample_rate: float
    _a_weight_power: np.ndarray | None = field(init=False, default=None)
    _a_weight_freqs: np.ndarray | None = field(init=False, default=None)

    def _a_weighting_power_curve(self, freqs: np.ndarray) -> np.ndarray:
        if self._a_weight_freqs is None or not np.array_equal(freqs, self._a_weight_freqs):
            sos = design_a_weighting_sos(self.sample_rate)
            _, response = sosfreqz(sos, worN=freqs, fs=self.sample_rate)
            self._a_weight_freqs = freqs.copy()
            self._a_weight_power = np.square(np.abs(response))
        return self._a_weight_power  # type: ignore[return-value]

    def analyse(self, samples: np.ndarray) -> dict[str, Any]:
        """Compute Z- and A-weighted 1/3-octave spectrum summaries."""
        mono = np.asarray(samples, dtype=np.float64).reshape(-1)
        if mono.size == 0:
            empty = _metrics_from_psd(
                np.array([0.0]),
                np.array([0.0]),
            )
            return {
                "band_type": "third_octave",
                "centers_hz": list(THIRD_OCTAVE_CENTERS_HZ),
                "z": empty,
                "a": empty,
            }

        nperseg = min(WELCH_NPERSEG, mono.size)
        freqs, psd_z = welch(
            mono,
            fs=self.sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density",
            detrend="constant",
        )
        psd_z = np.maximum(psd_z, 0.0)
        a_curve = self._a_weighting_power_curve(freqs)
        psd_a = psd_z * a_curve

        return {
            "band_type": "third_octave",
            "centers_hz": list(THIRD_OCTAVE_CENTERS_HZ),
            "z": _metrics_from_psd(freqs, psd_z),
            "a": _metrics_from_psd(freqs, psd_a),
        }
