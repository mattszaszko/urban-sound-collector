"""Human-perceived loudness metrics with IEC 61672 A-weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import bilinear_zpk, sosfilt, sosfilt_zi, zpk2sos

from core.pcm import chunk_stats

DEFAULT_CALIB_OFFSET = 120.0
_SPL_EPSILON = 1e-12
# IEC 61672 Fast time weighting (exponential averaging time constant).
FAST_TAU_SECONDS = 0.125


def _prewarp_hz(freq_hz: float, sample_rate: float) -> float:
    """Analog frequency pre-warping for bilinear transform (rad/s → matched Hz)."""
    # Want digital response at freq_hz: prewarp analog ω_a = 2/T * tan(ω_d * T/2)
    # For zpk bilinear with fs=sample_rate, poles are in rad/s; prewarp critical Hz:
    return (sample_rate / math.pi) * math.tan(math.pi * freq_hz / sample_rate)


def design_a_weighting_sos(sample_rate: float) -> np.ndarray:
    """Design an A-weighting IIR filter as SOS for ``sample_rate`` Hz.

    Analog prototype from IEC 61672-1, bilinear-transformed with frequency
    pre-warping so critical frequencies land correctly near Nyquist, then
    normalized so gain at 1 kHz is approximately 0 dB.
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    # Compensate ~2 dB so |H(1 kHz)| ≈ 0 dB after discretization.
    a1000 = 1.9997

    # Pre-warp corner frequencies before forming analog poles (rad/s).
    w1 = 2.0 * math.pi * _prewarp_hz(f1, sample_rate)
    w2 = 2.0 * math.pi * _prewarp_hz(f2, sample_rate)
    w3 = 2.0 * math.pi * _prewarp_hz(f3, sample_rate)
    w4 = 2.0 * math.pi * _prewarp_hz(f4, sample_rate)

    # Four zeros at the origin (s^4 numerator).
    zeros = np.zeros(4, dtype=np.complex128)
    poles = np.array(
        [-w1, -w1, -w2, -w3, -w4, -w4],
        dtype=np.complex128,
    )
    gain = (w4**2) * (10.0 ** (a1000 / 20.0))

    z_d, p_d, k_d = bilinear_zpk(zeros, poles, gain, sample_rate)
    return zpk2sos(z_d, p_d, k_d)


def _fast_max_db(
    a_weighted: np.ndarray,
    *,
    sample_rate: float,
    calib_offset: float,
) -> float:
    """LAFmax: max of Fast (125 ms) exponential level over the chunk."""
    mono = np.asarray(a_weighted, dtype=np.float64).reshape(-1)
    if mono.size == 0:
        return calib_offset + 20.0 * math.log10(_SPL_EPSILON)

    # Squared-pressure style detector on A-weighted waveform.
    x2 = np.square(mono)
    # Discrete one-pole: y[n] = α x2[n] + (1-α) y[n-1], α = 1 - exp(-1/(τ fs))
    alpha = 1.0 - math.exp(-1.0 / (FAST_TAU_SECONDS * float(sample_rate)))
    alpha = min(max(alpha, 0.0), 1.0)
    y = 0.0
    peak = 0.0
    for v in x2:
        y = alpha * float(v) + (1.0 - alpha) * y
        if y > peak:
            peak = y
    return 10.0 * math.log10(max(peak, _SPL_EPSILON**2)) + calib_offset


@dataclass
class LoudnessEngine:
    """Stateful A-weighted loudness analyser for contiguous PCM chunks."""

    sample_rate: float
    calib_offset: float = DEFAULT_CALIB_OFFSET
    sos: np.ndarray = field(init=False)
    _zi: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.sos = design_a_weighting_sos(self.sample_rate)
        self._zi = None

    def analyse(self, samples: np.ndarray) -> dict[str, float]:
        """Compute unweighted RMS, A-weighted LAeq≈1s, and LAFmax.

        ``dBA_spl`` is an LAeq,1s-style equivalent continuous level over the
        ~0.975 s capture chunk (A-weighted RMS + calib_offset).
        ``LAFmax_dB`` is Fast (125 ms) maximum A-weighted level in the chunk.
        """
        mono = np.asarray(samples, dtype=np.float64).reshape(-1)
        unweighted = chunk_stats(mono.astype(np.float32, copy=False))

        if self._zi is None:
            # Steady-state for constant input equal to the first sample.
            self._zi = sosfilt_zi(self.sos) * float(mono[0] if mono.size else 0.0)

        filtered, self._zi = sosfilt(self.sos, mono, zi=self._zi)
        weighted = chunk_stats(filtered.astype(np.float32, copy=False))

        dba_spl = (
            20.0 * math.log10(weighted["rms"] + _SPL_EPSILON) + self.calib_offset
        )
        laf_max = _fast_max_db(
            filtered,
            sample_rate=self.sample_rate,
            calib_offset=self.calib_offset,
        )

        return {
            "rms_unweighted": unweighted["rms"],
            "rms_a_weighted": weighted["rms"],
            "dBA_spl": dba_spl,
            "LAFmax_dB": laf_max,
        }
