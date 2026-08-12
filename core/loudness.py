"""Human-perceived loudness metrics with IEC 61672 A-weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import bilinear_zpk, sosfilt, sosfilt_zi, zpk2sos

from core.pcm import chunk_stats

DEFAULT_CALIB_OFFSET = 120.0
_SPL_EPSILON = 1e-12


def design_a_weighting_sos(sample_rate: float) -> np.ndarray:
    """Design an A-weighting IIR filter as SOS for ``sample_rate`` Hz.

    Analog prototype from IEC 61672-1, bilinear-transformed and normalized
    so gain at 1 kHz is approximately 0 dB.
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    # Compensate ~2 dB so |H(1 kHz)| ≈ 0 dB after discretization.
    a1000 = 1.9997

    # Four zeros at the origin (s^4 numerator).
    zeros = np.zeros(4, dtype=np.complex128)
    poles = np.array(
        [
            -2.0 * np.pi * f1,
            -2.0 * np.pi * f1,
            -2.0 * np.pi * f2,
            -2.0 * np.pi * f3,
            -2.0 * np.pi * f4,
            -2.0 * np.pi * f4,
        ],
        dtype=np.complex128,
    )
    gain = ((2.0 * np.pi * f4) ** 2) * (10.0 ** (a1000 / 20.0))

    z_d, p_d, k_d = bilinear_zpk(zeros, poles, gain, sample_rate)
    return zpk2sos(z_d, p_d, k_d)


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
        """Compute unweighted RMS, A-weighted RMS, and relative dBA SPL."""
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

        return {
            "rms_unweighted": unweighted["rms"],
            "rms_a_weighted": weighted["rms"],
            "dBA_spl": dba_spl,
        }
