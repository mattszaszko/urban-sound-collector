"""Tests for Branch A loudness (LAeq-style dBA_spl + LAFmax)."""

from __future__ import annotations

import unittest

import numpy as np

from core.loudness import FAST_TAU_SECONDS, LoudnessEngine, design_a_weighting_sos


class LoudnessEngineTests(unittest.TestCase):
    def test_steady_tone_has_lafmax_near_laeq(self) -> None:
        sr = 48_000.0
        t = np.arange(int(sr * 0.975), dtype=np.float64) / sr
        tone = (0.1 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        eng = LoudnessEngine(sample_rate=sr, calib_offset=120.0)
        m = eng.analyse(tone)
        self.assertIn("dBA_spl", m)
        self.assertIn("LAFmax_dB", m)
        # Steady tone: Fast max should be close to equivalent level.
        self.assertLess(abs(m["LAFmax_dB"] - m["dBA_spl"]), 3.0)

    def test_impulse_raises_lafmax_above_laeq(self) -> None:
        sr = 48_000.0
        n = int(sr * 0.975)
        quiet = np.full(n, 1e-4, dtype=np.float32)
        # Short loud burst near the start.
        burst = int(0.02 * sr)
        quiet[:burst] = 0.5
        eng = LoudnessEngine(sample_rate=sr, calib_offset=120.0)
        m = eng.analyse(quiet)
        self.assertGreater(m["LAFmax_dB"], m["dBA_spl"] + 5.0)

    def test_a_weight_sos_shapes(self) -> None:
        sos = design_a_weighting_sos(48_000.0)
        self.assertEqual(sos.ndim, 2)
        self.assertEqual(sos.shape[1], 6)

    def test_fast_tau_is_125_ms(self) -> None:
        self.assertAlmostEqual(FAST_TAU_SECONDS, 0.125)


if __name__ == "__main__":
    unittest.main()
