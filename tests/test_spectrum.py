"""Tests for Branch C spectral analysis."""

from __future__ import annotations

import math
import unittest

import numpy as np

from core.audio_constants import CAPTURE_CHUNK_SAMPLES, CAPTURE_SAMPLE_RATE
from core.spectrum import (
    THIRD_OCTAVE_CENTERS_HZ,
    SpectrumEngine,
    third_octave_edges,
)


def _sine_chunk(frequency_hz: float, amplitude: float = 0.5) -> np.ndarray:
    sample_count = CAPTURE_CHUNK_SAMPLES
    t = np.arange(sample_count, dtype=np.float64) / CAPTURE_SAMPLE_RATE
    return amplitude * np.sin(2.0 * math.pi * frequency_hz * t)


class SpectrumEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SpectrumEngine(sample_rate=float(CAPTURE_SAMPLE_RATE))

    def test_third_octave_edges_are_symmetric(self) -> None:
        low, high = third_octave_edges(1000.0)
        self.assertAlmostEqual(high / 1000.0, 1000.0 / low, places=6)

    def test_output_has_z_and_a_blocks(self) -> None:
        result = self.engine.analyse(_sine_chunk(440.0))
        self.assertEqual(result["band_type"], "third_octave")
        self.assertEqual(result["centers_hz"], list(THIRD_OCTAVE_CENTERS_HZ))
        self.assertIn("levels_db", result["z"])
        self.assertIn("levels_db", result["a"])
        self.assertEqual(len(result["z"]["levels_db"]), len(THIRD_OCTAVE_CENTERS_HZ))
        self.assertEqual(len(result["a"]["levels_db"]), len(THIRD_OCTAVE_CENTERS_HZ))

    def test_440_hz_tone_peaks_near_400_hz_band(self) -> None:
        result = self.engine.analyse(_sine_chunk(440.0))
        z_levels = result["z"]["levels_db"]
        idx_400 = THIRD_OCTAVE_CENTERS_HZ.index(400.0)
        idx_500 = THIRD_OCTAVE_CENTERS_HZ.index(500.0)
        idx_2000 = THIRD_OCTAVE_CENTERS_HZ.index(2000.0)

        self.assertGreater(z_levels[idx_400], z_levels[idx_2000])
        self.assertGreater(z_levels[idx_500], z_levels[idx_2000])

        peaks = result["z"]["peaks"]
        self.assertGreaterEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0]["hz"], 440.0, delta=25.0)
        self.assertAlmostEqual(result["z"]["centroid_hz"], 440.0, delta=80.0)

    def test_low_frequency_tone_has_high_low_energy(self) -> None:
        result = self.engine.analyse(_sine_chunk(100.0))
        z_pct = result["z"]["energy_pct"]
        idx_100 = THIRD_OCTAVE_CENTERS_HZ.index(100.0)

        self.assertGreater(z_pct["low"], z_pct["mid"])
        self.assertGreater(z_pct["low"], z_pct["high"])
        self.assertAlmostEqual(z_pct["low"] + z_pct["mid"] + z_pct["high"], 100.0, delta=0.2)
        self.assertLess(result["a"]["levels_db"][idx_100], result["z"]["levels_db"][idx_100])


if __name__ == "__main__":
    unittest.main()
