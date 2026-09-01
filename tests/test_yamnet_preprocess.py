"""Tests for dynamic YAMNet preprocessing."""

from __future__ import annotations

import math
import unittest

import numpy as np

from core.audio_constants import CAPTURE_CHUNK_SAMPLES, CAPTURE_SAMPLE_RATE
from core.yamnet_preprocess import (
    GATE_MODE_DYNAMIC_L90,
    YamnetPreprocessor,
    dbfs_to_linear,
    linear_to_dbfs,
    silence_predictions,
)


def _sine_chunk(frequency_hz: float, amplitude: float = 0.05) -> np.ndarray:
    sample_count = CAPTURE_CHUNK_SAMPLES
    t = np.arange(sample_count, dtype=np.float64) / CAPTURE_SAMPLE_RATE
    return (amplitude * np.sin(2.0 * math.pi * frequency_hz * t)).astype(np.float32)


def _constant_chunk(amplitude: float) -> np.ndarray:
    return np.full(CAPTURE_CHUNK_SAMPLES, amplitude, dtype=np.float32)


class YamnetPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            gain_smooth_chunks=5,
            ambient_window_chunks=300,
            gate_sensitivity_db=8.0,
            gate_hysteresis_db=3.0,
        )

    def test_silence_predictions_shape(self) -> None:
        preds = silence_predictions()
        self.assertEqual(len(preds), 3)
        self.assertEqual(preds[0]["label"], "Silence")

    def test_near_silence_is_gated(self) -> None:
        quiet = np.full(CAPTURE_CHUNK_SAMPLES, 1e-7, dtype=np.float32)
        result = self.engine.prepare(quiet)
        self.assertTrue(result.gated)
        self.assertIsNone(result.waveform_16k)
        self.assertEqual(result.metadata["gated"], True)
        self.assertEqual(result.metadata["gate_mode"], GATE_MODE_DYNAMIC_L90)

    def test_one_khz_tone_produces_waveform(self) -> None:
        engine = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            gate_sensitivity_db=0.0,
        )
        result = engine.prepare(_sine_chunk(1000.0, amplitude=0.02))
        self.assertFalse(result.gated)
        self.assertIsNotNone(result.waveform_16k)
        assert result.waveform_16k is not None
        self.assertEqual(result.waveform_16k.size, 15600)
        self.assertGreater(result.applied_gain, 0.0)

    def test_peak_limiter_caps_output(self) -> None:
        engine = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            gate_sensitivity_db=0.0,
        )
        impulse = np.zeros(CAPTURE_CHUNK_SAMPLES, dtype=np.float32)
        impulse[1000] = 0.5
        result = engine.prepare(impulse)
        self.assertFalse(result.gated)
        assert result.waveform_16k is not None
        self.assertLessEqual(float(np.max(np.abs(result.waveform_16k))), 0.9 + 1e-6)

    def test_smoothing_uses_history(self) -> None:
        engine = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            gain_smooth_chunks=5,
            gate_sensitivity_db=-100.0,
        )
        loud = _sine_chunk(1000.0, amplitude=0.05)
        quiet = _sine_chunk(1000.0, amplitude=0.001)

        first = engine.prepare(loud)
        for _ in range(3):
            engine.prepare(loud)
        after_loud = engine.prepare(quiet)

        self.assertFalse(first.gated)
        self.assertFalse(after_loud.gated)
        self.assertGreater(after_loud.applied_gain, first.applied_gain)

    def test_dbfs_helpers(self) -> None:
        self.assertAlmostEqual(linear_to_dbfs(1.0), 0.0)
        self.assertAlmostEqual(dbfs_to_linear(0.0), 1.0)

    def test_first_chunk_bootstraps_and_gates(self) -> None:
        engine = YamnetPreprocessor(gate_sensitivity_db=8.0)
        tone = _sine_chunk(1000.0, amplitude=0.02)
        result = engine.prepare(tone)
        self.assertTrue(result.gated)
        self.assertEqual(result.metadata["ambient_sample_count"], 1)
        self.assertAlmostEqual(
            result.metadata["ambient_noise_floor_dbfs"],
            result.metadata["raw_rms_dbfs"],
        )

    def test_dynamic_open_close_offsets(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=10,
            gate_sensitivity_db=8.0,
            gate_hysteresis_db=3.0,
        )
        quiet = _constant_chunk(1e-5)
        for _ in range(5):
            engine.prepare(quiet)
        result = engine.prepare(quiet)
        floor = result.metadata["ambient_noise_floor_dbfs"]
        open_dbfs = result.metadata["silence_gate_open_dbfs"]
        close_dbfs = result.metadata["silence_gate_close_dbfs"]
        self.assertAlmostEqual(open_dbfs, floor + 8.0, places=1)
        self.assertAlmostEqual(close_dbfs, open_dbfs - 3.0, places=1)

    def test_l90_ignores_spikes(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=100,
            gate_sensitivity_db=8.0,
        )
        quiet_amp = dbfs_to_linear(-80.0)
        loud_amp = dbfs_to_linear(-50.0)
        quiet = _constant_chunk(quiet_amp)

        for _ in range(90):
            engine.prepare(quiet)
        for _ in range(10):
            engine.prepare(_constant_chunk(loud_amp))

        result = engine.prepare(quiet)
        self.assertLess(result.metadata["ambient_noise_floor_dbfs"], -70.0)

    def test_warmup_partial_buffer(self) -> None:
        engine = YamnetPreprocessor(ambient_window_chunks=300)
        quiet = _constant_chunk(1e-5)
        n = 12
        for _ in range(n):
            result = engine.prepare(quiet)
        self.assertEqual(result.metadata["ambient_sample_count"], n)
        expected_floor = float(
            np.percentile(
                np.full(n, result.metadata["raw_rms_dbfs"], dtype=np.float64),
                10.0,
            )
        )
        self.assertAlmostEqual(
            result.metadata["ambient_noise_floor_dbfs"],
            expected_floor,
            places=1,
        )

    def test_hysteresis_with_dynamic_thresholds(self) -> None:
        engine = YamnetPreprocessor(
            gate_sensitivity_db=8.0,
            gate_hysteresis_db=3.0,
        )
        for dbfs in [-80.0] * 15:
            engine._ambient_rms_dbfs.append(dbfs)

        floor, open_dbfs, close_dbfs = engine._compute_thresholds(-80.0)
        self.assertAlmostEqual(open_dbfs, floor + 8.0, places=1)
        self.assertAlmostEqual(close_dbfs, open_dbfs - 3.0, places=1)

        self.assertTrue(engine._apply_gate(-78.0, open_dbfs, close_dbfs))

        engine._gate_open = False
        self.assertFalse(engine._apply_gate(open_dbfs, open_dbfs, close_dbfs))

        self.assertFalse(engine._apply_gate(close_dbfs + 0.5, open_dbfs, close_dbfs))

        self.assertTrue(engine._apply_gate(close_dbfs - 0.5, open_dbfs, close_dbfs))

    def test_invalid_gate_hysteresis(self) -> None:
        with self.assertRaises(ValueError):
            YamnetPreprocessor(gate_hysteresis_db=0.0)

    def test_invalid_ambient_window(self) -> None:
        with self.assertRaises(ValueError):
            YamnetPreprocessor(ambient_window_chunks=0)


if __name__ == "__main__":
    unittest.main()
