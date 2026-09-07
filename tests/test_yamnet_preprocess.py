"""Tests for dynamic YAMNet preprocessing."""

from __future__ import annotations

import math
import unittest

import numpy as np

from core.audio_constants import CAPTURE_CHUNK_SAMPLES, CAPTURE_SAMPLE_RATE
from core.yamnet_preprocess import (
    DEFAULT_AMBIENT_GAIN_MARGIN_DB,
    DEFAULT_EFFECTIVE_SLACK_DB,
    DEFAULT_GATE_SENSITIVITY_DB,
    DEFAULT_HPF_ORDER,
    DEFAULT_MAX_GAIN_CEILING_DB,
    DEFAULT_MAX_GAIN_MIN_DB,
    DEFAULT_TARGET_DBFS,
    GATE_MODE_DYNAMIC_L90,
    GATE_REASON_CLOSED,
    GATE_REASON_DELTA,
    GATE_REASON_HOLD,
    GATE_REASON_L90,
    YamnetPreprocessor,
    dbfs_to_linear,
    linear_to_dbfs,
    peak_subwindow_rms_dbfs,
    silence_predictions,
)


def _sine_chunk(frequency_hz: float, amplitude: float = 0.05) -> np.ndarray:
    sample_count = CAPTURE_CHUNK_SAMPLES
    t = np.arange(sample_count, dtype=np.float64) / CAPTURE_SAMPLE_RATE
    return (amplitude * np.sin(2.0 * math.pi * frequency_hz * t)).astype(np.float32)


def _sine_at_dbfs(frequency_hz: float, dbfs: float) -> np.ndarray:
    return _sine_chunk(frequency_hz, amplitude=dbfs_to_linear(dbfs))


class YamnetPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            gain_smooth_chunks=5,
            ambient_window_chunks=300,
            gate_sensitivity_db=5.0,
            gate_delta_db=0.0,
        )

    def test_default_sensitivity_is_five(self) -> None:
        self.assertEqual(DEFAULT_GATE_SENSITIVITY_DB, 5.0)

    def test_default_dynamic_gain_constants(self) -> None:
        self.assertEqual(DEFAULT_HPF_ORDER, 4)
        self.assertEqual(DEFAULT_AMBIENT_GAIN_MARGIN_DB, 18.0)
        self.assertEqual(DEFAULT_EFFECTIVE_SLACK_DB, 12.0)
        self.assertEqual(DEFAULT_MAX_GAIN_MIN_DB, 12.0)
        self.assertEqual(DEFAULT_MAX_GAIN_CEILING_DB, 40.0)
        self.assertEqual(self.engine.hpf_order, 4)

    def test_silence_predictions_shape(self) -> None:
        preds = silence_predictions()
        self.assertEqual(len(preds), 3)
        self.assertEqual(preds[0]["label"], "gated")
        self.assertEqual(preds[1]["label"], "gated")
        self.assertEqual(preds[2]["label"], "gated")

    def test_near_silence_is_gated(self) -> None:
        quiet = np.full(CAPTURE_CHUNK_SAMPLES, 1e-7, dtype=np.float32)
        result = self.engine.prepare(quiet)
        self.assertTrue(result.gated)
        self.assertIsNone(result.waveform_16k)
        self.assertEqual(result.metadata["gated"], True)
        self.assertEqual(result.metadata["gate_mode"], GATE_MODE_DYNAMIC_L90)
        self.assertNotIn("gate_hysteresis_db", result.metadata)
        self.assertIn("effective_level_dbfs", result.metadata)
        self.assertIn("ambient_gain_margin_db", result.metadata)

    def test_one_khz_tone_produces_waveform(self) -> None:
        engine = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            gate_sensitivity_db=0.0,
            gate_delta_db=0.0,
            effective_slack_db=80.0,  # disable warmth for this path test
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
            gate_delta_db=0.0,
            effective_slack_db=80.0,
        )
        impulse = np.zeros(CAPTURE_CHUNK_SAMPLES, dtype=np.float32)
        impulse[1000] = 0.5
        result = engine.prepare(impulse)
        self.assertFalse(result.gated)
        assert result.waveform_16k is not None
        self.assertLessEqual(float(np.max(np.abs(result.waveform_16k))), 0.9 + 1e-6)

    def test_max_gain_from_quiet_floor(self) -> None:
        engine = YamnetPreprocessor()
        # target - floor - margin = -23 - (-77) - 18 = 36
        self.assertAlmostEqual(engine._effective_max_gain_db(-77.0), 36.0, places=1)

    def test_max_gain_from_loud_floor(self) -> None:
        engine = YamnetPreprocessor()
        # target - floor - margin = -23 - (-60) - 18 = 19
        self.assertAlmostEqual(engine._effective_max_gain_db(-60.0), 19.0, places=1)

    def test_max_gain_hits_min_clamp(self) -> None:
        engine = YamnetPreprocessor()
        # -23 - (-50) - 18 = 9 → clamp to 12
        self.assertAlmostEqual(
            engine._effective_max_gain_db(-50.0), DEFAULT_MAX_GAIN_MIN_DB, places=1
        )

    def test_max_gain_hits_ceiling(self) -> None:
        engine = YamnetPreprocessor()
        # -23 - (-90) - 18 = 49 → clamp to 40
        self.assertAlmostEqual(
            engine._effective_max_gain_db(-90.0), DEFAULT_MAX_GAIN_CEILING_DB, places=1
        )

    def test_dynamic_cap_limits_applied_gain(self) -> None:
        engine = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            gate_sensitivity_db=-100.0,
            gate_delta_db=0.0,
            effective_slack_db=80.0,
            gain_smooth_chunks=1,
            ambient_gain_margin_db=18.0,
            max_gain_ceiling_db=24.0,
            max_gain_min_db=12.0,
        )
        # Seed a quiet floor so max_gain wants to be high, then ceiling=24 binds.
        for _ in range(5):
            engine._ambient_rms_dbfs.append(-77.0)
        quiet = _sine_at_dbfs(1000.0, -70.0)
        result = engine.prepare(quiet)
        self.assertFalse(result.gated)
        max_linear = dbfs_to_linear(24.0)
        self.assertLessEqual(result.applied_gain, max_linear + 1e-6)
        self.assertAlmostEqual(result.metadata["max_gain_db"], 24.0, places=1)

    def test_smoothing_uses_history(self) -> None:
        engine = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            gain_smooth_chunks=5,
            gate_sensitivity_db=-100.0,
            gate_delta_db=0.0,
            effective_slack_db=80.0,
            max_gain_ceiling_db=80.0,
            ambient_gain_margin_db=0.0,
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
        engine = YamnetPreprocessor(gate_sensitivity_db=5.0, gate_delta_db=0.0)
        tone = _sine_chunk(1000.0, amplitude=0.02)
        result = engine.prepare(tone)
        self.assertTrue(result.gated)
        self.assertEqual(result.metadata["ambient_sample_count"], 1)
        self.assertAlmostEqual(
            result.metadata["ambient_noise_floor_dbfs"],
            result.metadata["raw_rms_dbfs"],
        )

    def test_dynamic_open_offset(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=10,
            gate_sensitivity_db=5.0,
            gate_delta_db=0.0,
            gate_subwindow_ms=0.0,
        )
        quiet = _sine_at_dbfs(1000.0, -80.0)
        for _ in range(5):
            engine.prepare(quiet)
        result = engine.prepare(quiet)
        floor = result.metadata["ambient_noise_floor_dbfs"]
        open_dbfs = result.metadata["silence_gate_open_dbfs"]
        self.assertAlmostEqual(open_dbfs, floor + 5.0, places=1)
        self.assertEqual(result.metadata["gate_release_chunks"], 2)

    def test_near_miss_floor_plus_six_opens_when_warm(self) -> None:
        """Levels at floor+6 open with sensitivity 5 when warmth passes."""
        engine = YamnetPreprocessor(
            ambient_window_chunks=20,
            gate_sensitivity_db=5.0,
            gate_delta_db=0.0,
            gate_subwindow_ms=0.0,
            ambient_gain_margin_db=18.0,
            effective_slack_db=12.0,
        )
        quiet = _sine_at_dbfs(1000.0, -80.0)
        for _ in range(10):
            engine.prepare(quiet)

        # Amplitude ~3 dB above RMS: aim for gate_level around -71 (warm with max~36).
        mid = _sine_at_dbfs(1000.0, -68.0)
        result = engine.prepare(mid)
        self.assertFalse(result.gated)
        self.assertEqual(result.metadata["gate_open_reason"], GATE_REASON_L90)
        self.assertGreaterEqual(
            result.metadata["effective_level_dbfs"],
            DEFAULT_TARGET_DBFS - DEFAULT_EFFECTIVE_SLACK_DB - 0.5,
        )

    def test_warmth_blocks_barely_open_cold_level(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=20,
            gate_sensitivity_db=5.0,
            gate_delta_db=0.0,
            gate_subwindow_ms=0.0,
            ambient_gain_margin_db=18.0,
            effective_slack_db=12.0,
        )
        for _ in range(15):
            engine._ambient_rms_dbfs.append(-77.0)

        # Open threshold = -72; max_gain = 36; warmth needs >= -71.
        # gate_level -72 fails warmth (-72+36=-36 < -35).
        reason = engine._apply_gate(-72.0, -72.0, warm=False)
        self.assertEqual(reason, GATE_REASON_CLOSED)
        self.assertFalse(engine._gate_open)

    def test_delta_opens_above_min_floor_when_warm(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=20,
            gate_sensitivity_db=30.0,
            gate_delta_db=4.0,
            gate_delta_min_dbfs=-55.0,
            gate_subwindow_ms=0.0,
            effective_slack_db=80.0,  # warmth always passes
        )
        quiet = _sine_at_dbfs(1000.0, -52.0)
        for _ in range(10):
            engine.prepare(quiet)

        jump = _sine_at_dbfs(1000.0, -47.0)
        result = engine.prepare(jump)
        self.assertFalse(result.gated)
        self.assertEqual(result.metadata["gate_open_reason"], GATE_REASON_DELTA)
        self.assertGreaterEqual(result.metadata["delta_rms_dbfs"], 4.0)
        self.assertGreaterEqual(result.metadata["gate_level_dbfs"], -55.0)

    def test_delta_blocked_when_cold(self) -> None:
        engine = YamnetPreprocessor(
            gate_sensitivity_db=30.0,
            gate_delta_db=4.0,
            gate_delta_min_dbfs=-80.0,
        )
        reason = engine._maybe_force_delta_open(-70.0, 5.0, warm=False)
        self.assertIsNone(reason)
        self.assertFalse(engine._gate_open)

    def test_delta_blocked_below_min_floor(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=20,
            gate_sensitivity_db=30.0,
            gate_delta_db=4.0,
            gate_delta_min_dbfs=-55.0,
            gate_subwindow_ms=0.0,
            effective_slack_db=80.0,
        )
        quiet = _sine_at_dbfs(1000.0, -72.0)
        for _ in range(10):
            engine.prepare(quiet)

        jump = _sine_at_dbfs(1000.0, -67.0)
        result = engine.prepare(jump)
        self.assertTrue(result.gated)
        self.assertEqual(result.metadata["gate_open_reason"], GATE_REASON_CLOSED)
        self.assertLess(result.metadata["gate_level_dbfs"], -55.0)

    def test_subwindow_detects_diluted_impulse(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=20,
            gate_sensitivity_db=5.0,
            gate_delta_db=0.0,
            gate_subwindow_ms=200.0,
            effective_slack_db=80.0,
        )
        quiet = _sine_at_dbfs(1000.0, -80.0)
        for _ in range(10):
            engine.prepare(quiet)

        chunk = _sine_at_dbfs(1000.0, -80.0)
        window = int(round(CAPTURE_SAMPLE_RATE * 0.2))
        start = window
        burst = _sine_chunk(1000.0, amplitude=dbfs_to_linear(-70.0))[:window]
        chunk[start : start + window] = burst

        full_rms = linear_to_dbfs(
            float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))
        )
        peak_sub = peak_subwindow_rms_dbfs(
            chunk,
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            subwindow_ms=200.0,
        )
        self.assertLess(full_rms, peak_sub)

        result = engine.prepare(chunk)
        self.assertFalse(result.gated)
        self.assertEqual(result.metadata["gate_subwindow_ms"], 200.0)
        self.assertGreater(
            result.metadata["gate_level_dbfs"], result.metadata["raw_rms_dbfs"]
        )

    def test_l90_ignores_spikes(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=100,
            gate_sensitivity_db=5.0,
            gate_delta_db=0.0,
            gate_subwindow_ms=0.0,
            effective_slack_db=80.0,
        )
        quiet = _sine_at_dbfs(1000.0, -80.0)
        loud = _sine_at_dbfs(1000.0, -50.0)

        for _ in range(90):
            engine.prepare(quiet)
        for _ in range(10):
            engine.prepare(loud)

        result = engine.prepare(quiet)
        self.assertLess(result.metadata["ambient_noise_floor_dbfs"], -70.0)

    def test_warmup_partial_buffer(self) -> None:
        engine = YamnetPreprocessor(
            ambient_window_chunks=300,
            gate_delta_db=0.0,
            gate_subwindow_ms=0.0,
        )
        quiet = _sine_at_dbfs(1000.0, -80.0)
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

    def test_release_closes_after_two_chunks_below_open(self) -> None:
        engine = YamnetPreprocessor(
            gate_sensitivity_db=5.0,
            gate_delta_db=0.0,
            gate_release_chunks=2,
            gate_subwindow_ms=0.0,
        )
        for dbfs in [-80.0] * 15:
            engine._ambient_rms_dbfs.append(dbfs)

        _, open_dbfs = engine._compute_thresholds(-80.0)
        self.assertAlmostEqual(open_dbfs, -75.0, places=1)

        self.assertEqual(
            engine._apply_gate(-78.0, open_dbfs, warm=True), GATE_REASON_CLOSED
        )

        engine._gate_open = False
        self.assertEqual(
            engine._apply_gate(open_dbfs, open_dbfs, warm=True), GATE_REASON_L90
        )

        self.assertEqual(
            engine._apply_gate(open_dbfs - 1.0, open_dbfs, warm=True), GATE_REASON_HOLD
        )
        self.assertEqual(engine._below_open_streak, 1)
        self.assertTrue(engine._gate_open)

        self.assertEqual(
            engine._apply_gate(open_dbfs - 1.0, open_dbfs, warm=True), GATE_REASON_CLOSED
        )
        self.assertFalse(engine._gate_open)
        self.assertEqual(engine._below_open_streak, 0)

    def test_cold_hold_releases_like_below_open(self) -> None:
        engine = YamnetPreprocessor(gate_release_chunks=2)
        open_dbfs = -70.0
        engine._gate_open = True
        # Still above open level but cold → counts toward release.
        self.assertEqual(
            engine._apply_gate(-69.0, open_dbfs, warm=False), GATE_REASON_HOLD
        )
        self.assertEqual(engine._below_open_streak, 1)
        self.assertEqual(
            engine._apply_gate(-69.0, open_dbfs, warm=False), GATE_REASON_CLOSED
        )
        self.assertFalse(engine._gate_open)

    def test_release_resets_streak_when_above_open_and_warm(self) -> None:
        engine = YamnetPreprocessor(gate_release_chunks=2)
        open_dbfs = -70.0
        engine._gate_open = True
        self.assertEqual(engine._apply_gate(-71.0, open_dbfs, warm=True), GATE_REASON_HOLD)
        self.assertEqual(engine._below_open_streak, 1)
        self.assertEqual(engine._apply_gate(-69.0, open_dbfs, warm=True), GATE_REASON_HOLD)
        self.assertEqual(engine._below_open_streak, 0)

    def test_invalid_gate_release_chunks(self) -> None:
        with self.assertRaises(ValueError):
            YamnetPreprocessor(gate_release_chunks=0)

    def test_invalid_max_gain_bounds(self) -> None:
        with self.assertRaises(ValueError):
            YamnetPreprocessor(max_gain_min_db=-1.0)
        with self.assertRaises(ValueError):
            YamnetPreprocessor(max_gain_min_db=30.0, max_gain_ceiling_db=20.0)

    def test_invalid_ambient_window(self) -> None:
        with self.assertRaises(ValueError):
            YamnetPreprocessor(ambient_window_chunks=0)

    def test_invalid_gate_delta(self) -> None:
        with self.assertRaises(ValueError):
            YamnetPreprocessor(gate_delta_db=-1.0)

    def test_gate_uses_post_hpf_levels(self) -> None:
        """Strong sub-cutoff tone is attenuated before gate RMS."""
        engine = YamnetPreprocessor(
            ambient_window_chunks=5,
            gate_sensitivity_db=0.0,
            gate_delta_db=0.0,
            gate_subwindow_ms=0.0,
            hpf_hz=175.0,
            hpf_order=4,
        )
        quiet = _sine_at_dbfs(1000.0, -80.0)
        for _ in range(3):
            engine.prepare(quiet)

        rumble = _sine_at_dbfs(40.0, -40.0)
        result = engine.prepare(rumble)
        self.assertLess(result.metadata["raw_rms_dbfs"], -50.0)


if __name__ == "__main__":
    unittest.main()
