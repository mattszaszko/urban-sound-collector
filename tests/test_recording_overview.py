"""Tests for past-recording overview loudness helpers."""

from __future__ import annotations

import unittest

from core.recording_overview import (
    build_recording_overview,
    loud_disturbance_stats,
    percentile_nearest,
)


class PercentileNearestTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertIsNone(percentile_nearest([], 50))

    def test_single(self) -> None:
        self.assertEqual(percentile_nearest([42.0], 10), 42.0)
        self.assertEqual(percentile_nearest([42.0], 90), 42.0)

    def test_nearest_rank(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        # L90 (quiet) ≈ 10th percentile of ascending sorted dBA
        self.assertEqual(percentile_nearest(values, 10), 10.0)
        self.assertEqual(percentile_nearest(values, 50), 50.0)
        # L10 (loud) ≈ 90th percentile
        self.assertEqual(percentile_nearest(values, 90), 90.0)


class LoudDisturbanceStatsTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(loud_disturbance_stats([], threshold=65.0), (0, 0.0))

    def test_loud_pct_at_threshold(self) -> None:
        vals = [60.0, 65.0, 70.0, 50.0]
        n, pct = loud_disturbance_stats(vals, threshold=65.0)
        self.assertEqual(n, 2)
        self.assertEqual(pct, 50.0)

    def test_threshold_change(self) -> None:
        vals = [60.0, 65.0, 70.0, 50.0]
        n, pct = loud_disturbance_stats(vals, threshold=70.0)
        self.assertEqual(n, 1)
        self.assertEqual(pct, 25.0)


class BuildRecordingOverviewTests(unittest.TestCase):
    def test_summary_loud_and_percentiles(self) -> None:
        events = []
        for i, (dba, laf) in enumerate(
            [
                (50.0, 55.0),
                (60.0, 65.0),
                (70.0, 75.0),
                (55.0, 58.0),
            ]
        ):
            events.append(
                {
                    "created_at": f"2026-09-11T12:00:{i:02d}.000Z",
                    "recording_id": "2026-09-11T12-00-00Z",
                    "dBA_spl": dba,
                    "LAFmax_dB": laf,
                    "top_label": "Silence" if i % 2 == 0 else "Vehicle",
                    "yamnet_preprocess": {
                        "gated": False,
                        "gate_open": True,
                        "ambient_noise_floor_dbfs": -70.0,
                    },
                }
            )
        payload = build_recording_overview(
            events,
            name="sample.jsonl",
            has_wav=False,
            wav_name=None,
            recording_active=False,
            loud_threshold=65.0,
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["chunks"], 4)
        self.assertEqual(payload["summary"]["recording_id"], "2026-09-11T12-00-00Z")
        self.assertEqual(payload["summary"]["disturbances"]["loud_chunks"], 2)
        self.assertEqual(payload["summary"]["disturbances"]["loud_pct"], 50.0)
        self.assertEqual(payload["summary"]["loudness"]["dba_l50"], 55.0)
        self.assertEqual(len(payload["points"]), 4)
        self.assertTrue(payload["points_full"])

    def test_dual_read_legacy_run_id(self) -> None:
        events = [
            {
                "created_at": "2026-09-11T12:00:00.000Z",
                "run_id": "legacy-id",
                "dBA_spl": 50.0,
                "LAFmax_dB": 55.0,
                "top_label": "Silence",
            }
        ]
        payload = build_recording_overview(
            events,
            name="legacy.jsonl",
            has_wav=False,
            wav_name=None,
            recording_active=False,
        )
        self.assertEqual(payload["summary"]["recording_id"], "legacy-id")


if __name__ == "__main__":
    unittest.main()
