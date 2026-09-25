"""Tests for past-recording overview loudness helpers."""

from __future__ import annotations

import unittest

from core.recording_overview import (
    build_recording_overview,
    clap_triggered_markers,
    loud_disturbance_stats,
    percentile_nearest,
    slim_event_point,
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


class ClapEventSecondsTests(unittest.TestCase):
    def test_slim_and_marker_include_event_seconds(self) -> None:
        point = slim_event_point(
            {
                "created_at": "2026-01-01T00:00:05.000Z",
                "dBA_spl": 62.0,
                "clap_status": "triggered",
                "clap_top_label": "siren",
                "clap_meta": {"event_seconds": 3.25},
            }
        )
        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(point["clap_event_seconds"], 3.25)
        markers = clap_triggered_markers([point])
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]["event_seconds"], 3.25)
        self.assertEqual(markers[0]["label"], "siren")


class SpectrumSlimTests(unittest.TestCase):
    def test_extracts_a_weighted_centroid_and_dark_mid_bright(self) -> None:
        point = slim_event_point(
            {
                "created_at": "2026-01-01T00:00:00.000Z",
                "dBA_spl": 55.0,
                "spectrum": {
                    "z": {
                        "centroid_hz": 120.0,
                        "energy_pct": {"low": 90.0, "mid": 8.0, "high": 2.0},
                    },
                    "a": {
                        "centroid_hz": 890.0,
                        "energy_pct": {"low": 35.0, "mid": 48.0, "high": 17.0},
                    },
                },
            }
        )
        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(point["centroid_hz"], 890.0)
        self.assertEqual(
            point["spectrum_bands"],
            {"dark": 35.0, "mid": 48.0, "bright": 17.0},
        )

    def test_missing_spectrum_omits_fields(self) -> None:
        point = slim_event_point(
            {
                "created_at": "2026-01-01T00:00:00.000Z",
                "dBA_spl": 55.0,
            }
        )
        self.assertIsNotNone(point)
        assert point is not None
        self.assertNotIn("centroid_hz", point)
        self.assertNotIn("spectrum_bands", point)

    def test_z_only_spectrum_omits_fields(self) -> None:
        point = slim_event_point(
            {
                "created_at": "2026-01-01T00:00:00.000Z",
                "dBA_spl": 55.0,
                "spectrum": {
                    "z": {
                        "centroid_hz": 120.0,
                        "energy_pct": {"low": 90.0, "mid": 8.0, "high": 2.0},
                    },
                },
            }
        )
        self.assertIsNotNone(point)
        assert point is not None
        self.assertNotIn("centroid_hz", point)
        self.assertNotIn("spectrum_bands", point)


class OverviewWindowSeriesTests(unittest.TestCase):
    def _points(self, n: int = 300) -> list[dict]:
        from core.recording_overview import slim_event_point

        out: list[dict] = []
        for i in range(n):
            # 1 Hz points starting at midnight UTC
            sec = i
            p = slim_event_point(
                {
                    "created_at": f"2026-01-01T00:{sec // 60:02d}:{sec % 60:02d}.000Z",
                    "dBA_spl": 50.0 + (i % 10),
                    "LAFmax_dB": 55.0,
                    "top_label": "Silence",
                    "yamnet_preprocess": {
                        "gated": False,
                        "gate_open": True,
                        "ambient_noise_floor_dbfs": -70.0,
                    },
                }
            )
            assert p is not None
            out.append(p)
        return out

    def test_window_returns_about_two_minutes(self) -> None:
        from core.recording_overview import build_overview_window_series, created_at_to_ms

        points = self._points(400)
        center = created_at_to_ms(points[200]["t"])
        assert center is not None
        payload = build_overview_window_series(points, center_ms=center, window_s=120)
        self.assertTrue(payload["ok"])
        # 120 s window ≈ 121 samples inclusive
        self.assertGreaterEqual(len(payload["points"]), 115)
        self.assertLessEqual(len(payload["points"]), 125)
        first_ms = created_at_to_ms(payload["points"][0]["t"])
        last_ms = created_at_to_ms(payload["points"][-1]["t"])
        assert first_ms is not None and last_ms is not None
        self.assertLessEqual(first_ms, center)
        self.assertGreaterEqual(last_ms, center)

    def test_overview_scrub_is_always_downsampled_for_long(self) -> None:
        events = []
        for i in range(2500):
            events.append(
                {
                    "created_at": f"2026-01-01T{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}.000Z",
                    "dBA_spl": 50.0,
                    "LAFmax_dB": 55.0,
                    "top_label": "Silence",
                }
            )
        payload = build_recording_overview(
            events,
            name="long.jsonl",
            has_wav=False,
            wav_name=None,
            recording_active=False,
        )
        self.assertEqual(len(payload["scrub_points"]), 1500)
        self.assertFalse(payload["points_full"])
        self.assertEqual(payload["series_window_s"], 120)


if __name__ == "__main__":
    unittest.main()
