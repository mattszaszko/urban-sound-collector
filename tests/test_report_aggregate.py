"""Tests for dashboard report aggregation."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from core.report.aggregate import build_dashboard_report, sound_diet_from_events
from core.report.segment import AcousticEvent, energetic_leq
from core.report.votes import clear_label_map_cache


def _evt(
    i: int,
    *,
    dba: float,
    label: str = "Vehicle",
    conf: float = 0.5,
    base: datetime | None = None,
) -> dict:
    start = base or datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    ts = start + timedelta(seconds=i)
    return {
        "created_at": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "dBA_spl": dba,
        "LAFmax_dB": dba + 3,
        "top_label": label,
        "top_confidence": conf,
        "yamnet_preprocess": {"gated": False},
        "device_id": "pi-ams",
    }


class AggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_label_map_cache()

    def test_energetic_leq(self) -> None:
        # Two equal levels → same
        self.assertEqual(energetic_leq([50.0, 50.0]), 50.0)
        # 60 and 40 → closer to 60
        leq = energetic_leq([60.0, 40.0])
        assert leq is not None
        self.assertGreater(leq, 55.0)

    def test_duration_weighted_diet_not_event_count(self) -> None:
        bird = AcousticEvent(
            start_utc="a",
            end_utc="a",
            start_local="a",
            end_local="a",
            duration_s=1.0,
            chunk_count=1,
            max_dba=50.0,
            max_lafmax=50.0,
            leq_db=50.0,
            dominant_label="Bird",
            resolved_label="Bird",
            macro="Human & Community",
            vote_labels=["Bird"],
        )
        truck = AcousticEvent(
            start_utc="b",
            end_utc="b",
            start_local="b",
            end_local="b",
            duration_s=30.0,
            chunk_count=30,
            max_dba=70.0,
            max_lafmax=72.0,
            leq_db=68.0,
            dominant_label="Vehicle",
            resolved_label="Vehicle",
            macro="Traffic & Transit",
            vote_labels=["Vehicle"],
        )
        diet = sound_diet_from_events([bird, truck])
        by_macro = {r["macro"]: r for r in diet}
        self.assertAlmostEqual(by_macro["Traffic & Transit"]["pct"], 96.8, places=0)
        self.assertEqual(by_macro["Traffic & Transit"]["events"], 1)
        self.assertEqual(by_macro["Human & Community"]["events"], 1)

    def test_amsterdam_hour_bucket(self) -> None:
        # 2026-01-15 12:00 UTC = 13:00 Europe/Amsterdam (CET winter)
        base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        events = [_evt(i, dba=55.0 + (i % 3), base=base) for i in range(5)]
        # Quiet filler so we still have acoustic track
        report = build_dashboard_report(
            [("run.jsonl", events)],
            timezone_name="Europe/Amsterdam",
            site_label="Test Site",
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["meta"]["timezone"], "Europe/Amsterdam")
        hourly = report["zone_b"]["typical_day"]
        hour13 = next(h for h in hourly if h["hour"] == 13)
        self.assertIsNotNone(hour13["leq_db"])
        hour12 = next(h for h in hourly if h["hour"] == 12)
        self.assertIsNone(hour12["leq_db"])
        self.assertEqual(report["zone_b"]["default_view"], "timeline")
        self.assertEqual(report["zone_b"]["hourly"], report["zone_b"]["typical_day"])

    def test_timeline_crosses_midnight(self) -> None:
        # 21:30 UTC in mid-Sep = 23:30 Europe/Amsterdam (CEST).
        base = datetime(2026, 9, 15, 21, 30, tzinfo=timezone.utc)
        events = []
        for i in range(0, 10 * 3600, 60):  # 10 hours of minute samples
            events.append(_evt(i, dba=50.0, base=base))
        report = build_dashboard_report(
            [("overnight.jsonl", events)],
            timezone_name="Europe/Amsterdam",
        )
        timeline = report["zone_b"]["timeline"]
        self.assertGreaterEqual(len(timeline), 10)
        self.assertEqual(timeline[0]["hour"], 23)
        self.assertIsNotNone(timeline[0]["leq_db"])
        hours = [row["hour"] for row in timeline]
        self.assertIn(23, hours)
        self.assertIn(0, hours)
        self.assertTrue(any(row["label"].startswith("15 Sep") for row in timeline))
        self.assertTrue(any(row["label"].startswith("16 Sep") for row in timeline))

    def test_gated_still_in_leq_not_in_votes(self) -> None:
        events = [
            {
                "created_at": "2026-01-15T12:00:00.000Z",
                "dBA_spl": 35.0,
                "LAFmax_dB": 36.0,
                "top_label": "gated",
                "top_confidence": 1.0,
                "yamnet_preprocess": {"gated": True},
                "device_id": "pi-ams",
            },
            _evt(1, dba=60.0, label="Vehicle"),
            _evt(2, dba=58.0, label="Vehicle"),
        ]
        report = build_dashboard_report(
            [("run.jsonl", events)],
            timezone_name="Europe/Amsterdam",
        )
        self.assertEqual(report["zone_a"]["chunk_count"], 3)
        self.assertIsNotNone(report["zone_a"]["leq_db"])
        self.assertTrue(report["zone_c"]["takeaway"])


if __name__ == "__main__":
    unittest.main()
