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

    def test_zone_a_percentiles_and_human_context(self) -> None:
        from core.report.aggregate import leq_context

        self.assertEqual(leq_context(35.0), "Quiet bedroom")
        self.assertEqual(leq_context(48.0), "Conversational backdrop")
        self.assertEqual(leq_context(62.0), "Busy street / TV")
        self.assertEqual(leq_context(75.0), "Lawnmower / peak traffic")

        # Ten ascending levels → L90≈quiet floor, L10≈loud spikes, L50≈median.
        base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        events = [_evt(i, dba=40.0 + i, base=base) for i in range(10)]
        report = build_dashboard_report(
            [("levels.jsonl", events)],
            timezone_name="Europe/Amsterdam",
            site_label="Anchor Site",
        )
        zone_a = report["zone_a"]
        self.assertEqual(zone_a["l90_db"], 40.0)
        self.assertEqual(zone_a["l50_db"], 44.0)
        self.assertEqual(zone_a["l10_db"], 48.0)
        self.assertNotIn("chunk", zone_a["header_line"].lower())
        self.assertEqual(zone_a["leq_context"], leq_context(zone_a["leq_db"]))
        self.assertIsNotNone(zone_a["duration_s"])
        self.assertGreater(zone_a["duration_s"], 0)

    def test_hourly_gated_pct(self) -> None:
        base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        events = [
            _evt(0, dba=40.0, label="gated", base=base),
            {
                **_evt(1, dba=40.0, base=base),
                "yamnet_preprocess": {"gated": True},
                "top_label": "gated",
            },
            _evt(2, dba=55.0, label="Vehicle", base=base),
            _evt(3, dba=56.0, label="Vehicle", base=base),
        ]
        # Force first event gated via top_label
        events[0]["top_label"] = "gated"
        events[0]["yamnet_preprocess"] = {"gated": True}
        report = build_dashboard_report(
            [("gated.jsonl", events)],
            timezone_name="Europe/Amsterdam",
        )
        # 12:00 UTC = 13:00 Amsterdam in January
        hour13 = next(h for h in report["zone_b"]["typical_day"] if h["hour"] == 13)
        self.assertEqual(hour13["chunk_count"], 4)
        self.assertEqual(hour13["gated_count"], 2)
        self.assertEqual(hour13["gated_pct"], 50.0)
        self.assertIsNotNone(hour13["l90_db"])
        self.assertIsNotNone(hour13["l10_db"])
        self.assertLessEqual(hour13["l90_db"], hour13["l10_db"])
        timeline_row = report["zone_b"]["timeline"][0]
        self.assertIn("date_short", timeline_row)
        self.assertEqual(timeline_row["gated_pct"], 50.0)
        self.assertEqual(timeline_row["l90_db"], hour13["l90_db"])

    def test_time_budget_includes_relative_silence(self) -> None:
        base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        events = [
            {
                **_evt(0, dba=40.0, base=base),
                "yamnet_preprocess": {"gated": True},
                "top_label": "gated",
            },
            {
                **_evt(1, dba=40.0, base=base),
                "yamnet_preprocess": {"gated": True},
                "top_label": "gated",
            },
            _evt(2, dba=55.0, label="Vehicle", base=base),
            _evt(3, dba=56.0, label="Vehicle", base=base),
        ]
        report = build_dashboard_report(
            [("budget.jsonl", events)],
            timezone_name="Europe/Amsterdam",
        )
        budget = report["zone_c"]["time_budget"]
        by_cat = {r["category"]: r for r in budget["total"]}
        self.assertIn("Relative silence", by_cat)
        self.assertAlmostEqual(by_cat["Relative silence"]["pct"], 50.0, places=0)
        self.assertTrue(budget["day"] or budget["night"])
        self.assertTrue(budget["hourly"])

    def test_l90_offset_threshold_mode(self) -> None:
        base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        # L90 ≈ 40; with +5 offset → threshold 45. Levels 42 stay inactive.
        events = [_evt(i, dba=40.0 + (i % 3), label="Vehicle", base=base) for i in range(10)]
        events.extend([_evt(20 + i, dba=60.0, label="Vehicle", base=base) for i in range(3)])
        report = build_dashboard_report(
            [("l90.jsonl", events)],
            timezone_name="Europe/Amsterdam",
            threshold_mode="l90_offset",
            l90_offset_db=5.0,
            min_event_chunks=2,
        )
        opts = report["meta"]["report_options"]
        self.assertEqual(opts["threshold_mode"], "l90_offset")
        self.assertIsNotNone(opts["l90_db_used"])
        self.assertAlmostEqual(
            opts["active_dba_threshold"],
            float(opts["l90_db_used"]) + 5.0,
            places=1,
        )
        self.assertGreaterEqual(report["meta"]["acoustic_event_count"], 1)

    def test_min_event_chunks_override(self) -> None:
        base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        events = [
            _evt(0, dba=60.0, label="Vehicle", base=base),
            _evt(1, dba=40.0, label="Silence", base=base),
            _evt(2, dba=40.0, label="Silence", base=base),
            _evt(3, dba=40.0, label="Silence", base=base),
        ]
        keep = build_dashboard_report(
            [("one.jsonl", events)],
            timezone_name="Europe/Amsterdam",
            min_event_chunks=1,
            max_gap_chunks=0,
            threshold_mode="absolute",
            threshold_db=45.0,
        )
        drop = build_dashboard_report(
            [("one.jsonl", events)],
            timezone_name="Europe/Amsterdam",
            min_event_chunks=2,
            max_gap_chunks=0,
            threshold_mode="absolute",
            threshold_db=45.0,
        )
        self.assertEqual(keep["meta"]["acoustic_event_count"], 1)
        self.assertEqual(drop["meta"]["acoustic_event_count"], 0)


if __name__ == "__main__":
    unittest.main()
