"""Tests for dashboard report aggregation."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from core.audio_constants import YAMNET_CHUNK_DURATION_SECONDS
from core.report.aggregate import (
    MIN_HOUR_COVERAGE_SECONDS,
    build_dashboard_report,
    sound_diet_from_events,
)
from core.report.segment import AcousticEvent, energetic_leq
from core.report.votes import clear_label_map_cache

# Enough 1 Hz chunks for ≥ half an hour of nominal capture duration.
_CHUNKS_PER_COVERED_HOUR = int(
    MIN_HOUR_COVERAGE_SECONDS / YAMNET_CHUNK_DURATION_SECONDS
) + 1


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


def _covered_hour_events(
    base: datetime,
    *,
    dba: float = 50.0,
    label: str = "Vehicle",
    n: int = _CHUNKS_PER_COVERED_HOUR,
) -> list[dict]:
    """Build one calendar hour with enough chunks to pass the half-hour filter."""
    return [_evt(i, dba=dba, label=label, base=base) for i in range(n)]


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
        # Start at local 23:00 Amsterdam (CEST = UTC+2) → 21:00 UTC.
        base = datetime(2026, 9, 15, 21, 0, tzinfo=timezone.utc)
        events: list[dict] = []
        for hour_i in range(10):
            hour_base = base + timedelta(hours=hour_i)
            events.extend(_covered_hour_events(hour_base, dba=50.0))
        report = build_dashboard_report(
            [("overnight.jsonl", events)],
            timezone_name="Europe/Amsterdam",
        )
        timeline = report["zone_b"]["timeline"]
        self.assertGreaterEqual(len(timeline), 10)
        self.assertFalse(any(row.get("is_gap") for row in timeline))
        self.assertEqual(timeline[0]["hour"], 23)
        self.assertIsNotNone(timeline[0]["leq_db"])
        hours = [row["hour"] for row in timeline]
        self.assertIn(23, hours)
        self.assertIn(0, hours)
        self.assertTrue(any(row["label"].startswith("15 Sep") for row in timeline))
        self.assertTrue(any(row["label"].startswith("16 Sep") for row in timeline))

    def test_timeline_collapses_empty_hours_with_gap_marker(self) -> None:
        # 09:00 UTC = 10:00 Europe/Amsterdam (CET); 14:00 UTC = 15:00 local.
        early = datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc)
        later = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
        events_a = _covered_hour_events(early, dba=50.0)
        events_b = _covered_hour_events(later, dba=58.0)
        report = build_dashboard_report(
            [("morning.jsonl", events_a), ("afternoon.jsonl", events_b)],
            timezone_name="Europe/Amsterdam",
        )
        timeline = report["zone_b"]["timeline"]
        self.assertEqual(len(timeline), 3)
        self.assertEqual(timeline[0]["hour"], 10)
        self.assertFalse(timeline[0].get("is_gap"))
        self.assertTrue(timeline[1].get("is_gap"))
        self.assertEqual(timeline[1]["gap_hours"], 4)
        self.assertEqual(timeline[1]["label"], "4 h gap")
        self.assertEqual(timeline[2]["hour"], 15)
        self.assertIsNotNone(timeline[2]["leq_db"])

        budget_hours = report["zone_c"]["time_budget"]["hourly"]
        self.assertEqual(len(budget_hours), 3)
        self.assertTrue(budget_hours[1].get("is_gap"))
        self.assertEqual(budget_hours[1]["gap_hours"], 4)

    def test_thin_trailing_hour_omitted_from_timeline_and_budget(self) -> None:
        full = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        thin = datetime(2026, 1, 15, 13, 0, tzinfo=timezone.utc)
        events = _covered_hour_events(full, dba=50.0)
        events.extend(_evt(i, dba=90.0, base=thin) for i in range(20))  # ~20 s
        report = build_dashboard_report(
            [("thin-tail.jsonl", events)],
            timezone_name="UTC",
        )
        timeline = report["zone_b"]["timeline"]
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0]["hour"], 12)
        budget_hours = [
            h for h in report["zone_c"]["time_budget"]["hourly"] if not h.get("is_gap")
        ]
        self.assertEqual(len(budget_hours), 1)
        self.assertEqual(budget_hours[0]["hour"], 12)

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
        self.assertEqual(report["zone_a"]["briefing"], report["zone_c"]["takeaway"])
        self.assertIn("\n\n", report["zone_a"]["briefing"])

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
        self.assertEqual(zone_a["leq_context"], leq_context(zone_a["l50_db"]))
        self.assertIsNotNone(zone_a["duration_s"])
        self.assertGreater(zone_a["duration_s"], 0)
        comfort = zone_a["comfort_rating"]
        self.assertIsNotNone(comfort)
        self.assertIn(comfort["grade"], {"A", "B", "C", "D", "E"})
        self.assertTrue(0 <= comfort["score"] <= 100)
        tops = zone_a["top_disturbances"]
        self.assertTrue(1 <= len(tops) <= 3)
        self.assertEqual(tops[0]["lafmax_db"], zone_a["peak_lafmax_db"])
        self.assertEqual(tops[0]["source_file"], "levels.jsonl")
        self.assertIn("category", tops[0])
        self.assertIn("at_utc", tops[0])

    def test_top_disturbances_are_spaced_and_labelled(self) -> None:
        base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        events = [
            _evt(0, dba=90.0, label="Vehicle", base=base),
            _evt(1, dba=89.0, label="Vehicle", base=base),  # same minute cluster
            _evt(120, dba=80.0, label="Speech", base=base),
            _evt(240, dba=70.0, label="Bird", base=base),
            _evt(300, dba=50.0, label="Vehicle", base=base),
        ]
        report = build_dashboard_report(
            [("peaks.jsonl", events)],
            timezone_name="Europe/Amsterdam",
        )
        tops = report["zone_a"]["top_disturbances"]
        self.assertEqual(len(tops), 3)
        self.assertEqual(tops[0]["lafmax_db"], 93.0)  # 90 + 3
        self.assertEqual(tops[0]["category"], "Traffic & Transit")
        self.assertEqual(tops[1]["lafmax_db"], 83.0)
        self.assertEqual(tops[1]["category"], "Human & Community")
        self.assertEqual(tops[2]["lafmax_db"], 73.0)
        # Clustered second Vehicle peak (89) must not displace spaced ones.
        labels = {t["label"] for t in tops}
        self.assertIn("Vehicle", labels)
        self.assertIn("Speech", labels)
        snip = tops[0]["snippet"]
        self.assertIn("dba", snip)
        self.assertGreaterEqual(len(snip["dba"]), 1)
        self.assertIn("peak_index", snip)
        self.assertEqual(
            snip["dba"][snip["peak_index"]],
            max(snip["dba"]),
        )

    def test_hourly_gated_pct(self) -> None:
        base = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        n = _CHUNKS_PER_COVERED_HOUR
        events: list[dict] = []
        for i in range(n):
            if i % 2 == 0:
                events.append(
                    {
                        **_evt(i, dba=40.0, base=base),
                        "yamnet_preprocess": {"gated": True},
                        "top_label": "gated",
                    }
                )
            else:
                events.append(_evt(i, dba=55.0 + (i % 3), label="Vehicle", base=base))
        report = build_dashboard_report(
            [("gated.jsonl", events)],
            timezone_name="Europe/Amsterdam",
        )
        # 12:00 UTC = 13:00 Amsterdam in January
        hour13 = next(h for h in report["zone_b"]["typical_day"] if h["hour"] == 13)
        expected_gated = (n + 1) // 2  # i % 2 == 0 for i in range(n)
        self.assertEqual(hour13["chunk_count"], n)
        self.assertEqual(hour13["gated_count"], expected_gated)
        self.assertAlmostEqual(hour13["gated_pct"], 100.0 * expected_gated / n, places=1)
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
        # Silence last so stacked bars put it on top; Traffic (first macro) at bottom.
        self.assertEqual(budget["categories"][-1], "Relative silence")
        self.assertEqual(budget["categories"][0], "Traffic & Transit")
        self.assertTrue(budget["day"] or budget["night"])
        # Sparse selection: By-hour series drops under-covered hours; pies still work.
        self.assertEqual(budget["hourly"], [])

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
