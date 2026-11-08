"""Tests for Lived Experience & Health Impact builder."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.report.aggregate import build_dashboard_report
from core.report.lived_experience import (
    ATTEN_CLOSED_DB,
    ATTEN_OPEN_DB,
    build_lived_experience,
    compute_lden_db,
    indoor_from_exterior,
)
from core.report.segment import AcousticChunk
from core.report.votes import clear_label_map_cache


def _chunk(
    hour_utc: int,
    *,
    dba: float,
    minute: int = 0,
    day: int = 15,
) -> AcousticChunk:
    # January: Amsterdam = UTC+1 → local hour = hour_utc + 1
    dt_utc = datetime(2026, 1, day, hour_utc, minute, tzinfo=timezone.utc)
    dt_local = dt_utc.astimezone(ZoneInfo("Europe/Amsterdam"))
    return AcousticChunk(
        created_at=dt_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        dt_utc=dt_utc,
        dt_local=dt_local,
        dba=dba,
        lafmax=dba + 2,
        vote_label="Vehicle",
        gated=False,
        device_id="pi-test",
    )


def _evt(hour_utc: int, *, dba: float, i: int = 0) -> dict:
    dt = datetime(2026, 1, 15, hour_utc, 0, i % 60, tzinfo=timezone.utc)
    return {
        "created_at": dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "dBA_spl": dba,
        "LAFmax_dB": dba + 5,
        "top_label": "Vehicle",
        "top_confidence": 0.6,
        "yamnet_preprocess": {"gated": False},
        "device_id": "pi-ams",
    }


class AttenuationTests(unittest.TestCase):
    def test_closed_and_open(self) -> None:
        self.assertEqual(indoor_from_exterior(60.0, ATTEN_CLOSED_DB), 30.0)
        self.assertEqual(indoor_from_exterior(60.0, ATTEN_OPEN_DB), 45.0)

    def test_floor_clamp(self) -> None:
        self.assertEqual(indoor_from_exterior(40.0, ATTEN_CLOSED_DB), 20.0)

    def test_none_passthrough(self) -> None:
        self.assertIsNone(indoor_from_exterior(None, ATTEN_CLOSED_DB))


class LdenHelperTests(unittest.TestCase):
    def test_full_day_evening_night(self) -> None:
        # local 12:00 = UTC 11; local 20:00 = UTC 19; local 02:00 = UTC 01
        chunks = [
            _chunk(11, dba=50.0),
            _chunk(19, dba=48.0),
            _chunk(1, dba=42.0),
        ]
        m = compute_lden_db(chunks)
        self.assertTrue(m["has_day"])
        self.assertTrue(m["has_evening"])
        self.assertTrue(m["has_night"])
        self.assertEqual(m["lday_db"], 50.0)
        self.assertEqual(m["levening_db"], 48.0)
        self.assertEqual(m["lnight_db"], 42.0)
        assert m["lden_db"] is not None
        # Night +10 and evening +5 pull Lden above the day mean.
        self.assertGreater(m["lden_db"], 50.0)

    def test_missing_evening_renormalizes(self) -> None:
        chunks = [
            _chunk(11, dba=50.0),
            _chunk(1, dba=40.0),
        ]
        m = compute_lden_db(chunks)
        self.assertTrue(m["has_day"])
        self.assertFalse(m["has_evening"])
        self.assertTrue(m["has_night"])
        assert m["lden_db"] is not None
        # 12h @ 50 + 8h @ 50 (40+10) → energetic mean of equal 50 terms
        self.assertAlmostEqual(m["lden_db"], 50.0, places=1)

    def test_night_only(self) -> None:
        chunks = [_chunk(1, dba=45.0)]
        m = compute_lden_db(chunks)
        self.assertFalse(m["has_day"])
        self.assertFalse(m["has_evening"])
        self.assertTrue(m["has_night"])
        # Night-only Lden = night + 10 after weight renormalization to 8h
        self.assertEqual(m["lden_db"], 55.0)


class BadgeBoundaryTests(unittest.TestCase):
    def test_sleep_well_protected(self) -> None:
        # night 55 → closed 25, open 40
        lived = build_lived_experience([_chunk(1, dba=55.0)])
        assert lived is not None
        sleep = next(d for d in lived["dimensions"] if d["id"] == "sleep")
        self.assertEqual(sleep["badge"]["label"], "Well protected")
        self.assertEqual(sleep["badge"]["tone"], "ok")

    def test_sleep_protected_if_closed(self) -> None:
        # night 58 → closed 28, open 43
        lived = build_lived_experience([_chunk(1, dba=58.0)])
        assert lived is not None
        sleep = next(d for d in lived["dimensions"] if d["id"] == "sleep")
        self.assertEqual(sleep["badge"]["label"], "Protected if windows closed")
        self.assertEqual(sleep["badge"]["tone"], "watch")

    def test_sleep_high_disruption(self) -> None:
        # night 75 → closed 45, open 60
        lived = build_lived_experience([_chunk(1, dba=75.0)])
        assert lived is not None
        sleep = next(d for d in lived["dimensions"] if d["id"] == "sleep")
        self.assertEqual(sleep["badge"]["label"], "High sleep disruption risk")
        self.assertEqual(sleep["badge"]["tone"], "alert")

    def test_office_focus_friendly(self) -> None:
        # day-only 60 → lden 60 → closed 30, open 45
        lived = build_lived_experience([_chunk(11, dba=60.0)])
        assert lived is not None
        office = next(d for d in lived["dimensions"] if d["id"] == "office")
        self.assertEqual(office["badge"]["label"], "Good with windows closed")
        self.assertEqual(office["badge"]["tone"], "ok")

    def test_office_distracting_even_closed(self) -> None:
        lived = build_lived_experience([_chunk(11, dba=85.0)])
        assert lived is not None
        office = next(d for d in lived["dimensions"] if d["id"] == "office")
        self.assertEqual(office["badge"]["label"], "Distracting even closed")
        self.assertEqual(office["badge"]["tone"], "alert")

    def test_baby_peak_alert(self) -> None:
        lived = build_lived_experience(
            [_chunk(1, dba=50.0)],
            top_disturbances=[{"lafmax_db": 78.0}],
        )
        assert lived is not None
        baby = next(d for d in lived["dimensions"] if d["id"] == "baby")
        self.assertEqual(baby["badge"]["label"], "Sensitive Environment Alert")
        self.assertEqual(baby["badge"]["tone"], "alert")

    def test_payload_keeps_lden_internal(self) -> None:
        lived = build_lived_experience(
            [_chunk(11, dba=50.0), _chunk(1, dba=42.0)]
        )
        assert lived is not None
        self.assertIn("lden_db", lived["metrics"])
        self.assertIsNotNone(lived["metrics"]["lden_db"])
        # Dimension summaries must not advertise Lden to the UI layer.
        blob = " ".join(d["summary"] for d in lived["dimensions"]).lower()
        self.assertNotIn("lden", blob)
        self.assertNotIn("l_den", blob)


class AggregateSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_label_map_cache()

    def test_zone_a_includes_lived_experience(self) -> None:
        # Mix day + night so sleep and office both form.
        day = [_evt(11, dba=52.0, i=i) for i in range(30)]
        night = [_evt(1, dba=44.0, i=i) for i in range(30)]
        report = build_dashboard_report(
            [("rec.jsonl", day + night)],
            timezone_name="Europe/Amsterdam",
        )
        lived = report["zone_a"].get("lived_experience")
        self.assertIsNotNone(lived)
        assert lived is not None
        self.assertEqual(len(lived["dimensions"]), 3)
        self.assertIn("metrics", lived)
        self.assertIn("lden_db", lived["metrics"])


if __name__ == "__main__":
    unittest.main()
