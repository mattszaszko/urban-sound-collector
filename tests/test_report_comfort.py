"""Tests for Acoustic Comfort Rating."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.report.comfort import (
    DAY_CAP,
    NIGHT_CAP,
    VOLATILITY_CAP,
    build_comfort_rating,
    day_penalty,
    grade_from_score,
    night_penalty,
    volatility_penalty,
)
from core.report.segment import AcousticChunk


def _chunk(
    hour_utc: int,
    *,
    dba: float,
    gated: bool = False,
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
        vote_label=None if gated else "Vehicle",
        gated=gated,
        device_id="pi-test",
    )


class ComfortPenaltyTests(unittest.TestCase):
    def test_night_penalty_example(self) -> None:
        # 48 dBA → (48-40)*2 = 16
        self.assertEqual(night_penalty(48.0), 16.0)

    def test_day_penalty_example(self) -> None:
        # 54 dBA → (54-45)*1.5 = 13.5
        self.assertEqual(day_penalty(54.0), 13.5)

    def test_volatility_penalty_example(self) -> None:
        self.assertEqual(volatility_penalty(22.0), 12.0)

    def test_penalty_caps(self) -> None:
        self.assertEqual(night_penalty(100.0), NIGHT_CAP)
        self.assertEqual(day_penalty(100.0), DAY_CAP)
        self.assertEqual(volatility_penalty(100.0), VOLATILITY_CAP)

    def test_no_penalty_below_refs(self) -> None:
        self.assertEqual(night_penalty(40.0), 0.0)
        self.assertEqual(day_penalty(45.0), 0.0)
        self.assertEqual(volatility_penalty(10.0), 0.0)

    def test_grade_bands(self) -> None:
        self.assertEqual(grade_from_score(100), ("A", "Tranquil"))
        self.assertEqual(grade_from_score(85), ("A", "Tranquil"))
        self.assertEqual(grade_from_score(84), ("B", "Balanced"))
        self.assertEqual(grade_from_score(70), ("B", "Balanced"))
        self.assertEqual(grade_from_score(69), ("C", "Active"))
        self.assertEqual(grade_from_score(55), ("C", "Active"))
        self.assertEqual(grade_from_score(54), ("D", "Vibrant"))
        self.assertEqual(grade_from_score(40), ("D", "Vibrant"))
        self.assertEqual(grade_from_score(39), ("E", "Intense"))
        self.assertEqual(grade_from_score(0), ("E", "Intense"))


class ComfortRatingIntegrationTests(unittest.TestCase):
    def test_quiet_full_coverage(self) -> None:
        # Day 13:00 local = 12:00 UTC; night 02:00 local = 01:00 UTC
        chunks = [
            _chunk(12, dba=42.0, gated=True),
            _chunk(12, dba=43.0, gated=True, minute=1),
            _chunk(1, dba=38.0, gated=True),
            _chunk(1, dba=39.0, gated=True, minute=1),
        ]
        rating = build_comfort_rating(chunks)
        assert rating is not None
        self.assertFalse(rating["insufficient_data"])
        self.assertTrue(rating["coverage"]["day"])
        self.assertTrue(rating["coverage"]["night"])
        self.assertEqual(rating["penalties"]["scale"], 1.0)
        self.assertGreaterEqual(rating["score"], 85)
        self.assertEqual(rating["grade"], "A")

    def test_day_only_rescale_and_warning(self) -> None:
        # All daytime (13:00 local)
        chunks = [
            _chunk(12, dba=54.0, gated=False),
            _chunk(12, dba=54.0, gated=False, minute=1),
            _chunk(12, dba=54.0, gated=True, minute=2),
            _chunk(12, dba=54.0, gated=True, minute=3),
        ]
        # noisy = 50%, vol raw = 40 capped to 25
        # day raw = (54-45)*1.5 = 13.5
        # caps = 35+25 = 60, scale = 100/60
        rating = build_comfort_rating(chunks)
        assert rating is not None
        self.assertTrue(rating["insufficient_data"])
        self.assertTrue(rating["coverage"]["day"])
        self.assertFalse(rating["coverage"]["night"])
        self.assertIsNone(rating["inputs"]["l_night_db"])
        self.assertAlmostEqual(rating["penalties"]["scale"], 100.0 / 60.0, places=3)
        self.assertIn("daytime data only", rating["warning"].lower())
        raw_day = rating["penalties"]["raw"]["day"]
        raw_vol = rating["penalties"]["raw"]["volatility"]
        assert raw_day is not None and raw_vol is not None
        expected = round(100 - (raw_day + raw_vol) * (100.0 / 60.0))
        expected = max(0, min(100, expected))
        self.assertEqual(rating["score"], expected)
        self.assertEqual(rating["penalties"]["night"], 0.0)

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(build_comfort_rating([]))


if __name__ == "__main__":
    unittest.main()
