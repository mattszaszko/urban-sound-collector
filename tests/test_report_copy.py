"""Tests for deterministic soundscape briefing copy."""

from __future__ import annotations

import unittest

from core.report.copy import build_takeaway


class BuildTakeawayTests(unittest.TestCase):
    def test_briefing_covers_character_rhythm_and_effect(self) -> None:
        zone_a = {
            "l50_db": 54.8,
            "l90_db": 43.9,
            "l10_db": 62.4,
            "leq_context": "Conversational backdrop",
            "comfort_rating": {
                "grade": "E",
                "label": "Intense",
                "insufficient_data": False,
            },
            "top_disturbances": [
                {
                    "lafmax_db": 97.9,
                    "category": "Infrastructure & Mechanical",
                    "at_local": "2026-10-05 21:11:28",
                }
            ],
        }
        zone_b = {
            "typical_day": [
                {"hour": 16, "period": "day", "leq_db": 68.0},
                {"hour": 17, "period": "day", "leq_db": 67.0},
                {"hour": 3, "period": "night", "leq_db": 48.0},
                {"hour": 4, "period": "night", "leq_db": 46.0},
            ]
        }
        zone_c = {
            "time_budget": {
                "total": [
                    {"category": "Traffic & Transit", "pct": 42.0},
                    {"category": "Unclassified / Ambient", "pct": 12.0},
                    {"category": "Relative silence", "pct": 35.0},
                ]
            }
        }
        text = build_takeaway(zone_a=zone_a, zone_b=zone_b, zone_c=zone_c)
        lowered = text.lower()
        self.assertIn("acoustic comfort e (intense)", lowered)
        self.assertIn("54.8 dBA", text)
        self.assertIn("Traffic & Transit", text)
        self.assertIn("16:00", text)
        self.assertIn("46 dBA", text)
        self.assertIn("97.9 dBA", text)
        self.assertIn("Inspect", text)
        self.assertGreaterEqual(text.count("\n\n"), 2)

    def test_sparse_selection_still_returns_text(self) -> None:
        text = build_takeaway(
            zone_a={"l50_db": 40.0, "leq_context": "Quiet bedroom"},
            zone_b={"typical_day": []},
            zone_c={"time_budget": {"total": []}},
        )
        self.assertTrue(text)
        self.assertIn("40.0 dBA", text)


if __name__ == "__main__":
    unittest.main()
