"""Tests for live status series helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from web.app import _parse_event_time, _slim_series_point, _tail_jsonl_events


class StatusSeriesHelperTests(unittest.TestCase):
    def test_tail_jsonl_events_returns_last_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for i in range(10):
                    f.write(json.dumps({"chunk_index": i, "created_at": f"t{i}"}) + "\n")
            events = _tail_jsonl_events(path, max_lines=3)
            self.assertEqual(len(events), 3)
            self.assertEqual(events[0]["chunk_index"], 7)
            self.assertEqual(events[-1]["chunk_index"], 9)

    def test_slim_point_shifts_l90(self) -> None:
        point = _slim_series_point(
            {
                "created_at": "2026-09-11T14:00:00.000Z",
                "dBA_spl": 55.5,
                "top_label": "Vehicle",
                "yamnet_preprocess": {
                    "gated": False,
                    "ambient_noise_floor_dbfs": -70.0,
                },
                "clap_status": "triggered",
                "clap_top_label": "truck",
            },
            calib_offset=120.0,
        )
        assert point is not None
        self.assertEqual(point["dba"], 55.5)
        self.assertEqual(point["l90_rel"], 50.0)
        self.assertEqual(point["yamnet"], "Vehicle")
        self.assertFalse(point["gated"])
        self.assertEqual(point["clap"], "truck")

    def test_parse_event_time_zulu(self) -> None:
        dt = _parse_event_time("2026-09-11T14:00:00.000Z")
        assert dt is not None
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt, datetime(2026, 9, 11, 14, 0, tzinfo=timezone.utc))

    def test_parse_rejects_garbage(self) -> None:
        self.assertIsNone(_parse_event_time("not-a-time"))
        self.assertIsNone(_parse_event_time(None))


if __name__ == "__main__":
    unittest.main()
