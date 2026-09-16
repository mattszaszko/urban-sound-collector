"""Tests for Data-tab recording list formatting helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from web.app import _format_bytes, _format_duration, _jsonl_list_stats


class FormatDurationTests(unittest.TestCase):
    def test_under_one_minute(self) -> None:
        self.assertEqual(_format_duration(0), "0 min")
        self.assertEqual(_format_duration(45), "< 1 min")

    def test_minutes_only(self) -> None:
        self.assertEqual(_format_duration(60), "1 min")
        self.assertEqual(_format_duration(12 * 60 + 30), "12 min")

    def test_hours_and_minutes(self) -> None:
        self.assertEqual(_format_duration(3 * 3600 + 12 * 60), "3 h 12 min")
        self.assertEqual(_format_duration(3600), "1 h 0 min")

    def test_days_hours_minutes(self) -> None:
        self.assertEqual(
            _format_duration(2 * 86400 + 3 * 3600 + 12 * 60),
            "2 d 3 h 12 min",
        )


class FormatBytesTests(unittest.TestCase):
    def test_scales(self) -> None:
        self.assertEqual(_format_bytes(500), "500 B")
        self.assertEqual(_format_bytes(2048), "2.0 KB")
        self.assertEqual(_format_bytes(3 * 1024 * 1024), "3.0 MB")
        self.assertEqual(_format_bytes(2 * 1024 * 1024 * 1024), "2.0 GB")


class JsonlListStatsTests(unittest.TestCase):
    def test_first_last_and_count_from_chunk_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "created_at": "2026-09-16T08:00:00.000Z",
                            "chunk_index": 0,
                        }
                    )
                    + "\n"
                )
                # Large middle rows should not be fully scanned for timestamps.
                for i in range(1, 50):
                    f.write(
                        json.dumps(
                            {
                                "created_at": f"2026-09-16T08:00:{i:02d}.000Z",
                                "chunk_index": i,
                            }
                        )
                        + "\n"
                    )
                f.write(
                    json.dumps(
                        {
                            "created_at": "2026-09-16T10:15:00.000Z",
                            "chunk_index": 50,
                        }
                    )
                    + "\n"
                )
            lines, first, last = _jsonl_list_stats(path)
            self.assertEqual(lines, 51)
            assert first is not None and last is not None
            self.assertEqual((last - first).total_seconds(), 2 * 3600 + 15 * 60)


if __name__ == "__main__":
    unittest.main()
