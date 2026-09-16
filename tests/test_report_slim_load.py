"""Tests for slim report JSONL loading helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from web.app import _iter_jsonl_events_for_report, _slim_event_for_report


class SlimReportEventTests(unittest.TestCase):
    def test_drops_spectrum_and_keeps_gate_flag(self) -> None:
        slim = _slim_event_for_report(
            {
                "created_at": "2026-09-16T12:00:00.000Z",
                "device_id": "pi-test",
                "dBA_spl": 48.0,
                "LAFmax_dB": 50.0,
                "top_label": "Vehicle",
                "top_confidence": 0.4,
                "spectrum": {"z": {"levels_db": [1, 2, 3]}},
                "predictions": [{"label": "Vehicle", "confidence": 0.4}],
                "yamnet_preprocess": {"gated": False, "applied_gain": 12.0},
                "clap_status": "skipped",
                "clap_predictions": [{"label": "x"}],
            }
        )
        self.assertNotIn("spectrum", slim)
        self.assertNotIn("predictions", slim)
        self.assertNotIn("clap_predictions", slim)
        self.assertEqual(slim["yamnet_preprocess"], {"gated": False})
        self.assertEqual(slim["top_label"], "Vehicle")

    def test_iter_loads_slim_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "created_at": "2026-09-16T12:00:00.000Z",
                            "dBA_spl": 40.0,
                            "spectrum": {"big": True},
                            "top_label": "Silence",
                            "top_confidence": 0.9,
                        }
                    )
                    + "\n"
                )
            events = _iter_jsonl_events_for_report(path)
            self.assertEqual(len(events), 1)
            self.assertNotIn("spectrum", events[0])
            self.assertEqual(events[0]["dBA_spl"], 40.0)


if __name__ == "__main__":
    unittest.main()
