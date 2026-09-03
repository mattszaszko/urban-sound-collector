"""Tests for JSONL export field filtering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.export_filter import (
    export_filename,
    filter_event_for_export,
    iter_filtered_jsonl,
)


class ExportFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = {
            "device_id": "pi-test-01",
            "dBA_spl": 70.0,
            "top_label": "Vehicle",
            "spectrum": {"band_type": "third_octave"},
            "yamnet_preprocess": {"gate_mode": "dynamic_l90"},
        }

    def test_keep_all_fields(self) -> None:
        out = filter_event_for_export(
            self.event,
            include_spectrum=True,
            include_yamnet_preprocess=True,
        )
        self.assertIn("spectrum", out)
        self.assertIn("yamnet_preprocess", out)
        self.assertEqual(out["dBA_spl"], 70.0)

    def test_omit_spectrum(self) -> None:
        out = filter_event_for_export(
            self.event,
            include_spectrum=False,
            include_yamnet_preprocess=True,
        )
        self.assertNotIn("spectrum", out)
        self.assertIn("yamnet_preprocess", out)

    def test_omit_preprocess(self) -> None:
        out = filter_event_for_export(
            self.event,
            include_spectrum=True,
            include_yamnet_preprocess=False,
        )
        self.assertIn("spectrum", out)
        self.assertNotIn("yamnet_preprocess", out)

    def test_omit_both(self) -> None:
        out = filter_event_for_export(
            self.event,
            include_spectrum=False,
            include_yamnet_preprocess=False,
        )
        self.assertNotIn("spectrum", out)
        self.assertNotIn("yamnet_preprocess", out)
        self.assertEqual(out["top_label"], "Vehicle")

    def test_does_not_mutate_original(self) -> None:
        filter_event_for_export(
            self.event,
            include_spectrum=False,
            include_yamnet_preprocess=False,
        )
        self.assertIn("spectrum", self.event)
        self.assertIn("yamnet_preprocess", self.event)

    def test_iter_filtered_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text(
                json.dumps(self.event) + "\n" + json.dumps(self.event) + "\n",
                encoding="utf-8",
            )
            lines = list(
                iter_filtered_jsonl(
                    path,
                    include_spectrum=False,
                    include_yamnet_preprocess=False,
                )
            )
            self.assertEqual(len(lines), 2)
            parsed = json.loads(lines[0])
            self.assertNotIn("spectrum", parsed)
            self.assertNotIn("yamnet_preprocess", parsed)

    def test_export_filename(self) -> None:
        self.assertEqual(
            export_filename(
                "evening.jsonl",
                include_spectrum=True,
                include_yamnet_preprocess=True,
            ),
            "evening.jsonl",
        )
        self.assertEqual(
            export_filename(
                "evening.jsonl",
                include_spectrum=False,
                include_yamnet_preprocess=True,
            ),
            "evening-no-spectrum.jsonl",
        )
        self.assertEqual(
            export_filename(
                "evening.jsonl",
                include_spectrum=False,
                include_yamnet_preprocess=False,
            ),
            "evening-no-spectrum-no-preprocess.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
