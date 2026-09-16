"""Tests for JSONL recording_id helpers."""

from __future__ import annotations

import unittest

from core.events import build_noise_event, event_recording_id, new_recording_id


class EventRecordingIdTests(unittest.TestCase):
    def test_prefers_recording_id(self) -> None:
        event = {"recording_id": "new", "run_id": "old"}
        self.assertEqual(event_recording_id(event), "new")

    def test_falls_back_to_legacy_run_id(self) -> None:
        self.assertEqual(event_recording_id({"run_id": "legacy"}), "legacy")

    def test_missing(self) -> None:
        self.assertIsNone(event_recording_id({}))
        self.assertIsNone(event_recording_id(None))

    def test_build_noise_event_writes_recording_id(self) -> None:
        rid = new_recording_id()
        event = build_noise_event(
            device_id="pi-test",
            chunk_index=0,
            recording_id=rid,
            rms_unweighted=0.1,
            rms_a_weighted=0.05,
            dba_spl=60.0,
            predictions=[{"label": "Silence", "confidence": 0.9}],
        )
        self.assertEqual(event["recording_id"], rid)
        self.assertNotIn("run_id", event)


if __name__ == "__main__":
    unittest.main()
