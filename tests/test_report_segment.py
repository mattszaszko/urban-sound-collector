"""Tests for acoustic event segmentation."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from core.report.segment import (
    AcousticChunk,
    prepare_chunks,
    segment_acoustic_events,
)
from core.report.votes import clear_label_map_cache, vote_label_for_event


def _chunk(
    i: int,
    *,
    dba: float,
    label: str = "Vehicle",
    conf: float = 0.5,
    gated: bool = False,
    base: datetime | None = None,
) -> dict:
    start = base or datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
    ts = start + timedelta(seconds=i)
    return {
        "created_at": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "dBA_spl": dba,
        "LAFmax_dB": dba + 2,
        "top_label": "gated" if gated else label,
        "top_confidence": 1.0 if gated else conf,
        "yamnet_preprocess": {"gated": gated},
        "device_id": "pi-test",
    }


class SegmentAcousticEventsTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_label_map_cache()
        self.zone = ZoneInfo("Europe/Amsterdam")

    def _prepare(self, events: list[dict]):
        return prepare_chunks(events, zone=self.zone)

    def test_flickering_labels_one_event(self) -> None:
        labels = ["Wind", "Vehicle", "Vehicle horn, car horn, honking", "Vehicle"]
        events = [_chunk(i, dba=60.0, label=labels[i]) for i in range(4)]
        chunks = self._prepare(events)
        segs = segment_acoustic_events(chunks)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].chunk_count, 4)

    def test_one_quiet_chunk_bridged(self) -> None:
        # High, high, quiet, high, high — one quiet gap bridged (max_gap=2)
        events = [
            _chunk(0, dba=60.0, label="Vehicle"),
            _chunk(1, dba=58.0, label="Vehicle"),
            _chunk(2, dba=40.0, label="Silence"),  # below 45
            _chunk(3, dba=59.0, label="Vehicle"),
            _chunk(4, dba=57.0, label="Vehicle"),
        ]
        chunks = self._prepare(events)
        segs = segment_acoustic_events(chunks)
        self.assertEqual(len(segs), 1, "1s quiet dip must not split the event")
        self.assertGreaterEqual(segs[0].bridged_gap_chunks, 1)

    def test_three_quiet_chunks_split(self) -> None:
        events = [
            _chunk(0, dba=60.0, label="Vehicle"),
            _chunk(1, dba=58.0, label="Vehicle"),
            _chunk(2, dba=40.0, label="Silence"),
            _chunk(3, dba=40.0, label="Silence"),
            _chunk(4, dba=40.0, label="Silence"),
            _chunk(5, dba=59.0, label="Vehicle"),
            _chunk(6, dba=57.0, label="Vehicle"),
        ]
        chunks = self._prepare(events)
        segs = segment_acoustic_events(chunks)
        self.assertEqual(len(segs), 2)

    def test_tire_hiss_override_on_wind_majority(self) -> None:
        events = [
            _chunk(0, dba=58.0, label="Wind"),
            _chunk(1, dba=60.0, label="Wind"),
            _chunk(2, dba=55.0, label="White noise"),
            _chunk(3, dba=57.0, label="Wind"),
        ]
        chunks = self._prepare(events)
        segs = segment_acoustic_events(chunks)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].resolved_label, "Vehicle / Tire Rolling")
        self.assertEqual(segs[0].macro, "Traffic & Transit")


class VoteEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_label_map_cache()

    def test_gated_and_blacklist_excluded(self) -> None:
        self.assertIsNone(
            vote_label_for_event(
                {
                    "top_label": "Vehicle",
                    "top_confidence": 0.9,
                    "yamnet_preprocess": {"gated": True},
                }
            )
        )
        self.assertIsNone(
            vote_label_for_event(
                {
                    "top_label": "Printer",
                    "top_confidence": 0.9,
                    "yamnet_preprocess": {"gated": False},
                }
            )
        )
        self.assertEqual(
            vote_label_for_event(
                {
                    "top_label": "Vehicle",
                    "top_confidence": 0.4,
                    "yamnet_preprocess": {"gated": False},
                }
            ),
            "Vehicle",
        )


if __name__ == "__main__":
    unittest.main()
