"""Tests for CLAP ring buffer, triggers, and prompt helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.audio_ring_buffer import AudioRingBuffer
from core.clap_prompts import (
    ClapPromptPair,
    load_prompt_pairs,
    prompts_hash,
    save_prompt_pairs,
)
from core.clap_trigger import (
    CLAP_STATUS_CARRIED,
    CLAP_STATUS_GATED,
    CLAP_STATUS_SKIPPED,
    CLAP_STATUS_TRIGGERED,
    ClapTriggerConfig,
    ClapTriggerState,
    evaluate_trigger,
)
from core.classifier_clap import rebuild_text_embeddings


class AudioRingBufferTests(unittest.TestCase):
    def test_fills_and_snapshots_in_order(self) -> None:
        ring = AudioRingBuffer(maxlen_samples=8)
        ring.append(np.arange(5, dtype=np.float32))
        self.assertFalse(ring.full)
        ring.append(np.arange(5, 10, dtype=np.float32))
        self.assertTrue(ring.full)
        snap = ring.snapshot()
        self.assertEqual(snap.size, 8)
        np.testing.assert_array_equal(snap, np.arange(2, 10, dtype=np.float32))


class ClapTriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ClapTriggerConfig(
            cooldown_seconds=5.0,
            carry_ttl_seconds=30.0,
            dba_threshold=55.0,
            trigger_labels=["Vehicle", "Siren"],
            ambiguous_labels=["Noise"],
        )
        self.state = ClapTriggerState()

    def test_gated(self) -> None:
        run, status, reason = evaluate_trigger(
            gated=True,
            top_label="Vehicle",
            dba_spl=70.0,
            buffer_ready=True,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertFalse(run)
        self.assertEqual(status, CLAP_STATUS_GATED)
        self.assertEqual(reason, "gated")

    def test_direct_trigger(self) -> None:
        run, status, reason = evaluate_trigger(
            gated=False,
            top_label="Vehicle",
            dba_spl=40.0,
            buffer_ready=True,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertTrue(run)
        self.assertEqual(status, CLAP_STATUS_TRIGGERED)
        self.assertIn("Vehicle", reason)

    def test_ambiguous_needs_dba(self) -> None:
        run, status, _ = evaluate_trigger(
            gated=False,
            top_label="Noise",
            dba_spl=40.0,
            buffer_ready=True,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertFalse(run)
        self.assertEqual(status, CLAP_STATUS_SKIPPED)

        run2, status2, reason2 = evaluate_trigger(
            gated=False,
            top_label="Noise",
            dba_spl=60.0,
            buffer_ready=True,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertTrue(run2)
        self.assertEqual(status2, CLAP_STATUS_TRIGGERED)
        self.assertIn("ambiguous", reason2)

    def test_cooldown_carries(self) -> None:
        self.state.last_trigger_monotonic = 100.0
        self.state.last_result_monotonic = 100.0
        self.state.last_result = {
            "clap_predictions": [{"label": "car_passing", "confidence": 0.5}],
            "clap_top_label": "car_passing",
            "clap_top_confidence": 0.5,
            "clap_model_name": "x",
            "clap_model_version": "1",
        }
        run, status, reason = evaluate_trigger(
            gated=False,
            top_label="Vehicle",
            dba_spl=70.0,
            buffer_ready=True,
            config=self.cfg,
            state=self.state,
            now_monotonic=102.0,
        )
        self.assertFalse(run)
        self.assertEqual(status, CLAP_STATUS_CARRIED)
        self.assertEqual(reason, "cooldown")


class ClapPromptTests(unittest.TestCase):
    def test_roundtrip_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            pairs = [
                ClapPromptPair(label="a", prompt="Alpha sound outdoors"),
                ClapPromptPair(label="b", prompt="Beta sound outdoors"),
            ]
            save_prompt_pairs(pairs, path)
            loaded = load_prompt_pairs(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(prompts_hash(pairs), prompts_hash(loaded))

    def test_rebuild_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "prompts.json"
            embeds = Path(tmp) / "embeds.npz"
            save_prompt_pairs(
                [ClapPromptPair(label="siren", prompt="An outdoor siren")],
                prompts,
            )
            result = rebuild_text_embeddings(
                prompts_path=prompts, embeddings_path=embeds
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["count"], 1)
            self.assertTrue(embeds.exists())


if __name__ == "__main__":
    unittest.main()
