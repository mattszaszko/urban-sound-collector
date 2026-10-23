"""Unit tests for DASM Phase 1 core (no ONNX required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.classifier_dasm import frames_to_spans
from core.clap_event_buffer import ChunkTelemetry
from core.dasm_capture import CaptureState, FixedDasmCapture
from core.dasm_onnx import (
    OPEN_VOCAB_SLOTS,
    build_tgt_mask,
    pad_open_vocab_queries,
)
from core.dasm_preprocess import log_mel_spectrogram, pad_or_truncate
from core.dasm_prompts import (
    DasmPromptPair,
    EXPECTED_MODEL_NAME,
    MAX_OPEN_VOCAB_PROMPTS,
    embeddings_compatible,
    load_embeddings,
    load_window_config,
    prompts_hash,
    save_embeddings,
    save_prompt_pairs,
    save_window_config,
)
from core.events import build_noise_event


class TestDasmSpans(unittest.TestCase):
    def test_frames_to_spans(self) -> None:
        scores = np.array([0.1, 0.8, 0.9, 0.2, 0.0, 0.7, 0.6, 0.1], dtype=np.float32)
        spans = frames_to_spans(
            scores, threshold=0.5, frames_per_second=50.0, min_span_s=0.02
        )
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0]["t0_s"], 0.02)
        self.assertAlmostEqual(spans[0]["t1_s"], 0.06)
        self.assertGreaterEqual(spans[0]["confidence"], 0.8)


class TestDasmMask(unittest.TestCase):
    def test_tgt_mask_shape(self) -> None:
        mask = build_tgt_mask(10, 7)
        self.assertEqual(mask.shape, (10, 10))
        self.assertFalse(mask[0, 0])
        self.assertFalse(bool(mask[9, 0]))  # base columns unmasked for attend
        self.assertTrue(bool(mask[0, 9]))

    def test_tgt_mask_pads_blocked(self) -> None:
        # 7 base + 3 real open + 6 pad = 16
        mask = build_tgt_mask(16, 7, real_open=3)
        self.assertFalse(bool(mask[8, 0]))  # real open may attend base
        self.assertTrue(bool(mask[10, 0]))  # pad blocked from base
        self.assertTrue(bool(mask[0, 10]))  # base blocked from pad
        self.assertFalse(bool(mask[10, 10]))  # pad self kept open

    def test_pad_open_vocab_queries(self) -> None:
        self.assertEqual(OPEN_VOCAB_SLOTS, MAX_OPEN_VOCAB_PROMPTS)
        real = np.random.randn(3, 1024).astype(np.float32)
        padded, n = pad_open_vocab_queries(real, slots=16)
        self.assertEqual(n, 3)
        self.assertEqual(padded.shape, (16, 1024))
        np.testing.assert_allclose(padded[:3], real)
        self.assertTrue(np.all(padded[3:] == 0))
        with self.assertRaises(ValueError):
            pad_open_vocab_queries(np.zeros((17, 1024), np.float32), slots=16)

    def test_save_prompts_enforces_max(self) -> None:
        pairs = [
            DasmPromptPair(f"l{i}", f"prompt {i}")
            for i in range(MAX_OPEN_VOCAB_PROMPTS + 1)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            with self.assertRaises(ValueError):
                save_prompt_pairs(pairs, path)


class TestDasmCapture(unittest.TestCase):
    def test_fixed_window_length(self) -> None:
        sr = 48000
        pre_s, post_s = 0.2, 0.3
        cap = FixedDasmCapture(sr, pre_roll_s=pre_s, post_roll_s=post_s)
        chunk = int(0.05 * sr)
        telem = ChunkTelemetry(gate_open=True, dba_spl=60.0, chunk_index=0)
        for i in range(10):
            cap.feed(np.random.randn(chunk).astype(np.float32) * 0.1, telem)
        self.assertTrue(cap.warm)
        cap.arm({"trigger_reason": "test"})
        self.assertEqual(cap.state, CaptureState.CAPTURING)
        while not cap.ready:
            cap.feed(np.random.randn(chunk).astype(np.float32) * 0.1, telem)
        wav = cap.waveform()
        expected = int(round((pre_s + post_s) * sr))
        self.assertEqual(wav.size, expected)


class TestDasmPreprocess(unittest.TestCase):
    def test_log_mel_shape(self) -> None:
        pcm = np.zeros(32000 * 10, dtype=np.float32)
        pcm[1000:2000] = 0.2
        mel = log_mel_spectrogram(pcm)
        self.assertEqual(mel.shape, (1, 1, 1001, 64))

    def test_pad_or_truncate(self) -> None:
        self.assertEqual(pad_or_truncate(np.ones(5, np.float32), 10).size, 10)
        self.assertEqual(pad_or_truncate(np.ones(20, np.float32), 10).size, 10)


class TestDasmConfig(unittest.TestCase):
    def test_window_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dasm_window.json"
            cfg = load_window_config(path)
            self.assertEqual(cfg.pre_roll_s, 5.0)
            cfg2 = cfg.normalized()
            save_window_config(cfg2, path)
            loaded = load_window_config(path)
            self.assertEqual(loaded.post_roll_s, 5.0)

    def test_prompts_hash_stable(self) -> None:
        pairs = [
            DasmPromptPair("a", "sound of a"),
            DasmPromptPair("b", "sound of b"),
        ]
        self.assertEqual(prompts_hash(pairs), prompts_hash(list(pairs)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            save_prompt_pairs(pairs, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["prompts"]), 2)

    def test_embeddings_roundtrip(self) -> None:
        pairs = [
            DasmPromptPair("siren", "sound of siren"),
            DasmPromptPair("dog", "sound of dog"),
        ]
        file_hash = prompts_hash(pairs)
        matrix = np.random.randn(2, 1024).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dasm_query_embeddings.npz"
            save_embeddings(
                labels=[p.label for p in pairs],
                embeddings=matrix,
                file_hash=file_hash,
                model_name=EXPECTED_MODEL_NAME,
                path=path,
            )
            loaded = load_embeddings(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.labels, ["siren", "dog"])
            self.assertEqual(loaded.prompts_hash, file_hash)
            ok, err = embeddings_compatible(loaded)
            self.assertTrue(ok, err)
            np.testing.assert_allclose(loaded.embeddings, matrix)


class TestDasmEventSchema(unittest.TestCase):
    def test_nested_dasm_payload(self) -> None:
        event = build_noise_event(
            device_id="test",
            chunk_index=0,
            recording_id="r1",
            rms_unweighted=0.1,
            rms_a_weighted=0.1,
            dba_spl=50.0,
            predictions=[{"label": "Speech", "confidence": 0.9}],
            dasm={
                "status": "triggered",
                "event_seconds": 10.0,
                "labels": [
                    {
                        "label": "siren",
                        "clip_confidence": 0.8,
                        "spans": [{"t0_s": 1.0, "t1_s": 2.0, "confidence": 0.7}],
                    }
                ],
            },
        )
        self.assertIn("dasm", event)
        self.assertEqual(event["dasm"]["status"], "triggered")
        self.assertNotIn("clap_status", event)


if __name__ == "__main__":
    unittest.main()
