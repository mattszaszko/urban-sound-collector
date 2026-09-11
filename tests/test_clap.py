"""Tests for CLAP ring buffer, hybrid capture, triggers, and ONNX wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.audio_ring_buffer import AudioRingBuffer
from core.clap_capture import HybridClapCapture
from core.clap_onnx import (
    CLAP_MODEL_NAME,
    EMBED_DIM,
    ClapModelsMissingError,
    require_clap_assets,
)
from core.clap_preprocess import extract_clap_features, pad_or_truncate
from core.clap_prompts import (
    EXPECTED_EMBED_DIM,
    EXPECTED_MODEL_NAME,
    ClapPromptPair,
    embedding_sync_status,
    load_embeddings,
    load_prompt_pairs,
    prompts_hash,
    save_prompt_pairs,
)
from core.clap_trigger import (
    CLAP_STATUS_GATED,
    CLAP_STATUS_PENDING,
    CLAP_STATUS_SCHEDULED,
    CLAP_STATUS_SKIPPED,
    ClapTriggerConfig,
    ClapTriggerState,
    evaluate_arm,
)
from core.classifier_clap import ClapClassifier, rebuild_text_embeddings


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

    def test_last_n(self) -> None:
        ring = AudioRingBuffer(maxlen_samples=10)
        ring.append(np.arange(10, dtype=np.float32))
        np.testing.assert_array_equal(ring.last_n(3), np.array([7, 8, 9], dtype=np.float32))


class HybridClapCaptureTests(unittest.TestCase):
    def test_pre_then_post_until_ready(self) -> None:
        sr = 100
        job = HybridClapCapture(
            sample_rate=sr, pre_roll_seconds=2.0, post_roll_seconds=8.0
        )
        pre = np.ones(200, dtype=np.float32)
        job.start(pre, {"trigger_chunk_index": 5, "trigger_reason": "label:Vehicle"})
        self.assertTrue(job.active)
        self.assertFalse(job.ready)
        # 8 s post @ 100 Hz = 800 samples
        for _ in range(7):
            job.append_post(np.full(100, 2.0, dtype=np.float32))
            self.assertFalse(job.ready)
        job.append_post(np.full(100, 2.0, dtype=np.float32))
        self.assertTrue(job.ready)
        wave = job.waveform()
        self.assertEqual(wave.size, 1000)
        np.testing.assert_array_equal(wave[:200], 1.0)
        np.testing.assert_array_equal(wave[200:], 2.0)
        self.assertEqual(job.meta["trigger_chunk_index"], 5)


class ClapTriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ClapTriggerConfig(
            cooldown_seconds=5.0,
            dba_threshold=55.0,
            trigger_labels=["Vehicle", "Siren"],
            ambiguous_labels=["Noise"],
        )
        self.state = ClapTriggerState()

    def test_gated(self) -> None:
        arm, status, reason = evaluate_arm(
            gated=True,
            top_label="Vehicle",
            dba_spl=70.0,
            preroll_ready=True,
            capture_active=False,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertFalse(arm)
        self.assertEqual(status, CLAP_STATUS_GATED)
        self.assertEqual(reason, "gated")

    def test_direct_arm(self) -> None:
        arm, status, reason = evaluate_arm(
            gated=False,
            top_label="Vehicle",
            dba_spl=40.0,
            preroll_ready=True,
            capture_active=False,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertTrue(arm)
        self.assertEqual(status, CLAP_STATUS_SCHEDULED)
        self.assertIn("Vehicle", reason)

    def test_ambiguous_needs_dba(self) -> None:
        arm, status, _ = evaluate_arm(
            gated=False,
            top_label="Noise",
            dba_spl=40.0,
            preroll_ready=True,
            capture_active=False,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertFalse(arm)
        self.assertEqual(status, CLAP_STATUS_SKIPPED)

        arm2, status2, reason2 = evaluate_arm(
            gated=False,
            top_label="Noise",
            dba_spl=60.0,
            preroll_ready=True,
            capture_active=False,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertTrue(arm2)
        self.assertEqual(status2, CLAP_STATUS_SCHEDULED)
        self.assertIn("ambiguous", reason2)

    def test_cooldown_skips_without_carry(self) -> None:
        self.state.last_arm_monotonic = 100.0
        arm, status, reason = evaluate_arm(
            gated=False,
            top_label="Vehicle",
            dba_spl=70.0,
            preroll_ready=True,
            capture_active=False,
            config=self.cfg,
            state=self.state,
            now_monotonic=102.0,
        )
        self.assertFalse(arm)
        self.assertEqual(status, CLAP_STATUS_SKIPPED)
        self.assertEqual(reason, "cooldown")

    def test_capture_active_is_pending(self) -> None:
        arm, status, reason = evaluate_arm(
            gated=False,
            top_label="Vehicle",
            dba_spl=70.0,
            preroll_ready=True,
            capture_active=True,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertFalse(arm)
        self.assertEqual(status, CLAP_STATUS_PENDING)
        self.assertEqual(reason, "capturing")

    def test_no_match_skips_not_carry(self) -> None:
        arm, status, reason = evaluate_arm(
            gated=False,
            top_label="Music",
            dba_spl=70.0,
            preroll_ready=True,
            capture_active=False,
            config=self.cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertFalse(arm)
        self.assertEqual(status, CLAP_STATUS_SKIPPED)
        self.assertEqual(reason, "no_match")


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

    def test_legacy_placeholder_embeddings_not_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embeds.npz"
            np.savez_compressed(
                path,
                labels=np.asarray(["siren"]),
                embeddings=np.zeros((1, 64), dtype=np.float32),
                prompts_hash=np.asarray("deadbeef"),
            )
            loaded = load_embeddings(path)
            self.assertIsNotNone(loaded)
            status = embedding_sync_status(
                [ClapPromptPair(label="siren", prompt="siren")],
                embeddings_path=path,
            )
            self.assertFalse(status["compatible"])
            self.assertFalse(status["in_sync"])


class ClapPreprocessTests(unittest.TestCase):
    def test_pad_repeat_and_truncate(self) -> None:
        short, longer = pad_or_truncate(np.ones(1000, dtype=np.float32), max_samples=4800)
        self.assertEqual(short.size, 4800)
        self.assertFalse(longer)
        long_wave = np.arange(10000, dtype=np.float32)
        cropped, was_long = pad_or_truncate(long_wave, max_samples=4800)
        self.assertEqual(cropped.size, 4800)
        self.assertTrue(was_long)
        np.testing.assert_array_equal(cropped, long_wave[-4800:])

    def test_extract_features_shape(self) -> None:
        pcm = np.random.randn(480_000).astype(np.float32) * 0.01
        feats, is_longer = extract_clap_features(pcm)
        self.assertEqual(feats.shape, (1, 1, 1001, 64))
        self.assertEqual(is_longer.shape, (1, 1))
        self.assertFalse(bool(is_longer[0, 0]))

    def test_short_audio_repeatpad_shape(self) -> None:
        pcm = np.random.randn(48_000).astype(np.float32) * 0.01
        feats, is_longer = extract_clap_features(pcm)
        self.assertEqual(feats.shape, (1, 1, 1001, 64))
        self.assertFalse(bool(is_longer[0, 0]))


class _FakeTextEncoder:
    model_version = "testhash12"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, _ in enumerate(texts):
            out[i, i % EMBED_DIM] = 1.0
        return out

    def close(self) -> None:
        return None


class _FakeAudioEncoder:
    model_version = "testhash12"

    def __init__(self, match_index: int = 0) -> None:
        self.match_index = match_index

    def embed(self, pcm_48k: np.ndarray) -> np.ndarray:
        out = np.zeros(EMBED_DIM, dtype=np.float32)
        out[self.match_index % EMBED_DIM] = 1.0
        return out


class ClapClassifierMockTests(unittest.TestCase):
    def test_rebuild_and_predict_with_fakes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "prompts.json"
            embeds = Path(tmp) / "embeds.npz"
            save_prompt_pairs(
                [
                    ClapPromptPair(label="siren", prompt="An outdoor siren"),
                    ClapPromptPair(label="truck", prompt="A diesel truck"),
                ],
                prompts,
            )
            result = rebuild_text_embeddings(
                prompts_path=prompts,
                embeddings_path=embeds,
                text_encoder=_FakeTextEncoder(),
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["backend"], CLAP_MODEL_NAME)
            self.assertEqual(result["embed_dim"], EXPECTED_EMBED_DIM)
            loaded = load_embeddings(embeds)
            assert loaded is not None
            self.assertEqual(loaded.model_name, EXPECTED_MODEL_NAME)
            self.assertEqual(loaded.embed_dim, EXPECTED_EMBED_DIM)

            clf = ClapClassifier(
                prompts_path=prompts,
                embeddings_path=embeds,
                audio_encoder=_FakeAudioEncoder(match_index=1),
                top_k=2,
            )
            pred = clf.predict(np.zeros(480_000, dtype=np.float32))
            self.assertEqual(pred.top_label, "truck")
            self.assertEqual(pred.model_name, CLAP_MODEL_NAME)
            self.assertEqual(len(pred.predictions), 2)

    def test_peak_normalize_is_scale_invariant_with_fake_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "prompts.json"
            embeds = Path(tmp) / "embeds.npz"
            save_prompt_pairs(
                [ClapPromptPair(label="a", prompt="Alpha"), ClapPromptPair(label="b", prompt="Beta")],
                prompts,
            )
            rebuild_text_embeddings(
                prompts_path=prompts,
                embeddings_path=embeds,
                text_encoder=_FakeTextEncoder(),
            )

            class _CaptureAudio:
                model_version = "t"
                last: np.ndarray | None = None

                def embed(self, pcm_48k: np.ndarray) -> np.ndarray:
                    self.last = np.asarray(pcm_48k, dtype=np.float32).copy()
                    out = np.zeros(EMBED_DIM, dtype=np.float32)
                    out[0] = 1.0
                    return out

            audio = _CaptureAudio()
            clf = ClapClassifier(
                prompts_path=prompts,
                embeddings_path=embeds,
                audio_encoder=audio,  # type: ignore[arg-type]
            )
            wave = np.random.randn(4800).astype(np.float32) * 0.01
            clf.predict(wave)
            assert audio.last is not None
            self.assertAlmostEqual(float(np.max(np.abs(audio.last))), 1.0, places=5)
            clf.predict(wave * 10.0)
            assert audio.last is not None
            self.assertAlmostEqual(float(np.max(np.abs(audio.last))), 1.0, places=5)

    def test_missing_models_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "clap"
            empty.mkdir()
            with self.assertRaises(ClapModelsMissingError) as ctx:
                require_clap_assets(empty, need_audio=True)
            self.assertIn("download_clap_models", str(ctx.exception))


@unittest.skipUnless(
    (Path(__file__).resolve().parents[1] / "models" / "clap" / "audio_model_quantized.onnx").exists()
    and (Path(__file__).resolve().parents[1] / "models" / "clap" / "text_model_quantized.onnx").exists(),
    "CLAP ONNX models not downloaded",
)
class ClapOnnxIntegrationTests(unittest.TestCase):
    """Optional live ONNX smoke test when models/clap/*.onnx are present."""

    def test_text_and_audio_embed_dims(self) -> None:
        from core.clap_onnx import AudioEncoder, TextEncoder

        text = TextEncoder()
        try:
            vecs = text.embed(["A dog barking outdoors", "An ambulance siren"])
            self.assertEqual(vecs.shape, (2, EMBED_DIM))
            norms = np.linalg.norm(vecs, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-4)
        finally:
            text.close()

        audio = AudioEncoder()
        try:
            pcm = np.random.randn(480_000).astype(np.float32) * 0.01
            avec = audio.embed(pcm)
            self.assertEqual(avec.shape, (EMBED_DIM,))
            self.assertAlmostEqual(float(np.linalg.norm(avec)), 1.0, places=4)
        finally:
            audio.close()


if __name__ == "__main__":
    unittest.main()
