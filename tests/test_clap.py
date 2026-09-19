"""Tests for CLAP ring buffer, dynamic capture, triggers, and ONNX wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.audio_ring_buffer import AudioRingBuffer
from core.clap_capture import DynamicClapCapture
from core.clap_event_buffer import ChunkTelemetry
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
    load_trigger_config,
    save_trigger_config,
)
from core.classifier_clap import ClapClassifier, rebuild_text_embeddings


def _telem(
    *,
    gate_open: bool,
    dba: float = 50.0,
    rms: float | None = -40.0,
    ambient: float | None = -60.0,
    chunk_index: int | None = None,
) -> ChunkTelemetry:
    return ChunkTelemetry(
        gate_open=gate_open,
        dba_spl=dba,
        raw_rms_dbfs=rms,
        ambient_noise_floor_dbfs=ambient,
        chunk_index=chunk_index,
    )


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


class DynamicClapCaptureTests(unittest.TestCase):
    def test_retroactive_onset_from_gate_open(self) -> None:
        sr = 100
        job = DynamicClapCapture(
            sample_rate=sr,
            lookback_seconds=5.0,
            pre_onset_pad_ms=0.0,
            end_settle_chunks=2,
            max_event_seconds=7.0,
        )
        # closed, closed, open(3), open(4), trigger on open(5)
        job.feed(np.full(100, 0.0, dtype=np.float32), _telem(gate_open=False, chunk_index=0))
        job.feed(np.full(100, 0.0, dtype=np.float32), _telem(gate_open=False, chunk_index=1))
        job.feed(np.full(100, 1.0, dtype=np.float32), _telem(gate_open=True, chunk_index=2))
        job.feed(np.full(100, 1.0, dtype=np.float32), _telem(gate_open=True, chunk_index=3))
        job.feed(np.full(100, 1.0, dtype=np.float32), _telem(gate_open=True, chunk_index=4))
        job.arm({"trigger_chunk_index": 4, "trigger_reason": "label:Vehicle"})
        self.assertEqual(job.meta["t_start_reason"], "gate_open")
        self.assertFalse(job.ready)
        self.assertTrue(job.active)
        # Seed should start at first open chunk (index 2) → 300 samples so far
        self.assertEqual(job.captured_samples, 300)
        wave = job.waveform()
        np.testing.assert_array_equal(wave, np.ones(300, dtype=np.float32))

    def test_settle_two_closed_chunks(self) -> None:
        sr = 100
        job = DynamicClapCapture(
            sample_rate=sr,
            lookback_seconds=4.0,
            pre_onset_pad_ms=0.0,
            end_settle_chunks=2,
            max_event_seconds=7.0,
        )
        job.feed(np.full(100, 1.0, dtype=np.float32), _telem(gate_open=True, chunk_index=0))
        job.arm({"trigger_reason": "label:Vehicle"})
        self.assertFalse(job.ready)
        job.feed(np.full(100, 0.0, dtype=np.float32), _telem(gate_open=False, chunk_index=1))
        self.assertFalse(job.ready)
        job.feed(np.full(100, 0.0, dtype=np.float32), _telem(gate_open=False, chunk_index=2))
        self.assertTrue(job.ready)
        self.assertFalse(job.meta.get("capped"))
        self.assertEqual(job.meta.get("t_end_reason"), "gate_close")
        self.assertEqual(job.captured_samples, 300)

    def test_wind_swell_peak_decay_ends_early(self) -> None:
        """Sustained gate-open wind: end on peak decay, not the max cap."""
        sr = 100  # 100 samples == 1 s
        job = DynamicClapCapture(
            sample_rate=sr,
            lookback_seconds=4.0,
            pre_onset_pad_ms=0.0,
            end_settle_chunks=2,
            max_event_seconds=5.0,
            peak_decay_db=5.0,
        )
        # Closed, then arm on a peak chunk (gate stays open afterward).
        job.feed(np.zeros(100, dtype=np.float32), _telem(gate_open=False, dba=50.0))
        job.feed(np.ones(100, dtype=np.float32), _telem(gate_open=True, dba=65.0))
        job.arm({"trigger_reason": "label:Vehicle"})
        self.assertFalse(job.ready)
        self.assertEqual(job.meta.get("dba_peak"), 65.0)
        # Wind continues (gate open) but SPL is ≥5 dB below the peak for 2 chunks.
        job.feed(np.ones(100, dtype=np.float32), _telem(gate_open=True, dba=59.0))
        self.assertFalse(job.ready)
        job.feed(np.ones(100, dtype=np.float32), _telem(gate_open=True, dba=58.0))
        self.assertTrue(job.ready)
        self.assertEqual(job.meta.get("t_end_reason"), "peak_decay")
        self.assertFalse(job.meta.get("capped"))
        secs = float(job.meta["event_seconds"])
        self.assertGreaterEqual(secs, 1.5)
        self.assertLessEqual(secs, 3.0)
        # Without peak-decay this would keep capturing toward the 5 s cap.
        self.assertLess(job.captured_samples, int(5.0 * sr))

    def test_max_event_cap(self) -> None:
        sr = 100
        job = DynamicClapCapture(
            sample_rate=sr,
            lookback_seconds=2.0,
            pre_onset_pad_ms=0.0,
            end_settle_chunks=99,
            max_event_seconds=0.5,  # 50 samples
        )
        job.feed(np.full(100, 1.0, dtype=np.float32), _telem(gate_open=True))
        job.arm({"trigger_reason": "label:Vehicle"})
        # Seed alone is 100 samples > 50 → ready + capped immediately
        self.assertTrue(job.ready)
        self.assertTrue(job.meta.get("capped"))
        self.assertEqual(job.waveform().size, 50)

    def test_short_transient_repeatpad(self) -> None:
        sr = 1000
        job = DynamicClapCapture(
            sample_rate=sr,
            lookback_seconds=2.0,
            pre_onset_pad_ms=0.0,
            end_settle_chunks=2,
            max_event_seconds=7.0,
        )
        # One short open chunk then settle
        job.feed(np.linspace(0.1, 0.9, 200, dtype=np.float32), _telem(gate_open=True))
        job.arm({"trigger_reason": "label:Vehicle"})
        job.feed(np.zeros(200, dtype=np.float32), _telem(gate_open=False))
        job.feed(np.zeros(200, dtype=np.float32), _telem(gate_open=False))
        self.assertTrue(job.ready)
        wave = job.waveform()
        self.assertLess(wave.size, 1000)
        padded, is_longer = pad_or_truncate(wave, max_samples=480_000, padding="repeatpad")
        self.assertEqual(padded.size, 480_000)
        self.assertFalse(is_longer)
        # Tiled: first 200 samples match the event onset region of the seed
        np.testing.assert_allclose(padded[:200], wave[:200], atol=1e-6)

    def test_pre_onset_pad_applied(self) -> None:
        sr = 100
        job = DynamicClapCapture(
            sample_rate=sr,
            lookback_seconds=5.0,
            pre_onset_pad_ms=200.0,  # 20 samples
            end_settle_chunks=2,
            max_event_seconds=7.0,
        )
        job.feed(np.full(100, 0.0, dtype=np.float32), _telem(gate_open=False))
        job.feed(np.full(100, 1.0, dtype=np.float32), _telem(gate_open=True))
        job.arm({"trigger_reason": "label:Vehicle"})
        self.assertEqual(job.meta["t_start_reason"], "gate_open")
        # Onset at sample 100, pad 20 → start at 80 → 20 zeros + 100 ones
        wave = job.waveform()
        self.assertEqual(wave.size, 120)
        np.testing.assert_array_equal(wave[:20], 0.0)
        np.testing.assert_array_equal(wave[20:], 1.0)


class ClapTriggerConfigTests(unittest.TestCase):
    def test_load_ignores_legacy_pre_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triggers.json"
            path.write_text(
                '{"cooldown_seconds": 5, "dba_threshold": 55, '
                '"pre_roll_seconds": 7, "post_roll_seconds": 3, '
                '"trigger_labels": ["Vehicle"], "ambiguous_labels": []}\n',
                encoding="utf-8",
            )
            cfg = load_trigger_config(path)
            self.assertEqual(cfg.lookback_seconds, 4.0)
            self.assertEqual(cfg.max_event_seconds, 5.0)
            self.assertEqual(cfg.peak_decay_db, 5.0)
            self.assertEqual(cfg.trigger_labels, ["Vehicle"])
            self.assertIn("Wind", cfg.suppress_labels)
            save_trigger_config(cfg, path)
            raw = path.read_text(encoding="utf-8")
            self.assertIn("lookback_seconds", raw)
            self.assertIn("peak_decay_db", raw)
            self.assertIn("suppress_labels", raw)
            self.assertNotIn("pre_roll_seconds", raw)


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

    def test_suppress_blocks_even_if_listed_as_trigger(self) -> None:
        cfg = ClapTriggerConfig(
            cooldown_seconds=5.0,
            dba_threshold=55.0,
            trigger_labels=["Wind", "Vehicle"],
            ambiguous_labels=["White noise"],
            suppress_labels=["Wind", "White noise"],
        )
        cfg = cfg.normalized()
        self.assertNotIn("Wind", cfg.trigger_labels)
        self.assertNotIn("White noise", cfg.ambiguous_labels)
        arm, status, reason = evaluate_arm(
            gated=False,
            top_label="Wind",
            dba_spl=75.0,
            preroll_ready=True,
            capture_active=False,
            config=cfg,
            state=self.state,
            now_monotonic=100.0,
        )
        self.assertFalse(arm)
        self.assertEqual(status, CLAP_STATUS_SKIPPED)
        self.assertIn("suppress", reason)


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

    def test_hpf_before_peak_norm_removes_lf_peak_dominance(self) -> None:
        """A loud 20 Hz component should not set peak after HPF + peak-norm."""
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "prompts.json"
            embeds = Path(tmp) / "embeds.npz"
            save_prompt_pairs(
                [ClapPromptPair(label="a", prompt="Alpha")],
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
            sr = 48_000
            t = np.arange(sr, dtype=np.float32) / sr
            # Loud rumble + quieter 1 kHz tone — without HPF, rumble sets peak.
            wave = (
                0.9 * np.sin(2 * np.pi * 20.0 * t)
                + 0.1 * np.sin(2 * np.pi * 1000.0 * t)
            ).astype(np.float32)
            clf.predict(wave)
            assert audio.last is not None
            self.assertAlmostEqual(float(np.max(np.abs(audio.last))), 1.0, places=4)
            # After HPF the surviving energy is midband; residual LF should be small.
            skip = int(0.05 * sr)
            # Rough LF energy via moving average of abs (20 Hz period ~2400 samples).
            lf_proxy = float(np.mean(np.abs(audio.last[skip:])))
            self.assertGreater(lf_proxy, 0.05)  # mid tone still present after peak-norm

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
