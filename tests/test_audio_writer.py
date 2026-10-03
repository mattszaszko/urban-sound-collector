"""Tests for optional WAV/FLAC audio writers."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from core.audio_constants import CAPTURE_SAMPLE_RATE
from core.audio_writer import (
    normalize_audio_format,
    open_audio_writer,
    sibling_audio_path,
)


class AudioFormatHelpersTests(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize_audio_format("FLAC"), "flac")
        self.assertEqual(normalize_audio_format("wav"), "wav")
        self.assertEqual(normalize_audio_format("nope", default="flac"), "flac")

    def test_sibling_prefers_flac(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "clip.jsonl"
            jsonl.write_text("{}\n", encoding="utf-8")
            (root / "clip.wav").write_bytes(b"RIFF")
            (root / "clip.flac").write_bytes(b"fLaC")
            found = sibling_audio_path(jsonl)
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found.suffix, ".flac")


class OpenAudioWriterTests(unittest.TestCase):
    def test_wav_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            writer = open_audio_writer(path, audio_format="wav", sample_rate=8_000)
            writer.write_float32(np.array([0.25, -0.25], dtype=np.float32))
            writer.close()
            with wave.open(str(path), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getnframes(), 2)

    def test_flac_writer_roundtrip(self) -> None:
        try:
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("soundfile not installed")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.flac"
            writer = open_audio_writer(
                path, audio_format="flac", sample_rate=CAPTURE_SAMPLE_RATE
            )
            tone = (
                0.4
                * np.sin(2.0 * np.pi * 440.0 * np.arange(2400) / CAPTURE_SAMPLE_RATE)
            ).astype(np.float32)
            writer.write_float32(tone)
            writer.close()
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 200)
            data, rate = sf.read(str(path), dtype="float32")
            self.assertEqual(rate, CAPTURE_SAMPLE_RATE)
            self.assertEqual(data.shape[0], 2400)


if __name__ == "__main__":
    unittest.main()
