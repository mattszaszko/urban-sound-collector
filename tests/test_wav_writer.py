"""Tests for optional mono WAV recording."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from core.audio_constants import CAPTURE_SAMPLE_RATE
from core.wav_writer import WavWriter


class WavWriterTests(unittest.TestCase):
    def test_writes_valid_mono_16bit_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            writer = WavWriter(path, sample_rate=CAPTURE_SAMPLE_RATE)
            tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(4800) / 48_000.0)).astype(
                np.float32
            )
            writer.write_float32(tone)
            writer.close()

            with wave.open(str(path), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), CAPTURE_SAMPLE_RATE)
                self.assertEqual(wf.getnframes(), 4800)
                raw = wf.readframes(wf.getnframes())
            self.assertEqual(len(raw), 4800 * 2)
            # RIFF header should report a non-trivial data size after close.
            self.assertGreater(path.stat().st_size, 44)

    def test_full_scale_maps_near_int16_peak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "peak.wav"
            writer = WavWriter(path, sample_rate=8_000)
            writer.write_float32(np.array([1.0, -1.0, 0.0], dtype=np.float32))
            writer.close()

            with wave.open(str(path), "rb") as wf:
                frames = np.frombuffer(wf.readframes(3), dtype=np.int16)
            self.assertEqual(int(frames[0]), 32767)
            self.assertEqual(int(frames[1]), -32767)
            self.assertEqual(int(frames[2]), 0)

    def test_close_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "once.wav"
            writer = WavWriter(path, sample_rate=8_000)
            writer.write_float32(np.zeros(10, dtype=np.float32))
            writer.close()
            writer.close()
            with self.assertRaises(RuntimeError):
                writer.write_float32(np.zeros(1, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
