"""Optional mono audio recorders for ungained capture PCM (WAV / FLAC)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from core.audio_constants import CAPTURE_SAMPLE_RATE
from core.wav_writer import WavWriter

AUDIO_FORMATS = ("flac", "wav")
DEFAULT_AUDIO_FORMAT = "flac"


class AudioWriter(Protocol):
    path: Path

    def write_float32(self, samples: np.ndarray) -> None:
        """Append mono float32 samples in [-1, 1]."""

    def close(self) -> None:
        """Finalize the file. Safe to call more than once."""


def normalize_audio_format(raw: object, *, default: str = DEFAULT_AUDIO_FORMAT) -> str:
    text = str(raw or default).strip().lower()
    if text in AUDIO_FORMATS:
        return text
    return default


def audio_suffix(fmt: str) -> str:
    return ".flac" if normalize_audio_format(fmt) == "flac" else ".wav"


def float32_to_pcm16(samples: np.ndarray) -> np.ndarray:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return np.array([], dtype=np.int16)
    clipped = np.clip(mono, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


class FlacWriter:
    """Write float32 mono PCM [-1, 1] as 16-bit FLAC (lossless)."""

    def __init__(
        self,
        path: Path,
        *,
        sample_rate: int = CAPTURE_SAMPLE_RATE,
    ) -> None:
        try:
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "FLAC recording requires the soundfile package "
                "(pip install soundfile) and libsndfile "
                "(sudo apt install libsndfile1)."
            ) from exc

        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sf = sf.SoundFile(
            str(self.path),
            mode="w",
            samplerate=self.sample_rate,
            channels=1,
            format="FLAC",
            subtype="PCM_16",
        )
        self._closed = False

    def write_float32(self, samples: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError(f"FlacWriter already closed: {self.path}")
        pcm16 = float32_to_pcm16(samples)
        if pcm16.size == 0:
            return
        self._sf.write(pcm16)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sf.close()
        except Exception:  # noqa: BLE001 — best-effort finalize on stop
            pass


def open_audio_writer(
    path: Path,
    *,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    sample_rate: int = CAPTURE_SAMPLE_RATE,
) -> AudioWriter:
    """Open a WAV or FLAC writer for ungained mono capture audio."""
    fmt = normalize_audio_format(audio_format)
    out = Path(path)
    expected = audio_suffix(fmt)
    if out.suffix.lower() != expected:
        out = out.with_suffix(expected)
    if fmt == "flac":
        return FlacWriter(out, sample_rate=sample_rate)
    return WavWriter(out, sample_rate=sample_rate)


def sibling_audio_path(jsonl_path: Path) -> Path | None:
    """Return an existing sibling audio file (.flac preferred, then .wav)."""
    stem = Path(jsonl_path)
    for ext in (".flac", ".wav"):
        candidate = stem.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None
