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


def read_flac_total_samples(path: Path) -> int | None:
    """Return STREAMINFO total samples, or None if missing/unreadable."""
    try:
        with Path(path).open("rb") as f:
            header = f.read(8)
            if len(header) < 8 or header[:4] != b"fLaC":
                return None
            block_type = header[4] & 0x7F
            length = (header[5] << 16) | (header[6] << 8) | header[7]
            if block_type != 0 or length < 34:
                return None
            streaminfo = f.read(length)
    except OSError:
        return None
    if len(streaminfo) < 18:
        return None
    packed = int.from_bytes(streaminfo[10:18], "big")
    return packed & ((1 << 36) - 1)


def patch_flac_total_samples(path: Path, total_samples: int) -> bool:
    """Write total sample count into FLAC STREAMINFO (fixes unseekable files)."""
    if total_samples <= 0:
        return False
    try:
        data = bytearray(Path(path).read_bytes())
    except OSError:
        return False
    if len(data) < 42 or data[:4] != b"fLaC":
        return False
    block_type = data[4] & 0x7F
    length = (data[5] << 16) | (data[6] << 8) | data[7]
    if block_type != 0 or length < 34:
        return False
    si = 8
    packed = int.from_bytes(data[si + 10 : si + 18], "big")
    packed = (packed & ~((1 << 36) - 1)) | (int(total_samples) & ((1 << 36) - 1))
    data[si + 10 : si + 18] = packed.to_bytes(8, "big")
    try:
        Path(path).write_bytes(data)
    except OSError:
        return False
    return True


def _plausible_flac_sample_count(path: Path, samples: int) -> bool:
    """Reject absurd lengths (unfinalized FLAC often reports INT64_MAX frames)."""
    if samples <= 0:
        return False
    # Hard cap (~6 h @ 48 kHz) — collector clips are minutes, not days.
    if samples > 48_000 * 60 * 60 * 6:
        return False
    try:
        size = Path(path).stat().st_size
    except OSError:
        return True
    # Compressed FLAC is smaller than PCM; reject only pathological ratios
    # (e.g. libsndfile reporting 2^63-1 frames for an unfinalized file).
    if size > 0 and samples > size * 500:
        return False
    return True

def ensure_flac_seekable(path: Path, *, fallback_samples: int | None = None) -> bool:
    """Patch STREAMINFO when total_samples is 0 (common after hard stop)."""
    p = Path(path)
    if p.suffix.lower() != ".flac" or not p.is_file():
        return False
    total = read_flac_total_samples(p)
    if total is not None and total > 0:
        return False
    samples: int | None = None
    if fallback_samples is not None and _plausible_flac_sample_count(p, int(fallback_samples)):
        samples = int(fallback_samples)
    if samples is None:
        # Last resort: count frames via soundfile (may work even when STREAMINFO is 0).
        try:
            import soundfile as sf  # type: ignore[import-untyped]

            with sf.SoundFile(str(p), mode="r") as handle:
                try:
                    handle.seek(0, sf.SEEK_END)
                    candidate = int(handle.tell())
                    handle.seek(0)
                except Exception:  # noqa: BLE001
                    candidate = -1
            if _plausible_flac_sample_count(p, candidate):
                samples = candidate
        except Exception:  # noqa: BLE001
            samples = None
    if samples is None:
        return False
    return patch_flac_total_samples(p, int(samples))

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
        self._frames_written = 0
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
        self._frames_written += int(pcm16.size)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sf.close()
        except Exception:  # noqa: BLE001 — best-effort finalize on stop
            pass
        # libsndfile should update STREAMINFO on close; if the process was
        # interrupted mid-close or a buggy build left 0, patch from our counter.
        if self._frames_written > 0:
            ensure_flac_seekable(self.path, fallback_samples=self._frames_written)


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
