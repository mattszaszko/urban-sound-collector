"""ALSA capture for INMP441 S32_LE mono audio at 48 kHz."""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from core.audio_constants import CAPTURE_SAMPLE_RATE

DEFAULT_READ_FRAMES = 4_800  # 0.1 s reads for responsive buffering


class _CaptureBackend(ABC):
    @abstractmethod
    def read_frames(self) -> np.ndarray:
        """Return the next block of int32 mono frames."""

    @abstractmethod
    def close(self) -> None:
        """Release capture resources."""


class _PyAlsaBackend(_CaptureBackend):
    def __init__(self, device: str, read_frames: int) -> None:
        import alsaaudio  # type: ignore[import-untyped]

        self._pcm = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device=device,
        )
        self._pcm.setchannels(1)
        self._pcm.setrate(CAPTURE_SAMPLE_RATE)
        self._pcm.setformat(alsaaudio.PCM_FORMAT_S32_LE)
        self._pcm.setperiodsize(read_frames)

    def read_frames(self) -> np.ndarray:
        length, data = self._pcm.read()
        if length <= 0:
            return np.array([], dtype=np.int32)
        return np.frombuffer(data[: length * 4], dtype="<i4", count=length)

    def close(self) -> None:
        return


class _ArecordBackend(_CaptureBackend):
    def __init__(self, device: str, read_frames: int) -> None:
        if not shutil.which("arecord"):
            raise RuntimeError("arecord not found. Install: sudo apt install alsa-utils")

        self._read_bytes = read_frames * 4
        # DEVNULL for stderr avoids the classic deadlock where arecord fills
        # the stderr pipe and then blocks, so stdout.read() hangs forever.
        self._proc = subprocess.Popen(
            [
                "arecord",
                "-D",
                device,
                "-f",
                "S32_LE",
                "-r",
                str(CAPTURE_SAMPLE_RATE),
                "-c",
                "1",
                "-t",
                "raw",
                "-q",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if self._proc.stdout is None:
            raise RuntimeError("Failed to open arecord stdout pipe.")

    def read_frames(self) -> np.ndarray:
        assert self._proc.stdout is not None
        data = self._proc.stdout.read(self._read_bytes)
        if not data:
            rc = self._proc.poll()
            if rc is not None:
                raise RuntimeError(f"arecord exited early (code {rc})")
            return np.array([], dtype=np.int32)
        usable = len(data) - (len(data) % 4)
        if usable == 0:
            return np.array([], dtype=np.int32)
        return np.frombuffer(data[:usable], dtype="<i4")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import alsaaudio  # noqa: F401

        return "pyalsa"
    except ImportError:
        return "arecord"


def open_capture_backend(
    device: str,
    *,
    backend: str = "auto",
    read_frames: int = DEFAULT_READ_FRAMES,
) -> _CaptureBackend:
    resolved = resolve_backend(backend)
    if resolved == "pyalsa":
        return _PyAlsaBackend(device, read_frames)
    return _ArecordBackend(device, read_frames)


class AlsAudioCapture:
    """Buffered ALSA capture that yields fixed-size int32 chunk frames."""

    def __init__(
        self,
        device: str,
        chunk_samples: int,
        *,
        backend: str = "auto",
        read_frames: int = DEFAULT_READ_FRAMES,
    ) -> None:
        self.device = device
        self.chunk_samples = chunk_samples
        self.backend_name = resolve_backend(backend)
        self._backend = open_capture_backend(
            device,
            backend=backend,
            read_frames=read_frames,
        )
        self._buffer = np.array([], dtype=np.int32)

    def read_chunk(self) -> np.ndarray:
        """Block until ``chunk_samples`` int32 frames are available."""
        empty_reads = 0
        while self._buffer.size < self.chunk_samples:
            block = self._backend.read_frames()
            if block.size == 0:
                empty_reads += 1
                if empty_reads >= 50:
                    raise RuntimeError(
                        "ALSA capture produced no audio frames. "
                        "Check the mic device is free and working "
                        f"(device={self.device}, backend={self.backend_name})."
                    )
                continue
            empty_reads = 0
            self._buffer = np.concatenate((self._buffer, block))

        chunk = self._buffer[: self.chunk_samples]
        self._buffer = self._buffer[self.chunk_samples :]
        return chunk.copy()

    def iter_chunks(self) -> Iterator[np.ndarray]:
        while True:
            yield self.read_chunk()

    def close(self) -> None:
        self._backend.close()
