"""CLAP log-mel feature extraction (HF ClapFeatureExtractor-compatible)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal.windows import hann

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSOR_PATH = REPO_ROOT / "models" / "clap" / "preprocessor_config.json"

# Defaults match Xenova/clap-htsat-unfused preprocessor_config.json
DEFAULTS: dict[str, Any] = {
    "sampling_rate": 48000,
    "max_length_s": 10,
    "n_fft": 1024,
    "fft_window_size": 1024,
    "hop_length": 480,
    "feature_size": 64,
    "frequency_min": 50.0,
    "frequency_max": 14000.0,
    "nb_max_samples": 480000,
    "padding": "repeatpad",
    "truncation": "fusion",  # deterministic: keep last max_length (rand_trunc not used live)
}


def load_preprocessor_config(path: Path = DEFAULT_PREPROCESSOR_PATH) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg.update(raw)
    return cfg


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _htk_mel_filterbank(
    *,
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Return (n_mels, n_fft//2+1) HTK mel filterbank weights."""
    n_freqs = n_fft // 2 + 1
    mel_min = float(_hz_to_mel(fmin))
    mel_max = float(_hz_to_mel(fmax))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, n_freqs)
        for j in range(left, center):
            if center != left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                fb[i, j] = (right - j) / (right - center)
    # Slaney-style normalize so each filter sums to 1 (HF Clap uses this path).
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb.astype(np.float32)


_MEL_FB_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def _mel_filterbank(cfg: dict[str, Any]) -> np.ndarray:
    key = (
        int(cfg["sampling_rate"]),
        int(cfg["n_fft"]),
        int(cfg["feature_size"]),
        float(cfg["frequency_min"]),
        float(cfg["frequency_max"]),
    )
    if key not in _MEL_FB_CACHE:
        _MEL_FB_CACHE[key] = _htk_mel_filterbank(
            sr=key[0],
            n_fft=key[1],
            n_mels=key[2],
            fmin=key[3],
            fmax=key[4],
        )
    return _MEL_FB_CACHE[key]


def pad_or_truncate(
    waveform: np.ndarray,
    *,
    max_samples: int,
    padding: str = "repeatpad",
) -> tuple[np.ndarray, bool]:
    """Fit mono float waveform to max_samples. Returns (wave, is_longer)."""
    mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
    is_longer = bool(mono.size > max_samples)
    if mono.size == max_samples:
        return mono, False
    if mono.size > max_samples:
        # Deterministic crop: keep the most recent max_samples (ring-buffer end).
        return mono[-max_samples:].copy(), True
    if mono.size == 0:
        return np.zeros(max_samples, dtype=np.float32), False
    if padding == "repeatpad":
        reps = int(np.ceil(max_samples / mono.size))
        tiled = np.tile(mono, reps)[:max_samples]
        return tiled.astype(np.float32, copy=False), False
    out = np.zeros(max_samples, dtype=np.float32)
    out[: mono.size] = mono
    return out, False


def waveform_to_mel(
    waveform: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Convert mono PCM to log-mel features shaped (time, n_mels).

    Uses centered STFT (reflect pad) so 480_000 samples @ hop 480 → 1001 frames.
    """
    cfg = cfg or load_preprocessor_config()
    n_fft = int(cfg["n_fft"])
    win_length = int(cfg.get("fft_window_size", n_fft))
    hop = int(cfg["hop_length"])
    window = hann(win_length, sym=False).astype(np.float32)

    # Centered frames: pad n_fft//2 on each side (librosa/HF style).
    pad = n_fft // 2
    padded = np.pad(waveform.astype(np.float32), (pad, pad), mode="reflect")
    n_frames = 1 + (waveform.size // hop)
    # Collect frames
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, win_length),
        strides=(padded.strides[0] * hop, padded.strides[0]),
        writeable=False,
    ).copy()
    frames *= window[np.newaxis, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real.astype(np.float32) ** 2) + (spec.imag.astype(np.float32) ** 2)
    mel_fb = _mel_filterbank(cfg)
    mel = power @ mel_fb.T
    # Log mel (natural log, floor for stability — matches common CLAP exports).
    return np.log(np.clip(mel, 1e-10, None)).astype(np.float32)


def extract_clap_features(
    pcm_48k: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build ONNX audio inputs.

    Returns:
      input_features: float32 [1, 1, time, n_mels] (typically 1001 x 64)
      is_longer: bool [1, 1]
    """
    cfg = cfg or load_preprocessor_config()
    max_samples = int(cfg.get("nb_max_samples") or int(cfg["sampling_rate"]) * int(cfg["max_length_s"]))
    wave, is_longer = pad_or_truncate(
        pcm_48k,
        max_samples=max_samples,
        padding=str(cfg.get("padding", "repeatpad")),
    )
    mel = waveform_to_mel(wave, cfg=cfg)  # (time, n_mels)
    # Expected layout for HTS-AT CLAP audio ONNX: [batch, 1, time, mel]
    features = mel[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)
    longer = np.asarray([[is_longer]], dtype=np.bool_)
    return features, longer
