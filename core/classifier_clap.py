"""CLAP zero-shot classifier using Xenova clap-htsat-unfused ONNX."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.clap_onnx import (
    CLAP_MODEL_NAME,
    AudioEncoder,
    ClapModelsMissingError,
    TextEncoder,
    model_version_fingerprint,
    onnx_ready_status,
)
from core.clap_prompts import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_PROMPTS_PATH,
    EXPECTED_EMBED_DIM,
    EXPECTED_MODEL_NAME,
    embedding_sync_status,
    embeddings_compatible,
    load_embeddings,
    load_prompt_pairs,
    prompts_hash,
    save_embeddings,
)

# Re-export for callers / tests
__all__ = [
    "CLAP_MODEL_NAME",
    "ClapClassifier",
    "ClapModelsMissingError",
    "ClapPredictResult",
    "rebuild_text_embeddings",
]


@dataclass
class ClapPredictResult:
    predictions: list[dict[str, Any]]
    top_label: str
    top_confidence: float
    inference_ms: float
    model_name: str = CLAP_MODEL_NAME
    model_version: str = ""


def rebuild_text_embeddings(
    *,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    text_encoder: TextEncoder | None = None,
    clap_dir: Path | None = None,
) -> dict[str, Any]:
    """Rebuild cached text embeddings via the CLAP text ONNX tower."""
    pairs = load_prompt_pairs(prompts_path)
    if not pairs:
        raise ValueError("No CLAP prompts to embed")

    owns_encoder = text_encoder is None
    encoder = text_encoder
    try:
        if encoder is None:
            kwargs = {}
            if clap_dir is not None:
                kwargs["clap_dir"] = clap_dir
            encoder = TextEncoder(**kwargs)
        matrix = encoder.embed([p.prompt for p in pairs])
        if matrix.shape[1] != EXPECTED_EMBED_DIM:
            raise RuntimeError(
                f"Text embeds have dim {matrix.shape[1]}, expected {EXPECTED_EMBED_DIM}"
            )
        labels = [p.label for p in pairs]
        file_hash = prompts_hash(pairs)
        version = getattr(encoder, "model_version", "") or model_version_fingerprint()
        save_embeddings(
            labels=labels,
            embeddings=matrix,
            file_hash=file_hash,
            model_name=EXPECTED_MODEL_NAME,
            embed_dim=EXPECTED_EMBED_DIM,
            path=embeddings_path,
        )
        return {
            "ok": True,
            "count": len(labels),
            "prompts_hash": file_hash,
            "embeddings_path": str(embeddings_path),
            "backend": CLAP_MODEL_NAME,
            "model_name": EXPECTED_MODEL_NAME,
            "model_version": version,
            "embed_dim": EXPECTED_EMBED_DIM,
        }
    finally:
        if owns_encoder and encoder is not None:
            encoder.close()


class ClapClassifier:
    """Rank audio against cached prompt embeddings using the audio ONNX tower."""

    def __init__(
        self,
        *,
        prompts_path: Path = DEFAULT_PROMPTS_PATH,
        embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
        top_k: int = 3,
        audio_encoder: AudioEncoder | None = None,
        clap_dir: Path | None = None,
    ) -> None:
        self.prompts_path = prompts_path
        self.embeddings_path = embeddings_path
        self.top_k = max(1, int(top_k))
        self._labels: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._hash: str | None = None
        self._model_version = ""
        if audio_encoder is not None:
            self._audio = audio_encoder
        else:
            kwargs = {}
            if clap_dir is not None:
                kwargs["clap_dir"] = clap_dir
            self._audio = AudioEncoder(**kwargs)
        self._model_version = getattr(self._audio, "model_version", "") or ""
        self.reload()

    def reload(self) -> None:
        pairs = load_prompt_pairs(self.prompts_path)
        expected = prompts_hash(pairs)
        loaded = load_embeddings(self.embeddings_path)
        if loaded is None:
            raise FileNotFoundError(
                f"CLAP embeddings missing at {self.embeddings_path}. "
                "Rebuild from the Prompts tab or scripts/rebuild_clap_embeddings.py"
            )
        ok, err = embeddings_compatible(loaded)
        if not ok:
            raise RuntimeError(err or "CLAP embeddings incompatible; rebuild required")
        if loaded.prompts_hash != expected:
            raise RuntimeError(
                "CLAP embeddings are out of sync with clap_prompts.json; rebuild them"
            )
        self._labels = loaded.labels
        self._embeddings = loaded.embeddings
        self._hash = loaded.prompts_hash

    @property
    def ready(self) -> bool:
        return self._embeddings is not None and bool(self._labels)

    def sync_status(self) -> dict[str, Any]:
        status = embedding_sync_status(
            prompts_path=self.prompts_path,
            embeddings_path=self.embeddings_path,
        )
        status["onnx"] = onnx_ready_status()
        return status

    def predict(self, pcm_48k: np.ndarray) -> ClapPredictResult:
        if self._embeddings is None or not self._labels:
            raise RuntimeError("CLAP classifier has no embeddings loaded")
        started = time.perf_counter()
        mono = np.asarray(pcm_48k, dtype=np.float32).reshape(-1)
        # Peak-normalize the hybrid window (no Branch B AGC/HPF).
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 1e-8:
            mono = mono / peak
        audio_vec = self._audio.embed(mono)
        sims = self._embeddings @ audio_vec
        order = np.argsort(-sims)[: self.top_k]
        predictions = [
            {
                "label": self._labels[int(i)],
                "confidence": float(max(0.0, min(1.0, (float(sims[int(i)]) + 1.0) / 2.0))),
            }
            for i in order
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ClapPredictResult(
            predictions=predictions,
            top_label=predictions[0]["label"],
            top_confidence=float(predictions[0]["confidence"]),
            inference_ms=round(elapsed_ms, 1),
            model_name=CLAP_MODEL_NAME,
            model_version=self._model_version,
        )
