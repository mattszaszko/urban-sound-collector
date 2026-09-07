"""CLAP zero-shot classifier (placeholder embeds until ONNX models are installed)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.clap_prompts import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_PROMPTS_PATH,
    ClapPromptPair,
    embedding_sync_status,
    load_embeddings,
    load_prompt_pairs,
    prompts_hash,
    save_embeddings,
)

CLAP_MODEL_NAME = "clap-placeholder"
CLAP_MODEL_VERSION = "dev-hash-v1"
EMBED_DIM = 64


@dataclass
class ClapPredictResult:
    predictions: list[dict[str, Any]]
    top_label: str
    top_confidence: float
    inference_ms: float
    model_name: str = CLAP_MODEL_NAME
    model_version: str = CLAP_MODEL_VERSION


def _hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic unit vector from text (dev stand-in for CLAP text tower)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(digest * ((dim // len(digest)) + 1), dtype=np.uint8)[:dim]
    vec = raw.astype(np.float32) - 127.5
    norm = float(np.linalg.norm(vec)) + 1e-9
    return vec / norm


def rebuild_text_embeddings(
    *,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
) -> dict[str, Any]:
    """Rebuild cached text embeddings for current prompt pairs."""
    pairs = load_prompt_pairs(prompts_path)
    if not pairs:
        raise ValueError("No CLAP prompts to embed")
    labels = [p.label for p in pairs]
    matrix = np.stack([_hash_embed(p.prompt) for p in pairs], axis=0)
    file_hash = prompts_hash(pairs)
    save_embeddings(
        labels=labels,
        embeddings=matrix,
        file_hash=file_hash,
        path=embeddings_path,
    )
    return {
        "ok": True,
        "count": len(labels),
        "prompts_hash": file_hash,
        "embeddings_path": str(embeddings_path),
        "backend": CLAP_MODEL_NAME,
    }


class ClapClassifier:
    """Rank audio against cached prompt embeddings."""

    def __init__(
        self,
        *,
        prompts_path: Path = DEFAULT_PROMPTS_PATH,
        embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
        top_k: int = 3,
    ) -> None:
        self.prompts_path = prompts_path
        self.embeddings_path = embeddings_path
        self.top_k = max(1, int(top_k))
        self._labels: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._hash: str | None = None
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
        labels, embeddings, file_hash = loaded
        if file_hash != expected:
            raise RuntimeError(
                "CLAP embeddings are out of sync with clap_prompts.json; rebuild them"
            )
        self._labels = labels
        self._embeddings = embeddings
        self._hash = file_hash

    @property
    def ready(self) -> bool:
        return self._embeddings is not None and bool(self._labels)

    def sync_status(self) -> dict[str, Any]:
        return embedding_sync_status(
            prompts_path=self.prompts_path,
            embeddings_path=self.embeddings_path,
        )

    def predict(self, pcm_48k: np.ndarray) -> ClapPredictResult:
        if self._embeddings is None or not self._labels:
            raise RuntimeError("CLAP classifier has no embeddings loaded")
        started = time.perf_counter()
        mono = np.asarray(pcm_48k, dtype=np.float32).reshape(-1)
        # Placeholder audio tower: hash of coarse RMS + spectral centroid proxy.
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        # Cheap shape fingerprint so different buffers don't always tie.
        step = max(1, mono.size // 32)
        fingerprint = mono[::step][:32]
        key = f"rms={rms:.6f}|fp={fingerprint.tobytes().hex()}"
        audio_vec = _hash_embed(key)
        sims = self._embeddings @ audio_vec
        order = np.argsort(-sims)[: self.top_k]
        predictions = [
            {
                "label": self._labels[int(i)],
                "confidence": float(max(0.0, min(1.0, (sims[int(i)] + 1.0) / 2.0))),
            }
            for i in order
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ClapPredictResult(
            predictions=predictions,
            top_label=predictions[0]["label"],
            top_confidence=float(predictions[0]["confidence"]),
            inference_ms=round(elapsed_ms, 1),
        )
