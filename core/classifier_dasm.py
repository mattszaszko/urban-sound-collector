"""DASM multilabel classifier: fixed window → ORT → thresholded spans."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.audio_constants import CAPTURE_SAMPLE_RATE
from core.dasm_onnx import (
    DASM_MODEL_NAME,
    DasmModelsMissingError,
    DasmSession,
    build_tgt_mask,
    load_base_queries,
    model_version_fingerprint,
    onnx_ready_status,
)
from core.dasm_preprocess import load_bn0, prepare_dasm_mel
from core.dasm_prompts import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_WINDOW_PATH,
    DasmWindowConfig,
    embedding_sync_status,
    embeddings_compatible,
    load_embeddings,
    load_prompt_pairs,
    load_window_config,
    prompts_hash,
)
from core.yamnet_preprocess import DEFAULT_HPF_HZ, DEFAULT_HPF_ORDER

__all__ = [
    "DASM_MODEL_NAME",
    "DasmClassifier",
    "DasmModelsMissingError",
    "DasmPredictResult",
    "frames_to_spans",
]


@dataclass
class DasmPredictResult:
    labels: list[dict[str, Any]]
    top_label: str | None
    top_confidence: float
    inference_ms: float
    mel_meta: dict[str, Any]
    model_name: str = DASM_MODEL_NAME
    model_version: str = ""
    n_queries: int = 0
    base_len: int = 0


def frames_to_spans(
    frame_scores: np.ndarray,
    *,
    threshold: float,
    frames_per_second: float,
    min_span_s: float,
) -> list[dict[str, float]]:
    """Convert 1-D frame probabilities into contiguous above-threshold spans."""
    scores = np.asarray(frame_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return []
    above = scores >= float(threshold)
    spans: list[dict[str, float]] = []
    i = 0
    n = int(above.size)
    fps = max(1e-6, float(frames_per_second))
    min_frames = max(1, int(round(float(min_span_s) * fps))) if min_span_s > 0 else 1
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i + 1
        while j < n and above[j]:
            j += 1
        if (j - i) >= min_frames:
            seg = scores[i:j]
            spans.append(
                {
                    "t0_s": round(i / fps, 3),
                    "t1_s": round(j / fps, 3),
                    "confidence": round(float(np.max(seg)), 4),
                }
            )
        i = j
    return spans


class DasmClassifier:
    """Run DASM ORT on a ±5 s PCM window; return multilabel spans for open-vocab rows."""

    def __init__(
        self,
        *,
        prompts_path: Path = DEFAULT_PROMPTS_PATH,
        embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
        window_path: Path = DEFAULT_WINDOW_PATH,
        dasm_dir: Path | None = None,
        session: DasmSession | None = None,
        sample_rate: float = float(CAPTURE_SAMPLE_RATE),
        hpf_hz: float = DEFAULT_HPF_HZ,
        hpf_order: int = DEFAULT_HPF_ORDER,
        unload_after_predict: bool = False,
    ) -> None:
        self.prompts_path = prompts_path
        self.embeddings_path = embeddings_path
        self.window_path = window_path
        self.sample_rate = float(sample_rate)
        self.hpf_hz = float(hpf_hz)
        self.hpf_order = int(hpf_order)
        self.unload_after_predict = bool(unload_after_predict)
        self._dasm_dir = dasm_dir
        self._labels: list[str] = []
        self._open_queries: np.ndarray | None = None
        self._base_queries: np.ndarray | None = None
        self._hash: str | None = None
        self._window = load_window_config(window_path)
        self._bn0 = load_bn0(dasm_dir=dasm_dir) if dasm_dir else load_bn0()
        if session is not None:
            self._session = session
        else:
            kwargs: dict[str, Any] = {}
            if dasm_dir is not None:
                kwargs["dasm_dir"] = dasm_dir
            self._session = DasmSession(**kwargs)
        self._model_version = getattr(self._session, "model_version", "") or (
            model_version_fingerprint(dasm_dir)
            if dasm_dir is not None
            else model_version_fingerprint()
        )
        self.reload()

    @property
    def window(self) -> DasmWindowConfig:
        return self._window

    def reload(self) -> None:
        pairs = load_prompt_pairs(self.prompts_path)
        expected = prompts_hash(pairs)
        loaded = load_embeddings(self.embeddings_path)
        if loaded is None:
            raise FileNotFoundError(
                f"DASM embeddings missing at {self.embeddings_path}. "
                "Bake on a PC with scripts/bake_dasm_embeddings.py "
                "(--spike-root), commit the npz, then git pull on the Pi."
            )
        ok, err = embeddings_compatible(loaded)
        if not ok:
            raise RuntimeError(err or "DASM embeddings incompatible; rebuild required")
        if loaded.prompts_hash != expected:
            raise RuntimeError(
                "DASM embeddings are out of sync with dasm_prompts.json; rebuild them"
            )
        if [p.label for p in pairs] != loaded.labels:
            raise RuntimeError("DASM embedding labels do not match prompts file order")
        self._labels = loaded.labels
        self._open_queries = loaded.embeddings
        self._hash = loaded.prompts_hash
        self._base_queries = load_base_queries(
            self._dasm_dir if self._dasm_dir is not None else self._session.dasm_dir
        )
        self._window = load_window_config(self.window_path)

    def sync_status(self) -> dict[str, Any]:
        status = embedding_sync_status(
            prompts_path=self.prompts_path,
            embeddings_path=self.embeddings_path,
        )
        status["onnx"] = onnx_ready_status(
            self._dasm_dir if self._dasm_dir is not None else self._session.dasm_dir
        )
        status["window"] = {
            "pre_roll_s": self._window.pre_roll_s,
            "post_roll_s": self._window.post_roll_s,
            "frame_threshold": self._window.frame_threshold,
        }
        return status

    def predict(self, pcm_48k: np.ndarray) -> DasmPredictResult:
        if self._open_queries is None or self._base_queries is None or not self._labels:
            raise RuntimeError("DASM classifier has no queries loaded")
        started = time.perf_counter()
        mel, mel_meta = prepare_dasm_mel(
            pcm_48k,
            capture_rate=int(self.sample_rate),
            hpf_hz=self.hpf_hz,
            hpf_order=self.hpf_order,
            bn0=self._bn0,
            dasm_dir=self._session.dasm_dir,
        )
        base = self._base_queries
        open_q = self._open_queries
        query = np.concatenate([base, open_q], axis=0)
        base_len = int(base.shape[0])
        tgt_mask = build_tgt_mask(query.shape[0], base_len)

        if not self._session.loaded:
            self._session.load()
        try:
            strong, weak = self._session.run(mel, query, tgt_mask)
        finally:
            if self.unload_after_predict:
                self._session.unload()

        # strong: [B, Nq, T] or [Nq, T]; weak: [B, Nq] or [Nq]
        strong = np.asarray(strong, dtype=np.float32)
        weak = np.asarray(weak, dtype=np.float32)
        if strong.ndim == 3:
            strong = strong[0]
        if weak.ndim == 2:
            weak = weak[0]

        open_strong = strong[base_len:]
        open_weak = weak[base_len:]
        labels_out: list[dict[str, Any]] = []
        for i, label in enumerate(self._labels):
            spans = frames_to_spans(
                open_strong[i],
                threshold=self._window.frame_threshold,
                frames_per_second=self._window.frames_per_second,
                min_span_s=self._window.min_span_s,
            )
            labels_out.append(
                {
                    "label": label,
                    "clip_confidence": round(float(open_weak[i]), 4),
                    "spans": spans,
                }
            )
        labels_out.sort(key=lambda x: x["clip_confidence"], reverse=True)
        top = labels_out[0] if labels_out else None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DasmPredictResult(
            labels=labels_out,
            top_label=None if top is None else str(top["label"]),
            top_confidence=0.0 if top is None else float(top["clip_confidence"]),
            inference_ms=round(elapsed_ms, 1),
            mel_meta=mel_meta,
            model_name=DASM_MODEL_NAME,
            model_version=self._model_version,
            n_queries=int(query.shape[0]),
            base_len=base_len,
        )

    def close(self) -> None:
        self._session.unload()
