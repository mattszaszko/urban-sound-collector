"""Edge collector event schema for live classification output."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1
MODEL_NAME = "yamnet"
DEFAULT_MODEL_VERSION = "TensorFlow2/yamnet/1"


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string with Z suffix."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def new_run_id() -> str:
    """Create a per-process run identifier from the UTC start time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def build_noise_event(
    *,
    device_id: str,
    chunk_index: int,
    run_id: str,
    rms: float,
    predictions: List[Dict[str, Any]],
    model_version: str = DEFAULT_MODEL_VERSION,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one canonical noise-event payload for stdout (and later Firestore).

    ``chunk_index`` is a per-run counter only. Prefer ``created_at`` for
    time-series ordering across restarts and devices.
    """
    top_label = predictions[0]["label"] if predictions else "n/a"
    top_confidence = float(predictions[0]["confidence"]) if predictions else 0.0

    return {
        "device_id": device_id,
        "created_at": created_at or utc_now_iso(),
        "chunk_index": chunk_index,
        "run_id": run_id,
        "rms": round(float(rms), 8),
        "top_label": top_label,
        "top_confidence": float(top_confidence),
        "predictions": predictions,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "schema_version": SCHEMA_VERSION,
    }
