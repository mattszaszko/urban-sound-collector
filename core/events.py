"""Edge collector event schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.classifier_tflite import MODEL_NAME, MODEL_VERSION


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
    rms_unweighted: float,
    rms_a_weighted: float,
    dba_spl: float,
    predictions: List[Dict[str, Any]],
    spectrum: Optional[Dict[str, Any]] = None,
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one JSONL event (loudness + optional spectrum + YAMNet)."""
    top_label = predictions[0]["label"] if predictions else "n/a"
    top_confidence = float(predictions[0]["confidence"]) if predictions else 0.0

    event: Dict[str, Any] = {
        "device_id": device_id,
        "created_at": created_at or utc_now_iso(),
        "chunk_index": chunk_index,
        "run_id": run_id,
        "rms_unweighted": round(float(rms_unweighted), 8),
        "rms_a_weighted": round(float(rms_a_weighted), 8),
        "dBA_spl": round(float(dba_spl), 1),
        "top_label": top_label,
        "top_confidence": float(top_confidence),
        "predictions": predictions,
        "model_name": model_name,
        "model_version": model_version,
    }
    if spectrum is not None:
        event["spectrum"] = spectrum
    return event
