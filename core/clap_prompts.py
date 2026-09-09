"""CLAP prompt pairs and cached text-embedding helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = REPO_ROOT / "config" / "clap_prompts.json"
DEFAULT_EMBEDDINGS_PATH = REPO_ROOT / "config" / "clap_prompt_embeddings.npz"

# Expected by clap-htsat-unfused ONNX path
EXPECTED_EMBED_DIM = 512
EXPECTED_MODEL_NAME = "clap-htsat-unfused-onnx"


@dataclass(frozen=True)
class ClapPromptPair:
    label: str
    prompt: str


@dataclass(frozen=True)
class ClapEmbeddings:
    labels: list[str]
    embeddings: np.ndarray
    prompts_hash: str
    model_name: str
    embed_dim: int


def prompts_hash(pairs: list[ClapPromptPair]) -> str:
    payload = json.dumps(
        [{"label": p.label, "prompt": p.prompt} for p in pairs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_prompt_pairs(path: Path = DEFAULT_PROMPTS_PATH) -> list[ClapPromptPair]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("prompts", data if isinstance(data, list) else [])
    pairs: list[ClapPromptPair] = []
    seen: set[str] = set()
    for item in raw:
        label = str(item.get("label", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if not label or not prompt:
            raise ValueError("Each prompt pair needs non-empty label and prompt")
        if label in seen:
            raise ValueError(f"Duplicate CLAP label: {label}")
        seen.add(label)
        pairs.append(ClapPromptPair(label=label, prompt=prompt))
    return pairs


def save_prompt_pairs(
    pairs: list[ClapPromptPair], path: Path = DEFAULT_PROMPTS_PATH
) -> None:
    labels = [p.label for p in pairs]
    if len(labels) != len(set(labels)):
        raise ValueError("CLAP labels must be unique")
    for pair in pairs:
        if not pair.label.strip() or not pair.prompt.strip():
            raise ValueError("Each prompt pair needs non-empty label and prompt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompts": [{"label": p.label, "prompt": p.prompt} for p in pairs]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_embeddings(
    path: Path = DEFAULT_EMBEDDINGS_PATH,
) -> ClapEmbeddings | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    labels = [str(x) for x in data["labels"].tolist()]
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    file_hash = str(data["prompts_hash"].item()) if "prompts_hash" in data else ""
    model_name = (
        str(data["model_name"].item()) if "model_name" in data.files else ""
    )
    if "embed_dim" in data.files:
        embed_dim = int(data["embed_dim"].item())
    else:
        embed_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    return ClapEmbeddings(
        labels=labels,
        embeddings=embeddings,
        prompts_hash=file_hash,
        model_name=model_name,
        embed_dim=embed_dim,
    )


def save_embeddings(
    *,
    labels: list[str],
    embeddings: np.ndarray,
    file_hash: str,
    model_name: str,
    embed_dim: int,
    path: Path = DEFAULT_EMBEDDINGS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != embed_dim:
        raise ValueError(
            f"embeddings shape {matrix.shape} incompatible with embed_dim={embed_dim}"
        )
    np.savez_compressed(
        path,
        labels=np.asarray(labels),
        embeddings=matrix,
        prompts_hash=np.asarray(file_hash),
        model_name=np.asarray(model_name),
        embed_dim=np.asarray(embed_dim),
    )


def embeddings_compatible(loaded: ClapEmbeddings) -> tuple[bool, str | None]:
    """Return (ok, error_message)."""
    if loaded.model_name and loaded.model_name != EXPECTED_MODEL_NAME:
        return (
            False,
            f"Embeddings were built with model '{loaded.model_name}'; "
            f"rebuild required for '{EXPECTED_MODEL_NAME}'",
        )
    if not loaded.model_name:
        return (
            False,
            "Embeddings missing model_name (placeholder or legacy cache); rebuild required",
        )
    if loaded.embed_dim != EXPECTED_EMBED_DIM or loaded.embeddings.shape[1] != EXPECTED_EMBED_DIM:
        return (
            False,
            f"Embeddings dim {loaded.embed_dim} != {EXPECTED_EMBED_DIM}; rebuild required",
        )
    return True, None


def embedding_sync_status(
    pairs: list[ClapPromptPair] | None = None,
    *,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
) -> dict[str, Any]:
    if pairs is None:
        pairs = load_prompt_pairs(prompts_path) if prompts_path.exists() else []
    expected = prompts_hash(pairs)
    loaded = load_embeddings(embeddings_path)
    if loaded is None:
        return {
            "in_sync": False,
            "prompt_count": len(pairs),
            "embedding_count": 0,
            "prompts_hash": expected,
            "embeddings_hash": None,
            "embeddings_path": str(embeddings_path),
            "model_name": None,
            "embed_dim": None,
            "compatible": False,
            "compat_error": "Embeddings file missing; rebuild required",
        }
    compatible, compat_error = embeddings_compatible(loaded)
    return {
        "in_sync": bool(
            compatible
            and loaded.prompts_hash == expected
            and len(loaded.labels) == len(pairs)
        ),
        "prompt_count": len(pairs),
        "embedding_count": int(loaded.embeddings.shape[0]),
        "prompts_hash": expected,
        "embeddings_hash": loaded.prompts_hash,
        "embeddings_path": str(embeddings_path),
        "model_name": loaded.model_name or None,
        "embed_dim": loaded.embed_dim,
        "compatible": compatible,
        "compat_error": compat_error,
    }
