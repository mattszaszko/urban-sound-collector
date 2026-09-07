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


@dataclass(frozen=True)
class ClapPromptPair:
    label: str
    prompt: str


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
) -> tuple[list[str], np.ndarray, str] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    labels = [str(x) for x in data["labels"].tolist()]
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    file_hash = str(data["prompts_hash"].item()) if "prompts_hash" in data else ""
    return labels, embeddings, file_hash


def save_embeddings(
    *,
    labels: list[str],
    embeddings: np.ndarray,
    file_hash: str,
    path: Path = DEFAULT_EMBEDDINGS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.asarray(labels),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        prompts_hash=np.asarray(file_hash),
    )


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
        }
    labels, embeddings, file_hash = loaded
    return {
        "in_sync": file_hash == expected and len(labels) == len(pairs),
        "prompt_count": len(pairs),
        "embedding_count": int(embeddings.shape[0]),
        "prompts_hash": expected,
        "embeddings_hash": file_hash,
        "embeddings_path": str(embeddings_path),
    }
