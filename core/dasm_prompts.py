"""DASM open-vocab prompts, window settings, and cached query vectors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.dasm_onnx import QUERY_DIM

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = REPO_ROOT / "config" / "dasm_prompts.json"
DEFAULT_EMBEDDINGS_PATH = REPO_ROOT / "config" / "dasm_query_embeddings.npz"
DEFAULT_WINDOW_PATH = REPO_ROOT / "config" / "dasm_window.json"

EXPECTED_MODEL_NAME = "dasm-htsat-text-onnx"
DEFAULT_PROMPT_PREFIX = "sound of "
DEFAULT_PRE_ROLL_S = 5.0
DEFAULT_POST_ROLL_S = 5.0
DEFAULT_FRAME_THRESHOLD = 0.35
DEFAULT_MIN_SPAN_S = 0.15
# Spike strong output is 500 frames over ~10 s.
DEFAULT_FRAMES_PER_SECOND = 50.0


@dataclass(frozen=True)
class DasmPromptPair:
    label: str
    prompt: str


@dataclass(frozen=True)
class DasmWindowConfig:
    pre_roll_s: float = DEFAULT_PRE_ROLL_S
    post_roll_s: float = DEFAULT_POST_ROLL_S
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD
    min_span_s: float = DEFAULT_MIN_SPAN_S
    frames_per_second: float = DEFAULT_FRAMES_PER_SECOND

    def normalized(self) -> "DasmWindowConfig":
        return DasmWindowConfig(
            pre_roll_s=max(0.1, float(self.pre_roll_s)),
            post_roll_s=max(0.1, float(self.post_roll_s)),
            frame_threshold=float(min(1.0, max(0.0, self.frame_threshold))),
            min_span_s=max(0.0, float(self.min_span_s)),
            frames_per_second=max(1.0, float(self.frames_per_second)),
        )


@dataclass(frozen=True)
class DasmEmbeddings:
    labels: list[str]
    embeddings: np.ndarray
    prompts_hash: str
    model_name: str
    embed_dim: int


def prompts_hash(pairs: list[DasmPromptPair]) -> str:
    payload = json.dumps(
        [{"label": p.label, "prompt": p.prompt} for p in pairs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_prompt_pairs(path: Path = DEFAULT_PROMPTS_PATH) -> list[DasmPromptPair]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("prompts", data if isinstance(data, list) else [])
    pairs: list[DasmPromptPair] = []
    seen: set[str] = set()
    for item in raw:
        label = str(item.get("label", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if not label or not prompt:
            raise ValueError("Each DASM prompt pair needs non-empty label and prompt")
        if label in seen:
            raise ValueError(f"Duplicate DASM label: {label}")
        seen.add(label)
        pairs.append(DasmPromptPair(label=label, prompt=prompt))
    return pairs


def save_prompt_pairs(
    pairs: list[DasmPromptPair], path: Path = DEFAULT_PROMPTS_PATH
) -> None:
    labels = [p.label for p in pairs]
    if len(labels) != len(set(labels)):
        raise ValueError("DASM labels must be unique")
    for pair in pairs:
        if not pair.label.strip() or not pair.prompt.strip():
            raise ValueError("Each DASM prompt pair needs non-empty label and prompt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_prefix": DEFAULT_PROMPT_PREFIX,
        "prompts": [{"label": p.label, "prompt": p.prompt} for p in pairs],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_window_config(path: Path = DEFAULT_WINDOW_PATH) -> DasmWindowConfig:
    if not path.exists():
        return DasmWindowConfig().normalized()
    data = json.loads(path.read_text(encoding="utf-8"))
    return DasmWindowConfig(
        pre_roll_s=float(data.get("pre_roll_s", DEFAULT_PRE_ROLL_S)),
        post_roll_s=float(data.get("post_roll_s", DEFAULT_POST_ROLL_S)),
        frame_threshold=float(data.get("frame_threshold", DEFAULT_FRAME_THRESHOLD)),
        min_span_s=float(data.get("min_span_s", DEFAULT_MIN_SPAN_S)),
        frames_per_second=float(
            data.get("frames_per_second", DEFAULT_FRAMES_PER_SECOND)
        ),
    ).normalized()


def save_window_config(
    cfg: DasmWindowConfig, path: Path = DEFAULT_WINDOW_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = cfg.normalized()
    payload = {
        "pre_roll_s": n.pre_roll_s,
        "post_roll_s": n.post_roll_s,
        "frame_threshold": n.frame_threshold,
        "min_span_s": n.min_span_s,
        "frames_per_second": n.frames_per_second,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_embeddings(path: Path = DEFAULT_EMBEDDINGS_PATH) -> DasmEmbeddings | None:
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
    return DasmEmbeddings(
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
    model_name: str = EXPECTED_MODEL_NAME,
    embed_dim: int = QUERY_DIM,
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


def embeddings_compatible(loaded: DasmEmbeddings) -> tuple[bool, str | None]:
    if loaded.model_name and loaded.model_name != EXPECTED_MODEL_NAME:
        return (
            False,
            f"Embeddings were built with model '{loaded.model_name}'; "
            f"rebuild required for '{EXPECTED_MODEL_NAME}'",
        )
    if not loaded.model_name:
        return False, "Embeddings missing model_name; rebuild required"
    if loaded.embed_dim != QUERY_DIM or loaded.embeddings.shape[1] != QUERY_DIM:
        return (
            False,
            f"Embeddings dim {loaded.embed_dim} != {QUERY_DIM}; rebuild required",
        )
    return True, None


def embedding_sync_status(
    pairs: list[DasmPromptPair] | None = None,
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
            "compat_error": (
                "Embeddings file missing; bake on a PC with "
                "scripts/bake_dasm_embeddings.py then git pull"
            ),
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
