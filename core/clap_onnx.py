"""ONNX Runtime wrappers for CLAP text and audio towers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.clap_preprocess import extract_clap_features, load_preprocessor_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAP_DIR = REPO_ROOT / "models" / "clap"

CLAP_MODEL_NAME = "clap-htsat-unfused-onnx"
EMBED_DIM = 512
TEXT_MAX_LENGTH = 77

AUDIO_ONNX_NAME = "audio_model_quantized.onnx"
TEXT_ONNX_NAME = "text_model_quantized.onnx"
TOKENIZER_NAME = "tokenizer.json"


class ClapModelsMissingError(FileNotFoundError):
    """Raised when required ONNX / tokenizer files are not installed."""


def clap_model_paths(clap_dir: Path = DEFAULT_CLAP_DIR) -> dict[str, Path]:
    return {
        "audio": clap_dir / AUDIO_ONNX_NAME,
        "text": clap_dir / TEXT_ONNX_NAME,
        "tokenizer": clap_dir / TOKENIZER_NAME,
        "preprocessor": clap_dir / "preprocessor_config.json",
    }


def missing_clap_assets(clap_dir: Path = DEFAULT_CLAP_DIR) -> list[str]:
    missing: list[str] = []
    for key, path in clap_model_paths(clap_dir).items():
        if key == "preprocessor":
            continue  # optional if defaults baked into preprocess module
        if not path.exists():
            missing.append(path.name)
    return missing


def require_clap_assets(clap_dir: Path = DEFAULT_CLAP_DIR, *, need_text: bool = False, need_audio: bool = False) -> None:
    paths = clap_model_paths(clap_dir)
    needed: list[Path] = []
    if need_text:
        needed.extend([paths["text"], paths["tokenizer"]])
    if need_audio:
        needed.append(paths["audio"])
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise ClapModelsMissingError(
            "CLAP ONNX assets missing: "
            + ", ".join(missing)
            + ". Run: python scripts/download_clap_models.py"
        )


def onnx_ready_status(clap_dir: Path = DEFAULT_CLAP_DIR) -> dict[str, Any]:
    paths = clap_model_paths(clap_dir)
    audio_ok = paths["audio"].exists()
    text_ok = paths["text"].exists() and paths["tokenizer"].exists()
    missing = missing_clap_assets(clap_dir)
    return {
        "ready": audio_ok and text_ok,
        "audio_ready": audio_ok,
        "text_ready": text_ok,
        "missing": missing,
        "clap_dir": str(clap_dir),
        "model_name": CLAP_MODEL_NAME,
        "model_version": model_version_fingerprint(clap_dir) if audio_ok else None,
        "hint": None
        if (audio_ok and text_ok)
        else "Run: python scripts/download_clap_models.py",
    }


def model_version_fingerprint(clap_dir: Path = DEFAULT_CLAP_DIR) -> str:
    """Short version string from audio ONNX SHA256 prefix."""
    audio = clap_model_paths(clap_dir)["audio"]
    if not audio.exists():
        return "missing"
    h = hashlib.sha256()
    with audio.open("rb") as f:
        # First 1 MiB is enough for a stable fingerprint without hashing 34 MB every call.
        h.update(f.read(1024 * 1024))
        h.update(str(audio.stat().st_size).encode("ascii"))
    return h.hexdigest()[:12]


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (x / denom).astype(np.float32)


def _pick_name(names: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _ort_session(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for CLAP. pip install onnxruntime"
        ) from exc
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


class TextEncoder:
    """CLAP text tower — load for rebuild, then close to free RAM."""

    def __init__(self, clap_dir: Path = DEFAULT_CLAP_DIR) -> None:
        require_clap_assets(clap_dir, need_text=True)
        paths = clap_model_paths(clap_dir)
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "tokenizers is required for CLAP text embeds. pip install tokenizers"
            ) from exc
        self._tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
        self._tokenizer.enable_truncation(max_length=TEXT_MAX_LENGTH)
        # RoBERTa pad_token_id is typically 1.
        self._tokenizer.enable_padding(
            length=TEXT_MAX_LENGTH,
            pad_id=1,
            pad_token="<pad>",
        )
        self._session = _ort_session(paths["text"])
        inputs = [i.name for i in self._session.get_inputs()]
        outputs = [o.name for o in self._session.get_outputs()]
        self._input_ids = _pick_name(inputs, ["input_ids"]) or inputs[0]
        self._attention = _pick_name(inputs, ["attention_mask"])
        if self._attention is None and len(inputs) > 1:
            self._attention = inputs[1]
        self._output = _pick_name(
            outputs, ["text_embeds", "embeddings", "last_hidden_state"]
        ) or outputs[0]
        self.model_version = model_version_fingerprint(clap_dir)

    def close(self) -> None:
        self._session = None
        self._tokenizer = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._session is None or self._tokenizer is None:
            raise RuntimeError("TextEncoder is closed")
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        enc = self._tokenizer.encode_batch(list(texts))
        input_ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        feeds: dict[str, np.ndarray] = {self._input_ids: input_ids}
        if self._attention is not None:
            # tokenizers padding sets attention via type ids; build mask from non-pad.
            attention = (input_ids != 1).astype(np.int64)
            # Also respect encode attention if present.
            if all(hasattr(e, "attention_mask") for e in enc):
                attention = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
            feeds[self._attention] = attention
        out = self._session.run([self._output], feeds)[0]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 3:
            # Take CLS / first token if sequence output.
            out = out[:, 0, :]
        if out.shape[-1] != EMBED_DIM:
            raise RuntimeError(
                f"Unexpected text embed dim {out.shape[-1]} (expected {EMBED_DIM})"
            )
        return _l2_normalize(out, axis=1)


class AudioEncoder:
    """CLAP audio tower — kept resident while --enable-clap is active."""

    def __init__(self, clap_dir: Path = DEFAULT_CLAP_DIR) -> None:
        require_clap_assets(clap_dir, need_audio=True)
        paths = clap_model_paths(clap_dir)
        self._cfg = load_preprocessor_config(paths["preprocessor"])
        self._session = _ort_session(paths["audio"])
        inputs = [i.name for i in self._session.get_inputs()]
        outputs = [o.name for o in self._session.get_outputs()]
        self._feat_name = _pick_name(inputs, ["input_features"]) or inputs[0]
        self._longer_name = _pick_name(inputs, ["is_longer"])
        self._output = _pick_name(
            outputs, ["audio_embeds", "embeddings", "last_hidden_state"]
        ) or outputs[0]
        self.model_version = model_version_fingerprint(clap_dir)

    def close(self) -> None:
        self._session = None

    def embed(self, pcm_48k: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("AudioEncoder is closed")
        features, is_longer = extract_clap_features(pcm_48k, cfg=self._cfg)
        feeds: dict[str, np.ndarray] = {self._feat_name: features}
        if self._longer_name is not None:
            # Some exports want bool, others int64.
            longer_type = None
            for inp in self._session.get_inputs():
                if inp.name == self._longer_name:
                    longer_type = inp.type
                    break
            if longer_type and "int64" in longer_type:
                feeds[self._longer_name] = is_longer.astype(np.int64)
            else:
                feeds[self._longer_name] = is_longer
        out = self._session.run([self._output], feeds)[0]
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if out.size != EMBED_DIM:
            raise RuntimeError(
                f"Unexpected audio embed dim {out.size} (expected {EMBED_DIM})"
            )
        return _l2_normalize(out[np.newaxis, :], axis=1)[0]
