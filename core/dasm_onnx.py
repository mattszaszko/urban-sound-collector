"""ONNX Runtime wrapper for DASM (text-query) inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASM_DIR = REPO_ROOT / "models" / "dasm"

DASM_MODEL_NAME = "dasm-htsat-text-onnx"
ONNX_NAME = "dasm_text_fp32.onnx"
BASE_QUERY_NAME = "base_query.npy"
# ONNX attention Reshape ops are frozen at export-time query count.
# Runtime always pads open-vocab rows to this budget (re-export if you raise it).
OPEN_VOCAB_SLOTS = 16
BN0_NAME = "bn0.npz"

QUERY_DIM = 1024


class DasmModelsMissingError(FileNotFoundError):
    """Raised when required DASM ONNX / base-query assets are missing."""


def dasm_model_paths(dasm_dir: Path = DEFAULT_DASM_DIR) -> dict[str, Path]:
    return {
        "onnx": dasm_dir / ONNX_NAME,
        "base_query": dasm_dir / BASE_QUERY_NAME,
        "bn0": dasm_dir / BN0_NAME,
    }


def missing_dasm_assets(dasm_dir: Path = DEFAULT_DASM_DIR) -> list[str]:
    paths = dasm_model_paths(dasm_dir)
    missing: list[str] = []
    for key in ("onnx", "base_query"):
        if not paths[key].exists():
            missing.append(paths[key].name)
    return missing


def require_dasm_assets(dasm_dir: Path = DEFAULT_DASM_DIR) -> None:
    missing = missing_dasm_assets(dasm_dir)
    if missing:
        raise DasmModelsMissingError(
            "DASM assets missing: "
            + ", ".join(missing)
            + ". Copy spike onnx_export outputs into models/dasm/ "
            "(see docs/DASM-SPIKE.md / models/README.md)."
        )


def onnx_ready_status(dasm_dir: Path = DEFAULT_DASM_DIR) -> dict[str, Any]:
    paths = dasm_model_paths(dasm_dir)
    missing = missing_dasm_assets(dasm_dir)
    ready = not missing
    return {
        "ready": ready,
        "missing": missing,
        "dasm_dir": str(dasm_dir),
        "model_name": DASM_MODEL_NAME,
        "model_version": model_version_fingerprint(dasm_dir) if paths["onnx"].exists() else None,
        "bn0_ready": paths["bn0"].exists(),
        "hint": None
        if ready
        else "Install models/dasm/dasm_text_fp32.onnx and base_query.npy (see docs/DASM-SPIKE.md)",
    }


def model_version_fingerprint(dasm_dir: Path = DEFAULT_DASM_DIR) -> str:
    onnx_path = dasm_model_paths(dasm_dir)["onnx"]
    if not onnx_path.exists():
        return "missing"
    h = hashlib.sha256()
    with onnx_path.open("rb") as f:
        h.update(f.read(1024 * 1024))
        h.update(str(onnx_path.stat().st_size).encode("ascii"))
    return h.hexdigest()[:12]


def _ort_session(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for DASM. pip install onnxruntime"
        ) from exc
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def load_base_queries(dasm_dir: Path = DEFAULT_DASM_DIR) -> np.ndarray:
    path = dasm_model_paths(dasm_dir)["base_query"]
    if not path.exists():
        raise DasmModelsMissingError(f"Missing {path.name}")
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != QUERY_DIM:
        raise RuntimeError(
            f"base_query shape {arr.shape} incompatible with dim {QUERY_DIM}"
        )
    return arr


def build_tgt_mask(
    query_len: int,
    base_len: int,
    *,
    real_open: int | None = None,
) -> np.ndarray:
    """Attention mask matching Transformer4SED notebook helper.

    ``True`` = blocked. Everyone may attend to base queries and to self.
    When ``real_open`` is set, trailing pad slots (``base_len + real_open`` …)
    are fully blocked from attending / being attended (diagonal kept open).
    """
    att = np.ones((query_len, query_len), dtype=bool)
    att[:, :base_len] = False
    np.fill_diagonal(att, False)
    if real_open is not None:
        pad_start = int(base_len) + max(0, int(real_open))
        if 0 <= pad_start < query_len:
            att[pad_start:, :] = True
            att[:, pad_start:] = True
            for i in range(pad_start, query_len):
                att[i, i] = False
    return att


def pad_open_vocab_queries(
    open_q: np.ndarray,
    *,
    slots: int = OPEN_VOCAB_SLOTS,
) -> tuple[np.ndarray, int]:
    """Pad open-vocab embeddings to the ONNX slot budget; return (padded, real_n)."""
    matrix = np.asarray(open_q, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"open-vocab queries must be 2-D, got {matrix.shape}")
    real_n = int(matrix.shape[0])
    budget = int(slots)
    if budget < 1:
        raise ValueError("open-vocab slot budget must be >= 1")
    if real_n > budget:
        raise ValueError(
            f"Got {real_n} open-vocab queries; ONNX budget is {budget}. "
            "Remove prompts or re-export ONNX with a larger --n-extra."
        )
    if real_n == budget:
        return matrix, real_n
    out = np.zeros((budget, matrix.shape[1]), dtype=np.float32)
    if real_n:
        out[:real_n] = matrix
    return out, real_n


class DasmSession:
    """ORT session for DASM mel+query → strong/weak. Supports load/unload."""

    def __init__(self, dasm_dir: Path = DEFAULT_DASM_DIR) -> None:
        require_dasm_assets(dasm_dir)
        self.dasm_dir = Path(dasm_dir)
        self.model_version = model_version_fingerprint(dasm_dir)
        self._session = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self.load()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self._session is not None:
            return
        path = dasm_model_paths(self.dasm_dir)["onnx"]
        self._session = _ort_session(path)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

    def unload(self) -> None:
        self._session = None

    close = unload

    def run(
        self,
        mel: np.ndarray,
        query: np.ndarray,
        tgt_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._session is None:
            raise RuntimeError("DasmSession is unloaded")
        mel_np = np.asarray(mel, dtype=np.float32)
        query_np = np.asarray(query, dtype=np.float32)
        mask_np = np.asarray(tgt_mask)
        if mel_np.ndim == 3:
            mel_np = mel_np[np.newaxis, ...]
        feeds = {
            "mel": mel_np,
            "query": query_np,
            "tgt_mask": mask_np,
        }
        # Allow alternate export input names if present.
        renamed: dict[str, np.ndarray] = {}
        for name in self._input_names:
            key = name.lower()
            if "mel" in key:
                renamed[name] = feeds["mel"]
            elif "query" in key:
                renamed[name] = feeds["query"]
            elif "mask" in key:
                renamed[name] = feeds["tgt_mask"]
            elif name in feeds:
                renamed[name] = feeds[name]
        if len(renamed) < 3:
            renamed = {k: feeds[k] for k in ("mel", "query", "tgt_mask")}
        outs = self._session.run(self._output_names or None, renamed)
        strong = np.asarray(outs[0], dtype=np.float32)
        weak = np.asarray(outs[1], dtype=np.float32)
        return strong, weak
