#!/usr/bin/env python3
"""Bake DASM open-vocab query vectors for the product runtime (PC / CI).

The Pi collector never loads MGA-CLAP / BERT. Edit prompts in the web UI (or
``config/dasm_prompts.json``), then run this script on a machine that has the
Phase 0 spike tree (Transformer4SED + MGA-CLAP + ``htast_mga.pt``). Commit the
resulting ``config/dasm_query_embeddings.npz`` and pull it on the Pi.

Example (from this repo, spike elsewhere)::

  # Spike venv has torch + MGA-CLAP deps; product venv is fine for numpy write
  # if the spike root is reachable — typically run with the spike interpreter:
  /path/to/dasm-spike/Transformer4SED/.venv/bin/python \\
    scripts/bake_dasm_embeddings.py \\
    --spike-root /path/to/dasm-spike/Transformer4SED

  git add config/dasm_prompts.json config/dasm_query_embeddings.npz
  git commit -m "Update DASM prompt embeddings"
  # on Pi: git pull
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dasm_prompts import (  # noqa: E402
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_PROMPTS_PATH,
    EXPECTED_MODEL_NAME,
    load_prompt_pairs,
    prompts_hash,
    save_embeddings,
)


def _force_torch_load_cpu() -> None:
    import torch

    orig = torch.load

    def _load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]


def _patch_sparsemax_cpu() -> None:
    """MGA-CLAP Sparsemax defaults device='cuda'."""
    from models.ase_model import Sparsemax

    orig = Sparsemax.forward

    def _forward(self, input, device=None):
        if device is None:
            device = input.device
        return orig(self, input, device=device)

    Sparsemax.forward = _forward  # type: ignore[assignment]


def load_clap(spike_root: Path, clap_ckpt: Path):
    import torch
    import yaml

    mga = spike_root / "third_parties" / "MGA-CLAP"
    if not mga.is_dir():
        raise FileNotFoundError(f"MGA-CLAP missing under spike root: {mga}")

    prev_cwd = Path.cwd()
    prev_path = list(sys.path)
    try:
        os.chdir(mga)
        sys.path.insert(0, str(mga))
        from models.ase_model import ASE  # type: ignore

        _patch_sparsemax_cpu()
        with open("settings/inference_sed.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        clap = ASE(config).to("cpu")
        ckpt = torch.load(str(clap_ckpt), map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        clap.load_state_dict(state, strict=False)
        clap.eval()
        return clap
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


def encode_prompt_texts(clap, texts: list[str]):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        _, word_embeds, attn_mask = clap.encode_text(texts)
        text_embeds = clap.msc(word_embeds, clap.codebook, attn_mask)
        return F.normalize(text_embeds, dim=-1)


def resolve_spike_root(cli_value: str | None) -> Path:
    raw = (
        cli_value
        or os.environ.get("DASM_SPIKE_ROOT")
        or os.environ.get("T4SED_ROOT")
        or ""
    ).strip()
    if not raw:
        raise SystemExit(
            "Pass --spike-root PATH (or set DASM_SPIKE_ROOT / T4SED_ROOT) to the "
            "Transformer4SED tree that contains third_parties/MGA-CLAP and "
            "pretrained_model/detect_any_sound/htast_mga.pt"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"spike root is not a directory: {root}")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bake product DASM embeddings with MGA-CLAP (offline / PC)."
    )
    parser.add_argument(
        "--spike-root",
        default=None,
        help="Transformer4SED root (env: DASM_SPIKE_ROOT or T4SED_ROOT)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Product prompts JSON (default: config/dasm_prompts.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Output npz (default: config/dasm_query_embeddings.npz)",
    )
    parser.add_argument(
        "--clap-ckpt",
        default="pretrained_model/detect_any_sound/htast_mga.pt",
        help="Relative to spike root",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify prompts file + report hash; do not load MGA-CLAP",
    )
    args = parser.parse_args(argv)

    prompts_path = args.prompts if args.prompts.is_absolute() else REPO_ROOT / args.prompts
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out

    if not prompts_path.is_file():
        print(f"missing prompts: {prompts_path}", file=sys.stderr)
        return 1

    pairs = load_prompt_pairs(prompts_path)
    if not pairs:
        print("no prompts to bake", file=sys.stderr)
        return 1

    file_hash = prompts_hash(pairs)
    print(f"prompts: {prompts_path}  ({len(pairs)} rows, hash {file_hash})")
    for p in pairs:
        print(f"  - {p.label!r}: {p.prompt!r}")

    if args.check:
        print("check ok (no bake)")
        return 0

    spike_root = resolve_spike_root(args.spike_root)
    clap_ckpt = spike_root / args.clap_ckpt
    if not clap_ckpt.is_file():
        print(f"missing MGA-CLAP checkpoint: {clap_ckpt}", file=sys.stderr)
        return 1

    _force_torch_load_cpu()
    print(f"loading MGA-CLAP from {spike_root} …")
    t0 = time.perf_counter()
    clap = load_clap(spike_root, clap_ckpt)
    print(f"CLAP ready in {time.perf_counter() - t0:.1f}s")

    texts = [p.prompt for p in pairs]
    print(f"encoding {len(texts)} prompt string(s)…")
    t0 = time.perf_counter()
    embeds = encode_prompt_texts(clap, texts).cpu().numpy().astype(np.float32)
    print(f"encode wall: {time.perf_counter() - t0:.2f}s  shape={embeds.shape}")

    del clap

    save_embeddings(
        labels=[p.label for p in pairs],
        embeddings=embeds,
        file_hash=file_hash,
        model_name=EXPECTED_MODEL_NAME,
        path=out_path,
    )
    meta = {
        "prompts_path": str(prompts_path),
        "embeddings_path": str(out_path),
        "prompts_hash": file_hash,
        "model_name": EXPECTED_MODEL_NAME,
        "labels": [p.label for p in pairs],
        "count": len(pairs),
        "embed_dim": int(embeds.shape[1]),
    }
    print(json.dumps(meta, indent=2))
    print(
        f"wrote {out_path}\n"
        "Next: commit this file (+ prompts JSON if changed), pull on the Pi."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
