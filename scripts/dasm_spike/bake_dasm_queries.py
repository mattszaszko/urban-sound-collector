#!/usr/bin/env python3
"""Bake offline DASM text-query vectors with MGA-CLAP (Phase 0).

Run on the Pi from the Transformer4SED tree after export_dasm_onnx.py:

  cd ~/dasm-spike/Transformer4SED
  source .venv/bin/activate
  python bake_dasm_queries.py --prompts alarm siren dog music bird

Product runtime should load only:
  - dasm_text_fp32.onnx  (ORT)
  - this bake output     (numpy query pack)

…and never import MGA-CLAP / BERT on the Pi.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def _force_torch_load_cpu() -> None:
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


def get_att_mask(query_len: int, base_len: int) -> torch.Tensor:
    att_mask = torch.ones(query_len, query_len, dtype=torch.bool)
    att_mask[:, :base_len] = False
    att_mask.fill_diagonal_(False)
    return att_mask


def load_clap(root: Path, clap_ckpt: Path):
    os.chdir(root / "third_parties" / "MGA-CLAP")
    sys.path.insert(0, str(root / "third_parties" / "MGA-CLAP"))
    from models.ase_model import ASE

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


@torch.no_grad()
def encode_prompts(clap, labels: list[str], prompt_prefix: str) -> torch.Tensor:
    queries = [f"{prompt_prefix}{x.lower()}" for x in labels]
    _, word_embeds, attn_mask = clap.encode_text(queries)
    text_embeds = clap.msc(word_embeds, clap.codebook, attn_mask)
    return F.normalize(text_embeds, dim=-1)


def load_base_queries(path: Path) -> torch.Tensor:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        for key in ("query", "text_query", "at_query", "model"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key].detach().cpu()
        # first tensor value
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v.detach().cpu()
    raise TypeError(f"unrecognized query file structure: {type(obj)} keys={getattr(obj, 'keys', lambda: None)()}")


def ort_smoke(onnx_path: Path, mel_path: Path, query: np.ndarray, tgt_mask: np.ndarray) -> None:
    import onnxruntime as ort

    if not mel_path.is_file():
        print(f"skip ORT smoke (missing {mel_path})")
        return
    mel = np.load(mel_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    strong, weak = sess.run(
        None, {"mel": mel, "query": query, "tgt_mask": tgt_mask}
    )
    dt = time.perf_counter() - t0
    print(f"ORT smoke ok in {dt:.2f}s  strong={strong.shape} weak={weak.shape}")
    # show top open-vocab weak scores (skip base block)
    print(f"weak min/max: {weak.min():.4f}/{weak.max():.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("T4SED_ROOT", os.getcwd()),
        help="Transformer4SED root",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["alarm", "sirens", "air raid sirens", "dog", "music", "bird"],
        help="Open-vocab class names (without 'sound of ' prefix)",
    )
    parser.add_argument("--prompt-prefix", default="sound of ")
    parser.add_argument(
        "--clap-ckpt",
        default="pretrained_model/detect_any_sound/htast_mga.pt",
    )
    parser.add_argument(
        "--base-query",
        default="pretrained_model/detect_any_sound/text_query/text_query.pt",
        help="Checkpoint base queries (407 x 1024 for text_query model)",
    )
    parser.add_argument("--out-dir", default="onnx_export")
    parser.add_argument("--ort-smoke", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clap_ckpt = root / args.clap_ckpt
    base_path = root / args.base_query
    if not clap_ckpt.is_file():
        print(f"missing CLAP ckpt: {clap_ckpt}", file=sys.stderr)
        return 1
    if not base_path.is_file():
        print(f"missing base query: {base_path}", file=sys.stderr)
        return 1

    _force_torch_load_cpu()

    print("loading MGA-CLAP (one-time bake)…")
    t0 = time.perf_counter()
    clap = load_clap(root, clap_ckpt)
    print(f"CLAP ready in {time.perf_counter() - t0:.1f}s")

    print(f"encoding {len(args.prompts)} prompts…")
    t0 = time.perf_counter()
    open_vocab = encode_prompts(clap, args.prompts, args.prompt_prefix).cpu()
    print(f"encode wall: {time.perf_counter() - t0:.2f}s  shape={tuple(open_vocab.shape)}")

    # Free CLAP before packing — bake machine can be the Pi, but runtime won't load this.
    del clap
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    os.chdir(root)
    base = load_base_queries(base_path)
    if base.ndim != 2:
        print(f"base query unexpected shape: {tuple(base.shape)}", file=sys.stderr)
        return 1
    if open_vocab.shape[-1] != base.shape[-1]:
        print(
            f"dim mismatch base {base.shape[-1]} vs open {open_vocab.shape[-1]}",
            file=sys.stderr,
        )
        return 1

    base_len = base.shape[0]
    query = torch.cat([base, open_vocab], dim=0)
    tgt_mask = get_att_mask(query.shape[0], base_len)

    query_np = query.numpy().astype(np.float32)
    mask_np = tgt_mask.numpy()

    labels = {
        "base_len": base_len,
        "prompt_prefix": args.prompt_prefix,
        "open_vocab_labels": list(args.prompts),
        "open_vocab_prompts": [f"{args.prompt_prefix}{x.lower()}" for x in args.prompts],
        "query_shape": list(query_np.shape),
        "note": "query rows [0:base_len) are checkpoint base; rest are baked open-vocab",
    }

    np.save(out_dir / "dasm_query.npy", query_np)
    np.save(out_dir / "dasm_tgt_mask.npy", mask_np)
    np.savez_compressed(
        out_dir / "dasm_query_pack.npz",
        query=query_np,
        tgt_mask=mask_np,
        base_len=np.asarray([base_len], dtype=np.int32),
        open_vocab=open_vocab.numpy().astype(np.float32),
    )
    with open(out_dir / "dasm_query_meta.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"wrote {out_dir / 'dasm_query.npy'}  {query_np.shape}")
    print(f"wrote {out_dir / 'dasm_tgt_mask.npy'}  {mask_np.shape}")
    print(f"wrote {out_dir / 'dasm_query_pack.npz'}")
    print(f"wrote {out_dir / 'dasm_query_meta.json'}")

    if args.ort_smoke:
        ort_smoke(
            out_dir / "dasm_text_fp32.onnx",
            out_dir / "smoke_mel.npy",
            query_np,
            mask_np,
        )

    print("done — Pi runtime can use ORT + these vectors only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
