#!/usr/bin/env python3
"""Extract HTS-AT bn0 BatchNorm stats for DASM mel preprocess parity.

Run inside the Transformer4SED spike venv after the model loads:

  cd ~/dasm-spike/Transformer4SED
  source .venv/bin/activate
  python extract_dasm_bn0.py --out ~/urban-sound-collector/models/dasm/bn0.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml_with_relative_ref(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        main_content = yaml.safe_load(f)
    if isinstance(main_content, dict) and "include" in main_content:
        included = main_content.pop("include")
        base = load_yaml_with_relative_ref(included["base_path"])
        for key in included["keys"]:
            main_content[key] = base[key]
    return main_content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.getcwd())
    parser.add_argument(
        "--out",
        default="bn0.npz",
        help="Output npz path (weight, bias, running_mean, running_var, eps)",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    orig = torch.load

    def _load(*a, **k):
        k["map_location"] = "cpu"
        return orig(*a, **k)

    torch.load = _load  # type: ignore[assignment]

    from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT

    config_path = root / "pretrained_model/detect_any_sound/text_query/config.yaml"
    weight_path = (
        root / "pretrained_model/detect_any_sound/text_query/as_full_text_query_best_model.pt"
    )
    configs = load_yaml_with_relative_ref(str(config_path))
    model = DASM_HTSAT(**configs["DASM_HTSAT"]["init_kwargs"])
    model.load_state_dict(torch.load(str(weight_path)))
    bn0 = model.backbone.bn0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        weight=bn0.weight.detach().cpu().numpy().astype(np.float32),
        bias=bn0.bias.detach().cpu().numpy().astype(np.float32),
        running_mean=bn0.running_mean.detach().cpu().numpy().astype(np.float32),
        running_var=bn0.running_var.detach().cpu().numpy().astype(np.float32),
        eps=np.float32(bn0.eps),
    )
    print(f"wrote {out}  mel_bins={bn0.num_features}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
