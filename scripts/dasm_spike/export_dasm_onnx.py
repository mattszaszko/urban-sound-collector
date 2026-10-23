#!/usr/bin/env python3
"""Export DASM (text-query) to ONNX for the Phase 0 Pi spike.

Run on the Pi from the Transformer4SED tree (not this product repo):

  cd ~/dasm-spike/Transformer4SED
  source .venv/bin/activate
  python /path/to/export_dasm_onnx.py

What this exports
  - DASM audio+decoder path only (mel + precomputed query vectors in, strong/weak out)
  - NOT MGA-CLAP / BERT (those stay offline for query-vector baking)

Outputs (default ./onnx_export/)
  - dasm_text_fp32.onnx
  - dasm_text_int8.onnx   (dynamic quant, if --quantize)
  - smoke_queries.pt      (example query pack used for the ORT check)
"""

from __future__ import annotations

import argparse
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
    """Checkpoints were saved on CUDA; Pi is CPU-only."""
    orig = torch.load

    def _load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]


def load_yaml_with_relative_ref(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        main_content = yaml.safe_load(f)
    if isinstance(main_content, dict) and "include" in main_content:
        included_content = main_content.pop("include")
        key_list = included_content["keys"]
        base_dict = load_yaml_with_relative_ref(included_content["base_path"])
        for key in key_list:
            main_content[key] = base_dict[key]
    return main_content


class DASMOnnxWrapper(nn.Module):
    """ONNX-friendly wrapper around DASM_HTSAT.

    Inputs
      mel:       float32 [B, ...]  (whatever backbone.wav2mel produced)
      query:     float32 [Nq, 1024]  (MGA-CLAP text embeds, already normalized)
      tgt_mask:  bool    [Nq, Nq]    (attention mask from the notebook helper)

    Outputs
      strong: float32 [B, Nq, T]
      weak:   float32 [B, Nq]
    """

    def __init__(self, dasm: nn.Module, temp_w: float = 0.5):
        super().__init__()
        self.dasm = dasm
        self.temp_w = float(temp_w)

    def forward(self, mel: torch.Tensor, query: torch.Tensor, tgt_mask: torch.Tensor):
        # Inlined from DASM_HTSAT.forward, with pad_mask omitted (apply in numpy later).
        htast_feat = self.dasm.backbone(mel.clone())["fine_grained_embedding"]
        x = self.dasm.interpolate_module(htast_feat, self.dasm.backbone_upsample_ratio)

        if hasattr(self.dasm, "cnn"):
            cnn_feat = self.dasm.cnn(mel)
            cnn_feat = F.interpolate(
                cnn_feat.squeeze(-1),
                size=x.shape[1],
                mode=self.dasm.interpolate_module.mode,
            ).transpose(1, 2)
            x = self.dasm.transformer_projector(x) + self.dasm.merge_weight * self.dasm.cnn_projector(
                cnn_feat
            )
        else:
            x = self.dasm.transformer_projector(x)
        x = self.dasm.norm_after_merge(x)

        at_feat = self.dasm.at_projector(htast_feat)
        at_out, mask_feat = self.dasm.at_branch(
            at_feat, query, query_type="text", tgt_mask=tgt_mask
        )

        x = self.dasm.sed_branch(x)
        x = self.dasm.sed_head(x)
        mask_embedding = self.dasm.mask_embedding_layer(mask_feat)
        x = torch.einsum("bqc, bct -> bqt", mask_embedding, x.transpose(1, 2)).transpose(1, 2)
        sed_out = torch.sigmoid(x / self.temp_w) * at_out.unsqueeze(1)
        sed_out = torch.clamp(sed_out, 1e-7, 1.0)
        weak_out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        weak_out = torch.clamp(weak_out, 1e-7, 1.0)
        strong = sed_out.transpose(1, 2)  # [B, Nq, T]
        return strong, weak_out


def get_att_mask(query_len: int, base_len: int) -> torch.Tensor:
    att_mask = torch.ones(query_len, query_len, dtype=torch.bool)
    att_mask[:, :base_len] = False
    att_mask.fill_diagonal_(False)
    return att_mask


def build_example_io(dasm, configs, example_wav: Path, n_extra: int = 16):
    from src.preprocess.feats_extraction import waveform_modification
    from src.codec.encoder import Encoder

    encoder = Encoder(
        [],
        audio_len=configs["feature"]["audio_max_len"],
        frame_len=configs["feature"]["win_length"],
        frame_hop=configs["feature"]["hopsize"],
        net_pooling=configs["feature"]["net_subsample"],
        sr=configs["feature"]["sr"],
    )
    wav, _pad = waveform_modification(
        str(example_wav), encoder.audio_len * encoder.sr, encoder
    )
    wav = wav.unsqueeze(0)
    mel = dasm.get_feature_extractor()(wav)

    # Base learnable/text queries already on the model (407 for this checkpoint).
    if isinstance(dasm.at_query, nn.ParameterList):
        base = dasm.at_query[0].detach()
    else:
        base = dasm.at_query.detach()
    base_len = base.shape[0]

    # Dummy extra open-vocab queries (real ones come from offline MGA-CLAP later).
    extra = F.normalize(torch.randn(n_extra, base.shape[-1]), dim=-1)
    query = torch.cat([base, extra], dim=0)
    tgt_mask = get_att_mask(query.shape[0], base_len)
    return mel, query, tgt_mask, base_len


def export_onnx(wrapper, mel, query, tgt_mask, out_path: Path, opset: int) -> None:
    wrapper.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer legacy exporter; dynamo path is flaky on exotic SED graphs.
    kwargs = dict(
        input_names=["mel", "query", "tgt_mask"],
        output_names=["strong", "weak"],
        dynamic_axes={
            "mel": {0: "batch"},
            "query": {0: "n_queries"},
            "tgt_mask": {0: "n_queries", 1: "n_queries"},
            "strong": {0: "batch", 1: "n_queries", 2: "time"},
            "weak": {0: "batch", 1: "n_queries"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(
            wrapper,
            (mel, query, tgt_mask),
            str(out_path),
            dynamo=False,
            **kwargs,
        )
    except TypeError:
        # Older torch without dynamo= kw
        torch.onnx.export(wrapper, (mel, query, tgt_mask), str(out_path), **kwargs)


def ort_check(onnx_path: Path, mel, query, tgt_mask, torch_strong, torch_weak) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {
        "mel": mel.detach().cpu().numpy(),
        "query": query.detach().cpu().numpy(),
        "tgt_mask": tgt_mask.detach().cpu().numpy(),
    }
    t0 = time.perf_counter()
    strong_o, weak_o = sess.run(None, feeds)
    dt = time.perf_counter() - t0
    print(f"ORT latency: {dt:.2f}s  strong={strong_o.shape} weak={weak_o.shape}")
    print(
        "max|strong torch-ort|="
        f"{np.max(np.abs(torch_strong.detach().cpu().numpy() - strong_o)):.5f}  "
        f"max|weak|={np.max(np.abs(torch_weak.detach().cpu().numpy() - weak_o)):.5f}"
    )


def quantize(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("T4SED_ROOT", os.getcwd()),
        help="Transformer4SED root (default: cwd)",
    )
    parser.add_argument(
        "--out-dir",
        default="onnx_export",
        help="Output directory under --root",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true", help="Also write dynamic INT8 ONNX")
    parser.add_argument("--skip-ort", action="store_true")
    parser.add_argument(
        "--n-extra",
        type=int,
        default=16,
        help=(
            "Open-vocab query slots to freeze into ONNX (product pads to this; "
            "must match core.dasm_onnx.OPEN_VOCAB_SLOTS, default 16)"
        ),
    )
    parser.add_argument(
        "--example-wav",
        default="docs/DASM/examples/Y69AIBPnJN5E.wav",
    )
    args = parser.parse_args()
    if args.n_extra < 1:
        print("--n-extra must be >= 1", file=sys.stderr)
        return 1

    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    _force_torch_load_cpu()

    from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT

    weight_path = root / "pretrained_model/detect_any_sound/text_query/as_full_text_query_best_model.pt"
    config_path = root / "pretrained_model/detect_any_sound/text_query/config.yaml"
    example_wav = root / args.example_wav
    out_dir = root / args.out_dir

    if not weight_path.is_file():
        print(f"missing weights: {weight_path}", file=sys.stderr)
        return 1
    if not example_wav.is_file():
        print(f"missing example wav: {example_wav}", file=sys.stderr)
        return 1

    print("loading config + model (CPU)…")
    configs = load_yaml_with_relative_ref(str(config_path))
    dasm = DASM_HTSAT(**configs["DASM_HTSAT"]["init_kwargs"])
    dasm.load_state_dict(torch.load(str(weight_path)))
    dasm.eval()

    temp_w = float(configs["DASM_HTSAT"].get("val_kwargs", {}).get("temp_w", 0.5))
    wrapper = DASMOnnxWrapper(dasm, temp_w=temp_w).eval()

    print("building example mel/query…")
    mel, query, tgt_mask, base_len = build_example_io(
        dasm, configs, example_wav, n_extra=int(args.n_extra)
    )
    print(
        f"mel={tuple(mel.shape)} query={tuple(query.shape)} "
        f"mask={tuple(tgt_mask.shape)} base_len={base_len} n_extra={args.n_extra}"
    )

    with torch.no_grad():
        torch_strong, torch_weak = wrapper(mel, query, tgt_mask)
    print(f"torch out strong={tuple(torch_strong.shape)} weak={tuple(torch_weak.shape)}")

    fp32_path = out_dir / "dasm_text_fp32.onnx"
    print(f"exporting ONNX → {fp32_path}")
    t0 = time.perf_counter()
    try:
        export_onnx(wrapper, mel, query, tgt_mask, fp32_path, args.opset)
    except Exception as exc:
        print(f"EXPORT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "If this is an unsupported op / dict output, paste the full traceback — "
            "next step is a split-graph export.",
            file=sys.stderr,
        )
        return 2
    print(f"export wall time: {time.perf_counter() - t0:.1f}s  size={fp32_path.stat().st_size / 1e6:.1f} MB")

    torch.save(
        {"query": query, "tgt_mask": tgt_mask, "base_len": base_len, "temp_w": temp_w},
        out_dir / "smoke_queries.pt",
    )

    if not args.skip_ort:
        print("ORT check (fp32)…")
        ort_check(fp32_path, mel, query, tgt_mask, torch_strong, torch_weak)

    if args.quantize:
        int8_path = out_dir / "dasm_text_int8.onnx"
        print(f"dynamic quantize → {int8_path}")
        quantize(fp32_path, int8_path)
        print(f"int8 size={int8_path.stat().st_size / 1e6:.1f} MB")
        if not args.skip_ort:
            print("ORT check (int8)…")
            ort_check(int8_path, mel, query, tgt_mask, torch_strong, torch_weak)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
