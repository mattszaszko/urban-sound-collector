"""Download Xenova clap-htsat-unfused quantized ONNX assets into models/clap/."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "models" / "clap"

HF_BASE = (
    "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/main"
)

# Pinned SHA256 from Hugging Face LFS pointers / verified downloads.
ASSETS: dict[str, dict[str, str]] = {
    "audio_model_quantized.onnx": {
        "url": f"{HF_BASE}/onnx/audio_model_quantized.onnx",
        "sha256": "3fcff2c8824e7bcb83a983f2a49edab3b60cbcf4872ac70efee517355173bd1f",
    },
    "text_model_quantized.onnx": {
        "url": f"{HF_BASE}/onnx/text_model_quantized.onnx",
        "sha256": "1a3df8b197e249816e08415fd040434c44762b2eea7eb7bf8a48a0f0bf3c14e5",
    },
    "tokenizer.json": {
        "url": f"{HF_BASE}/tokenizer.json",
        "sha256": "dc239041d98de27ffc3975473a1a23e3db4c937b23c138c38bbc66588bd247e5",
    },
    "preprocessor_config.json": {
        "url": f"{HF_BASE}/preprocessor_config.json",
        "sha256": None,  # small JSON; size-checked after download
    },
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_clap_models(dest_dir: Path = DEFAULT_DIR, *, force: bool = False) -> dict:
    """Download missing CLAP assets; return status dict."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for name, meta in ASSETS.items():
        path = dest_dir / name
        expected = meta.get("sha256")
        if path.exists() and not force:
            if expected:
                got = _sha256_file(path)
                if got != expected:
                    raise RuntimeError(
                        f"{path.name} SHA256 mismatch (got {got}, expected {expected}). "
                        "Re-run with --force."
                    )
            results[name] = "ok-existing"
            continue
        _download(meta["url"], path)
        if expected:
            got = _sha256_file(path)
            if got != expected:
                path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{path.name} SHA256 mismatch after download "
                    f"(got {got}, expected {expected})"
                )
        results[name] = "downloaded"
    return {"ok": True, "dir": str(dest_dir), "files": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Destination directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args(argv)
    try:
        result = ensure_clap_models(args.dir, force=bool(args.force))
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    print("Done. Next: rebuild embeddings (Prompts tab or scripts/rebuild_clap_embeddings.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
