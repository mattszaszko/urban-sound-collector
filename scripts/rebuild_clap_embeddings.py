"""Rebuild CLAP text embeddings from config/clap_prompts.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.classifier_clap import rebuild_text_embeddings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=REPO_ROOT / "config" / "clap_prompts.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "config" / "clap_prompt_embeddings.npz",
    )
    args = parser.parse_args(argv)
    result = rebuild_text_embeddings(prompts_path=args.prompts, embeddings_path=args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
