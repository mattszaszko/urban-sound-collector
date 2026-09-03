"""Filter JSONL event fields for download / export."""

from __future__ import annotations

from typing import Any, Iterator
from pathlib import Path
import json


def filter_event_for_export(
    event: dict[str, Any],
    *,
    include_spectrum: bool = True,
    include_yamnet_preprocess: bool = True,
) -> dict[str, Any]:
    """Return a shallow copy of ``event`` with optional heavy fields removed."""
    filtered = dict(event)
    if not include_spectrum:
        filtered.pop("spectrum", None)
    if not include_yamnet_preprocess:
        filtered.pop("yamnet_preprocess", None)
    return filtered


def iter_filtered_jsonl(
    path: Path,
    *,
    include_spectrum: bool = True,
    include_yamnet_preprocess: bool = True,
) -> Iterator[str]:
    """Yield JSONL lines, optionally stripping spectrum / preprocess fields."""
    passthrough = include_spectrum and include_yamnet_preprocess
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if passthrough:
                yield line + "\n"
                continue
            event = json.loads(line)
            filtered = filter_event_for_export(
                event,
                include_spectrum=include_spectrum,
                include_yamnet_preprocess=include_yamnet_preprocess,
            )
            yield json.dumps(filtered, ensure_ascii=False) + "\n"


def export_filename(
    original: str,
    *,
    include_spectrum: bool,
    include_yamnet_preprocess: bool,
) -> str:
    """Build a download filename that reflects omitted fields."""
    if include_spectrum and include_yamnet_preprocess:
        return original
    stem = Path(original).stem
    suffix = Path(original).suffix or ".jsonl"
    tags: list[str] = []
    if not include_spectrum:
        tags.append("no-spectrum")
    if not include_yamnet_preprocess:
        tags.append("no-preprocess")
    return f"{stem}-{'-'.join(tags)}{suffix}"
