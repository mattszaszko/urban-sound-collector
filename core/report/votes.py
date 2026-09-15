"""Vote eligibility and dashboard label-map loading."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.report.timeutil import DEFAULT_SITE_TIMEZONE

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_MAP_PATH = REPO_ROOT / "config" / "dashboard_label_map.json"
DEFAULT_CATALOG_PATH = REPO_ROOT / "config" / "yamnet_label_catalog.json"

MACRO_UNCLASSIFIED = "Unclassified / Ambient"


@lru_cache(maxsize=4)
def load_label_map(path: str | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_LABEL_MAP_PATH
    with target.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("dashboard_label_map.json must be an object")
    return data


@lru_cache(maxsize=2)
def load_display_theme_map(catalog_path: str | None = None) -> dict[str, str]:
    target = Path(catalog_path) if catalog_path else DEFAULT_CATALOG_PATH
    if not target.is_file():
        return {}
    with target.open(encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, str] = {}
    for row in data.get("labels") or []:
        if not isinstance(row, dict):
            continue
        name = row.get("display_name")
        theme = row.get("theme")
        if isinstance(name, str) and isinstance(theme, str):
            out[name] = theme
    return out


def clear_label_map_cache() -> None:
    load_label_map.cache_clear()
    load_display_theme_map.cache_clear()


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_yamnet_gated(event: dict) -> bool:
    prep = event.get("yamnet_preprocess")
    if isinstance(prep, dict) and bool(prep.get("gated")):
        return True
    label = event.get("top_label")
    return isinstance(label, str) and label == "gated"


def vote_label_for_event(
    event: dict,
    *,
    label_map: dict[str, Any] | None = None,
    theme_by_label: dict[str, str] | None = None,
) -> str | None:
    """Return a YAMNet label eligible for event voting, or None."""
    cfg = label_map or load_label_map()
    themes = theme_by_label if theme_by_label is not None else load_display_theme_map()
    min_conf = float(cfg.get("min_confidence", 0.25))
    blacklist = {str(x) for x in (cfg.get("blacklist_labels") or [])}
    blacklist_themes = {str(x) for x in (cfg.get("blacklist_themes") or [])}

    if is_yamnet_gated(event):
        return None
    if event.get("clap_status") == "gated":
        return None

    label = event.get("top_label")
    if not isinstance(label, str) or not label.strip() or label in {"gated", "n/a"}:
        return None

    conf = _f(event.get("top_confidence"))
    if conf is None or conf < min_conf:
        return None

    if label in blacklist:
        return None
    theme = themes.get(label)
    if theme in blacklist_themes:
        return None

    return label


__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_LABEL_MAP_PATH",
    "DEFAULT_SITE_TIMEZONE",
    "MACRO_UNCLASSIFIED",
    "clear_label_map_cache",
    "is_yamnet_gated",
    "load_display_theme_map",
    "load_label_map",
    "vote_label_for_event",
]
