"""Map resolved YAMNet labels to dashboard macro-categories."""

from __future__ import annotations

from typing import Any

from core.report.votes import (
    MACRO_UNCLASSIFIED,
    load_display_theme_map,
    load_label_map,
)


def label_to_macro(
    label: str | None,
    *,
    label_map: dict[str, Any] | None = None,
    theme_by_label: dict[str, str] | None = None,
) -> str:
    if not label:
        return MACRO_UNCLASSIFIED
    cfg = label_map or load_label_map()
    overrides = cfg.get("label_overrides") or {}
    if label in overrides:
        return str(overrides[label])

    themes = theme_by_label if theme_by_label is not None else load_display_theme_map()
    theme = themes.get(label)
    theme_map = cfg.get("theme_to_macro") or {}
    if theme and theme in theme_map:
        return str(theme_map[theme])
    return MACRO_UNCLASSIFIED
