"""Acoustic Comfort Rating (A–E) from day/night levels and noisy-moment share."""

from __future__ import annotations

from typing import Any

from core.recording_overview import percentile_nearest
from core.report.segment import AcousticChunk, energetic_leq

NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6}

NIGHT_CAP = 40.0
DAY_CAP = 35.0
VOLATILITY_CAP = 25.0

NIGHT_REF_DB = 40.0
NIGHT_PER_DB = 2.0
DAY_REF_DB = 45.0
DAY_PER_DB = 1.5
NOISY_REF_PCT = 10.0
NOISY_PER_PCT = 1.0

GRADE_BANDS: tuple[tuple[int, str, str], ...] = (
    (85, "A", "Tranquil"),
    (70, "B", "Balanced"),
    (55, "C", "Active"),
    (40, "D", "Vibrant"),
    (0, "E", "Intense"),
)


def _l50(levels_db: list[float]) -> float | None:
    if not levels_db:
        return None
    sorted_levels = sorted(float(v) for v in levels_db)
    value = percentile_nearest(sorted_levels, 50)
    return round(value, 1) if value is not None else None


def grade_from_score(score: int) -> tuple[str, str]:
    for threshold, grade, label in GRADE_BANDS:
        if score >= threshold:
            return grade, label
    return "E", "Intense"


def night_penalty(l_night_db: float | None) -> float | None:
    if l_night_db is None:
        return None
    return min(NIGHT_CAP, max(0.0, (float(l_night_db) - NIGHT_REF_DB) * NIGHT_PER_DB))


def day_penalty(l50_day_db: float | None) -> float | None:
    if l50_day_db is None:
        return None
    return min(DAY_CAP, max(0.0, (float(l50_day_db) - DAY_REF_DB) * DAY_PER_DB))


def volatility_penalty(noisy_pct: float | None) -> float | None:
    if noisy_pct is None:
        return None
    return min(VOLATILITY_CAP, max(0.0, (float(noisy_pct) - NOISY_REF_PCT) * NOISY_PER_PCT))


def _coverage_warning(*, has_day: bool, has_night: bool) -> str | None:
    if has_day and has_night:
        return None
    if has_day and not has_night:
        return "Rated from daytime data only — nighttime wasn’t in this selection."
    if has_night and not has_day:
        return "Rated from nighttime data only — daytime wasn’t in this selection."
    return "Not enough data to rate acoustic comfort for this selection."


def build_comfort_rating(chunks: list[AcousticChunk]) -> dict[str, Any] | None:
    """Compute A–E comfort rating for the selected chunks, or None if empty."""
    if not chunks:
        return None

    night_chunks = [c for c in chunks if c.dt_local.hour in NIGHT_HOURS]
    day_chunks = [c for c in chunks if c.dt_local.hour not in NIGHT_HOURS]
    has_night = bool(night_chunks)
    has_day = bool(day_chunks)

    l_night_db = energetic_leq([c.dba for c in night_chunks]) if has_night else None
    l50_day_db = _l50([c.dba for c in day_chunks]) if has_day else None

    ungated = sum(1 for c in chunks if not c.gated)
    noisy_pct = round(100.0 * ungated / len(chunks), 1)

    raw_night = night_penalty(l_night_db)
    raw_day = day_penalty(l50_day_db)
    raw_vol = volatility_penalty(noisy_pct)

    active: list[tuple[str, float, float]] = []
    if raw_night is not None:
        active.append(("night", raw_night, NIGHT_CAP))
    if raw_day is not None:
        active.append(("day", raw_day, DAY_CAP))
    if raw_vol is not None:
        active.append(("volatility", raw_vol, VOLATILITY_CAP))

    caps = sum(cap for _, _, cap in active)
    scale = (100.0 / caps) if caps > 0 else 1.0

    scaled = {name: 0.0 for name in ("night", "day", "volatility")}
    for name, raw, _cap in active:
        scaled[name] = round(raw * scale, 2)

    total_penalty = sum(scaled.values())
    score = int(round(max(0.0, min(100.0, 100.0 - total_penalty))))
    grade, label = grade_from_score(score)
    insufficient = not (has_day and has_night)

    return {
        "score": score,
        "grade": grade,
        "label": label,
        "insufficient_data": insufficient,
        "warning": _coverage_warning(has_day=has_day, has_night=has_night),
        "coverage": {"day": has_day, "night": has_night},
        "inputs": {
            "l_night_db": l_night_db,
            "l50_day_db": l50_day_db,
            "noisy_pct": noisy_pct,
        },
        "penalties": {
            "night": scaled["night"],
            "day": scaled["day"],
            "volatility": scaled["volatility"],
            "scale": round(scale, 3),
            "raw": {
                "night": round(raw_night, 2) if raw_night is not None else None,
                "day": round(raw_day, 2) if raw_day is not None else None,
                "volatility": round(raw_vol, 2) if raw_vol is not None else None,
            },
        },
    }
