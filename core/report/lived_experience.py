"""Lived Experience & Health Impact card — exterior levels + Dutch glazing heuristics.

Lden is computed for scoring (home-office / annoyance framing) but must not be
shown in the report UI. Sleep badges use outdoor night Leq. Indoor estimates
apply fixed open/closed façade attenuation (not measured indoor levels).
"""

from __future__ import annotations

import math
from typing import Any

from core.report.segment import AcousticChunk, energetic_leq

# EU Environmental Noise Directive period hours (local).
LDEN_DAY_HOURS = set(range(7, 19))       # 07–18
LDEN_EVENING_HOURS = set(range(19, 23))  # 19–22
LDEN_NIGHT_HOURS = {23, 0, 1, 2, 3, 4, 5, 6}

# Dutch double-glazing / ventilation heuristics (dB).
ATTEN_CLOSED_DB = 30.0
ATTEN_OPEN_DB = 15.0
INDOOR_FLOOR_DB = 20.0

# Indoor dBA badge thresholds (WHO-oriented heuristics).
SLEEP_OK_CLOSED = 30.0
SLEEP_OK_OPEN = 40.0
SLEEP_ALERT_OPEN = 45.0

OFFICE_OK_CLOSED = 35.0
OFFICE_WATCH_CLOSED = 45.0
OFFICE_ALERT_CLOSED = 50.0

BABY_OK_CLOSED = 30.0
BABY_WATCH_OPEN = 40.0
BABY_PEAK_ALERT_DB = 70.0  # outdoor LAFmax spike concern


def indoor_from_exterior(exterior_db: float | None, atten_db: float) -> float | None:
    if exterior_db is None:
        return None
    return round(max(INDOOR_FLOOR_DB, float(exterior_db) - float(atten_db)), 1)


def _period_leq(chunks: list[AcousticChunk], hours: set[int]) -> float | None:
    levels = [c.dba for c in chunks if c.dt_local.hour in hours]
    return energetic_leq(levels)


def compute_lden_db(chunks: list[AcousticChunk]) -> dict[str, Any]:
    """EU-style Lden from day / evening / night energetic means.

    Missing periods are omitted and duration weights renormalized.
    """
    day = _period_leq(chunks, LDEN_DAY_HOURS)
    evening = _period_leq(chunks, LDEN_EVENING_HOURS)
    night = _period_leq(chunks, LDEN_NIGHT_HOURS)

    # Standard durations (hours) when present.
    terms: list[tuple[float, float, float]] = []  # (hours, level_db, penalty_db)
    if day is not None:
        terms.append((12.0, day, 0.0))
    if evening is not None:
        terms.append((4.0, evening, 5.0))
    if night is not None:
        terms.append((8.0, night, 10.0))

    lden: float | None = None
    if terms:
        total_h = sum(h for h, _, _ in terms)
        if total_h > 0:
            acc = 0.0
            for hours, level, penalty in terms:
                acc += hours * (10.0 ** ((level + penalty) / 10.0))
            lden = round(10.0 * math.log10(acc / total_h), 1)

    return {
        "lden_db": lden,
        "lday_db": day,
        "levening_db": evening,
        "lnight_db": night,
        "has_day": day is not None,
        "has_evening": evening is not None,
        "has_night": night is not None,
    }


def _badge(label: str, tone: str) -> dict[str, str]:
    return {"label": label, "tone": tone}


def _sleep_dimension(
    *,
    lnight: float | None,
    indoor_closed: float | None,
    indoor_open: float | None,
) -> dict[str, Any]:
    if lnight is None or indoor_closed is None or indoor_open is None:
        return {
            "id": "sleep",
            "title": "Sleep Quality & Rest",
            "badge": _badge("Night data needed", "watch"),
            "summary": (
                "This selection doesn’t include enough nighttime audio to judge "
                "sleep. Add evening-to-morning recordings for a clearer picture."
            ),
            "exterior_db": lnight,
            "indoor_closed_db": indoor_closed,
            "indoor_open_db": indoor_open,
        }

    if indoor_closed <= SLEEP_OK_CLOSED and indoor_open <= SLEEP_OK_OPEN:
        badge = _badge("Well protected", "ok")
        summary = (
            "With windows closed, traffic should fade to a faint, distant hum that "
            "is unlikely to disrupt deep sleep. Even with windows open for fresh air, "
            "nighttime levels look manageable for most sleepers."
        )
    elif indoor_closed <= SLEEP_OK_CLOSED and indoor_open <= SLEEP_ALERT_OPEN:
        badge = _badge("Protected if windows closed", "watch")
        summary = (
            "With windows closed, traffic fades to a faint, distant hum that shouldn’t "
            "disrupt deep sleep. With windows open for fresh air, passing vehicles and "
            "nighttime spikes can break through and interrupt sleep cycles."
        )
    elif indoor_closed <= SLEEP_OK_OPEN:
        badge = _badge("Easily disrupted when open", "watch")
        summary = (
            "Closed double-glazing still keeps most night noise at bay, but open "
            "windows let urban traffic into the bedroom. Plan on closing windows for "
            "uninterrupted rest, especially on busier nights."
        )
    else:
        badge = _badge("High sleep disruption risk", "alert")
        summary = (
            "Nighttime levels outside are high enough that even closed windows may "
            "not fully protect sleep. Open windows make awakenings from traffic and "
            "spikes much more likely."
        )

    return {
        "id": "sleep",
        "title": "Sleep Quality & Rest",
        "badge": badge,
        "summary": summary,
        "exterior_db": lnight,
        "indoor_closed_db": indoor_closed,
        "indoor_open_db": indoor_open,
    }


def _office_dimension(
    *,
    lden: float | None,
    indoor_closed: float | None,
    indoor_open: float | None,
) -> dict[str, Any]:
    if lden is None or indoor_closed is None or indoor_open is None:
        return {
            "id": "office",
            "title": "Home Office Suitability",
            "badge": _badge("Daytime data needed", "watch"),
            "summary": (
                "There isn’t enough daytime coverage to judge focus and calls. "
                "Include daytime hours for a clearer home-office picture."
            ),
            "exterior_db": lden,
            "indoor_closed_db": indoor_closed,
            "indoor_open_db": indoor_open,
        }

    if indoor_closed <= OFFICE_OK_CLOSED and indoor_open <= OFFICE_WATCH_CLOSED:
        badge = _badge("Good with windows closed", "ok")
        summary = (
            "During working hours, closed double-glazing provides a quiet backdrop "
            "ideal for video calls and deep focus. Opening windows brings in urban "
            "traffic that you may want to close again during meetings."
        )
    elif indoor_closed <= OFFICE_WATCH_CLOSED:
        badge = _badge("Calls need windows closed", "watch")
        summary = (
            "Closed glazing keeps concentration workable for most tasks. Opening "
            "windows brings in enough street noise that calls and deep focus will "
            "usually need them shut."
        )
    elif indoor_closed <= OFFICE_ALERT_CLOSED:
        badge = _badge("Distracting with windows open", "watch")
        summary = (
            "Even with windows closed, the outdoor day is lively enough to notice. "
            "Open windows make focus and video calls clearly harder — keep them "
            "closed for meetings."
        )
    else:
        badge = _badge("Distracting even closed", "alert")
        summary = (
            "Outdoor levels stay high through the day. Closed windows help, but "
            "expect a noticeable backdrop that can still compete with calls and "
            "concentration."
        )

    return {
        "id": "office",
        "title": "Home Office Suitability",
        "badge": badge,
        "summary": summary,
        "exterior_db": lden,
        "indoor_closed_db": indoor_closed,
        "indoor_open_db": indoor_open,
    }


def _baby_dimension(
    *,
    lnight: float | None,
    lden: float | None,
    night_closed: float | None,
    night_open: float | None,
    day_open: float | None,
    peak_lafmax: float | None,
) -> dict[str, Any]:
    if night_closed is None and day_open is None:
        return {
            "id": "baby",
            "title": "Baby Rest Experience",
            "badge": _badge("More data needed", "watch"),
            "summary": (
                "There isn’t enough day or night coverage yet to judge nap and "
                "bedtime sensitivity for infants."
            ),
            "exterior_db": lnight if lnight is not None else lden,
            "indoor_closed_db": night_closed,
            "indoor_open_db": night_open if night_open is not None else day_open,
        }

    spike = peak_lafmax is not None and float(peak_lafmax) >= BABY_PEAK_ALERT_DB
    open_level = night_open if night_open is not None else day_open
    closed_ok = night_closed is not None and night_closed <= BABY_OK_CLOSED
    open_sensitive = open_level is not None and open_level > BABY_WATCH_OPEN

    if spike or (open_sensitive and not closed_ok):
        badge = _badge("Sensitive Environment Alert", "alert")
        summary = (
            "Babies are particularly sensitive to sudden spikes (sirens, heavy trucks). "
            "Insulated glazing helps daytime naps, but open windows let road noise "
            "through that can startle light sleepers — and the loudest peaks here "
            "are sharp enough to notice indoors."
            if spike
            else (
                "Babies are particularly sensitive to sudden transient spikes. "
                "With insulated glazing, naps are better protected, but open windows "
                "let road noise through that can startle light sleepers."
            )
        )
    elif open_sensitive and closed_ok:
        badge = _badge("Sensitive when windows open", "watch")
        summary = (
            "With insulated glazing, daytime naps and bedtime are well protected. "
            "If windows are open, sudden road noise risks startling light sleepers — "
            "keep them closed during naps when you can."
        )
    else:
        badge = _badge("Protected naps (windows closed)", "ok")
        summary = (
            "With insulated glazing, daytime naps look well protected from typical "
            "street noise. Open windows still raise the chance of brief startles from "
            "passing vehicles, so closed is safer for light sleepers."
        )

    return {
        "id": "baby",
        "title": "Baby Rest Experience",
        "badge": badge,
        "summary": summary,
        "exterior_db": lnight if lnight is not None else lden,
        "indoor_closed_db": night_closed,
        "indoor_open_db": open_level,
    }


def build_lived_experience(
    chunks: list[AcousticChunk],
    *,
    top_disturbances: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build Lived Experience payload for zone_a, or None if no chunks."""
    if not chunks:
        return None

    metrics = compute_lden_db(chunks)
    lden = metrics["lden_db"]
    lnight = metrics["lnight_db"]

    sleep_closed = indoor_from_exterior(lnight, ATTEN_CLOSED_DB)
    sleep_open = indoor_from_exterior(lnight, ATTEN_OPEN_DB)
    office_closed = indoor_from_exterior(lden, ATTEN_CLOSED_DB)
    office_open = indoor_from_exterior(lden, ATTEN_OPEN_DB)
    day_open = indoor_from_exterior(metrics["lday_db"], ATTEN_OPEN_DB)

    peak = None
    if top_disturbances:
        try:
            peak = float(top_disturbances[0].get("lafmax_db"))
        except (TypeError, ValueError, IndexError, AttributeError):
            peak = None

    dimensions = [
        _sleep_dimension(
            lnight=lnight,
            indoor_closed=sleep_closed,
            indoor_open=sleep_open,
        ),
        _office_dimension(
            lden=lden,
            indoor_closed=office_closed,
            indoor_open=office_open,
        ),
        _baby_dimension(
            lnight=lnight,
            lden=lden,
            night_closed=sleep_closed,
            night_open=sleep_open,
            day_open=day_open,
            peak_lafmax=peak,
        ),
    ]

    insufficient = not (
        metrics["has_night"] and (metrics["has_day"] or metrics["has_evening"])
    )
    warning = None
    if not metrics["has_night"]:
        warning = (
            "Sleep judgment is limited — little or no nighttime audio in this selection."
        )
    elif not metrics["has_day"] and not metrics["has_evening"]:
        warning = (
            "Daytime judgment is limited — little daytime audio in this selection."
        )

    return {
        "title": "Lived Experience & Health Impact",
        "intro": (
            "How this outdoor soundscape may feel indoors with typical Dutch "
            "double glazing — windows closed versus open for ventilation. "
            "These are estimates from outdoor measurements, not indoor readings."
        ),
        "attenuation": {
            "closed_db": ATTEN_CLOSED_DB,
            "open_db": ATTEN_OPEN_DB,
        },
        "dimensions": dimensions,
        "insufficient_data": insufficient,
        "warning": warning,
        # Internal / tests — UI must not display lden_db or the word Lden.
        "metrics": {
            "lden_db": lden,
            "lnight_db": lnight,
            "lday_db": metrics["lday_db"],
            "levening_db": metrics["levening_db"],
        },
    }
