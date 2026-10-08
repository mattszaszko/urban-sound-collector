"""Natural-language soundscape briefing for the noise report dashboard."""

from __future__ import annotations

from typing import Any

RELATIVE_SILENCE = "Relative silence"


def _fmt_db(value: float | None, *, digits: int = 0) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.{digits}f} dBA"
    except (TypeError, ValueError):
        return None


def _peak_day_band(hourly: list[dict[str, Any]]) -> str | None:
    day_rows = [
        h for h in hourly if h.get("period") == "day" and h.get("leq_db") is not None
    ]
    if not day_rows:
        return None
    best = max(day_rows, key=lambda h: float(h["leq_db"]))
    peak_hour = int(best["hour"])
    by_hour = {int(h["hour"]): float(h["leq_db"]) for h in day_rows}
    start = peak_hour
    end = peak_hour
    thr = float(best["leq_db"]) - 3.0
    while True:
        prev = start - 1
        if prev not in by_hour or by_hour[prev] < thr:
            break
        start = prev
    while True:
        nxt = end + 1
        if nxt not in by_hour or by_hour[nxt] < thr:
            break
        end = nxt
    if start == end:
        return f"around {start:02d}:00"
    return f"between {start:02d}:00 and {end:02d}:00"


def _night_floor_db(hourly: list[dict[str, Any]]) -> float | None:
    night_rows = [
        h
        for h in hourly
        if h.get("period") == "night" and h.get("leq_db") is not None
    ]
    if not night_rows:
        return None
    return min(float(h["leq_db"]) for h in night_rows)


def _top_active_macros(
    time_budget: dict[str, Any],
    *,
    limit: int = 2,
) -> list[tuple[str, float]]:
    """Leading non-silence macros from the whole-selection time budget."""
    rows = list(time_budget.get("total") or [])
    active = [
        (str(r.get("category") or ""), float(r.get("pct") or 0.0))
        for r in rows
        if r.get("category") and r.get("category") != RELATIVE_SILENCE
    ]
    active = [(name, pct) for name, pct in active if pct > 0]
    active.sort(key=lambda item: item[1], reverse=True)
    return active[:limit]


def _silence_pct(time_budget: dict[str, Any]) -> float | None:
    for row in time_budget.get("total") or []:
        if row.get("category") == RELATIVE_SILENCE:
            try:
                return float(row.get("pct"))
            except (TypeError, ValueError):
                return None
    return None


def _human_effect(
    *,
    l50_db: float | None,
    night_floor: float | None,
    comfort_label: str | None,
) -> str:
    """Plain-language living impact (not medical advice)."""
    bits: list[str] = []
    if night_floor is not None:
        if night_floor >= 55:
            bits.append(
                f"Nighttime levels near {_fmt_db(night_floor)} stay elevated enough "
                "that sleep can be harder to protect with windows open"
            )
        elif night_floor >= 45:
            bits.append(
                f"Nights settle near {_fmt_db(night_floor)} — often acceptable indoors "
                "with a closed window, but still noticeable outdoors"
            )
        else:
            bits.append(
                f"Nights are comparatively quiet (about {_fmt_db(night_floor)}), "
                "which supports rest when people are at home"
            )
    if l50_db is not None:
        if l50_db >= 70:
            bits.append(
                "typical daytime levels are loud enough to compete with conversation "
                "and make outdoor speech feel effortful"
            )
        elif l50_db >= 55:
            bits.append(
                "daytime levels sit in a busy urban range where outdoor talk remains "
                "possible but background noise is always present"
            )
        elif l50_db >= 40:
            bits.append(
                "most of the day stays near a conversational backdrop — usable "
                "outdoors, with occasional louder moments"
            )
        else:
            bits.append(
                "overall levels stay low, closer to a quiet indoor room than a street"
            )
    if not bits:
        if comfort_label:
            return (
                f"Overall the site rates as {comfort_label.lower()} on the acoustic "
                "comfort scale for this selection."
            )
        return "There is not enough day/night coverage to describe how people would experience this place."
    if len(bits) == 1:
        return bits[0][0].upper() + bits[0][1:] + "."
    return bits[0][0].upper() + bits[0][1:] + ", and " + bits[1] + "."


def build_takeaway(
    *,
    zone_a: dict[str, Any] | None = None,
    zone_b: dict[str, Any],
    zone_c: dict[str, Any],
) -> str:
    """Build a multi-sentence soundscape briefing from report zones (deterministic)."""
    zone_a = zone_a or {}
    hourly = zone_b.get("typical_day") or zone_b.get("hourly") or []
    budget = zone_c.get("time_budget") or {}
    comfort = zone_a.get("comfort_rating") or {}

    l50 = zone_a.get("l50_db")
    l90 = zone_a.get("l90_db")
    l10 = zone_a.get("l10_db")
    context = zone_a.get("leq_context")
    grade = comfort.get("grade")
    label = comfort.get("label")
    peak_band = _peak_day_band(hourly)
    night_floor = _night_floor_db(hourly)
    macros = _top_active_macros(budget)
    silence = _silence_pct(budget)
    tops = list(zone_a.get("top_disturbances") or [])

    paragraphs: list[str] = []

    # 1 — Character of the place
    char_bits: list[str] = []
    if grade and label:
        char_bits.append(
            f"This selection rates acoustic comfort {grade} ({label.lower()})"
        )
    if l50 is not None:
        anchor = f" — like a {context.lower()}" if context and context != "—" else ""
        char_bits.append(
            f"with a typical level around {_fmt_db(l50, digits=1)}{anchor}"
        )
    if l90 is not None and l10 is not None:
        char_bits.append(
            f"Quiet stretches sit near {_fmt_db(l90, digits=1)}, while louder moments "
            f"reach about {_fmt_db(l10, digits=1)}"
        )
    if char_bits:
        if len(char_bits) == 1:
            paragraphs.append(char_bits[0] + ".")
        elif len(char_bits) == 2:
            paragraphs.append(f"{char_bits[0]}, {char_bits[1]}.")
        else:
            paragraphs.append(
                f"{char_bits[0]}, {char_bits[1]}. {char_bits[2]}."
            )

    # 2 — Rhythm + what dominates (time budget, aligned with the pies)
    rhythm_bits: list[str] = []
    if peak_band:
        rhythm_bits.append(f"Levels tend to peak {peak_band} on a typical day")
    if macros:
        lead_name, lead_pct = macros[0]
        if len(macros) > 1 and macros[1][1] >= 8.0:
            second_name, second_pct = macros[1]
            rhythm_bits.append(
                f"Among classified moments, {lead_name} accounts for about "
                f"{lead_pct:.0f}% of the time budget, with {second_name} next "
                f"({second_pct:.0f}%)"
            )
        else:
            rhythm_bits.append(
                f"Among classified moments, {lead_name} dominates "
                f"(about {lead_pct:.0f}% of the time budget)"
            )
    if silence is not None and silence >= 20.0:
        rhythm_bits.append(
            f"Relative silence still covers roughly {silence:.0f}% of the selection"
        )
    if rhythm_bits:
        if len(rhythm_bits) == 1:
            paragraphs.append(rhythm_bits[0] + ".")
        else:
            paragraphs.append(
                rhythm_bits[0] + ". " + ". ".join(rhythm_bits[1:]) + "."
            )

    # 3 — Human effect
    paragraphs.append(
        _human_effect(
            l50_db=float(l50) if l50 is not None else None,
            night_floor=night_floor,
            comfort_label=str(label) if label else None,
        )
    )

    # 4 — Standout peaks (optional)
    if tops:
        top = tops[0]
        peak_db = top.get("lafmax_db")
        cat = top.get("category") or top.get("label")
        when = top.get("at_local")
        if peak_db is not None and cat:
            peak_line = (
                f"The loudest recorded moment reached {_fmt_db(float(peak_db), digits=1)} "
                f"({cat}"
            )
            if when:
                peak_line += f" at {when}"
            peak_line += "). Use Inspect on Top 3 disturbances to hear that moment in context."
            paragraphs.append(peak_line)

    if comfort.get("insufficient_data") and comfort.get("warning"):
        paragraphs.append(str(comfort["warning"]))

    return "\n\n".join(p for p in paragraphs if p)
