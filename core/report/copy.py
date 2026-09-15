"""Natural-language takeaway for the noise report dashboard."""

from __future__ import annotations

from typing import Any


def build_takeaway(
    *,
    zone_b: dict[str, Any],
    zone_c: dict[str, Any],
) -> str:
    hourly = zone_b.get("hourly") or []
    day_rows = [
        h
        for h in hourly
        if h.get("period") == "day" and h.get("leq_db") is not None
    ]
    night_rows = [
        h
        for h in hourly
        if h.get("period") == "night" and h.get("leq_db") is not None
    ]

    peak_band = "daytime hours"
    if day_rows:
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
        peak_band = f"{start:02d}:00 and {end:02d}:00"

    diet = zone_c.get("sound_diet") or []
    top_macro = "mixed sources"
    if diet:
        ranked = sorted(diet, key=lambda d: float(d.get("event_seconds") or 0), reverse=True)
        if ranked and float(ranked[0].get("event_seconds") or 0) > 0:
            top_macro = str(ranked[0].get("macro") or top_macro)

    night_floor = None
    if night_rows:
        night_floor = min(float(h["leq_db"]) for h in night_rows)

    sentence1 = (
        f"Noise levels peak predictably between {peak_band} "
        f"driven primarily by {top_macro}."
    )
    if night_floor is not None:
        sentence2 = (
            f"Background levels drop to about {night_floor:.0f} dBA "
            f"during nighttime hours."
        )
    else:
        sentence2 = "Nighttime background levels could not be estimated from this selection."
    return f"{sentence1} {sentence2}"
