"""Build three-zone dashboard report payload from JSONL events."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from core.recording_overview import percentile_nearest
from core.report.copy import build_takeaway
from core.report.segment import (
    AcousticChunk,
    AcousticEvent,
    energetic_leq,
    prepare_chunks,
    segment_acoustic_events,
)
from core.report.timeutil import (
    DEFAULT_SITE_TIMEZONE,
    format_local,
    format_local_date,
    resolve_zone,
)
from core.report.votes import load_display_theme_map, load_label_map

NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6}
DAY_HOURS = set(range(7, 22))


def leq_context(leq_db: float | None) -> str:
    """Map A-weighted level to a layperson anchor (WHO-style bands)."""
    if leq_db is None:
        return "—"
    if leq_db < 40:
        return "Quiet bedroom"
    if leq_db < 55:
        return "Conversational backdrop"
    if leq_db < 70:
        return "Busy street / TV"
    return "Lawnmower / peak traffic"


def _level_percentiles(levels_db: list[float]) -> dict[str, float | None]:
    """Acoustic Ln from chunk dBA: L10=loud spikes, L50=median, L90=quiet floor."""
    if not levels_db:
        return {"l10_db": None, "l50_db": None, "l90_db": None}
    sorted_levels = sorted(float(v) for v in levels_db)
    l10 = percentile_nearest(sorted_levels, 90)
    l50 = percentile_nearest(sorted_levels, 50)
    l90 = percentile_nearest(sorted_levels, 10)
    return {
        "l10_db": round(l10, 1) if l10 is not None else None,
        "l50_db": round(l50, 1) if l50 is not None else None,
        "l90_db": round(l90, 1) if l90 is not None else None,
    }


def _header_date_span(start: str, end: str) -> str:
    """Pretty local date span for header."""
    if start == end:
        return start
    try:
        from datetime import datetime

        a = datetime.strptime(start, "%Y-%m-%d")
        b = datetime.strptime(end, "%Y-%m-%d")
        if a.year == b.year and a.month == b.month:
            return f"{a.strftime('%b')} {a.day} – {b.day}"
        if a.year == b.year:
            return f"{a.strftime('%b')} {a.day} – {b.strftime('%b')} {b.day}"
        return f"{a.strftime('%b %d, %Y')} – {b.strftime('%b %d, %Y')}"
    except ValueError:
        return f"{start} – {end}"


def sound_diet_from_events(events: list[AcousticEvent]) -> list[dict[str, Any]]:
    """Duration-weighted macro share (pct from event_seconds, never event count)."""
    seconds: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for ev in events:
        seconds[ev.macro] += float(ev.duration_s)
        counts[ev.macro] += 1
    total = sum(seconds.values())
    rows: list[dict[str, Any]] = []
    for macro, secs in sorted(seconds.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = round(100.0 * secs / total, 1) if total > 0 else 0.0
        rows.append(
            {
                "macro": macro,
                "pct": pct,
                "event_seconds": round(secs, 1),
                "events": int(counts[macro]),
            }
        )
    return rows


def build_hourly_profile(chunks: list[AcousticChunk]) -> list[dict[str, Any]]:
    """Clock-hour (typical day) profile: all chunks bucketed by local hour 0–23."""
    buckets: dict[int, list[AcousticChunk]] = {h: [] for h in range(24)}
    for c in chunks:
        buckets[c.dt_local.hour].append(c)
    out: list[dict[str, Any]] = []
    for hour in range(24):
        members = buckets[hour]
        period = "night" if hour in NIGHT_HOURS else "day"
        stats = _bucket_loudness_stats(members)
        out.append(
            {
                "hour": hour,
                "period": period,
                **stats,
            }
        )
    return out


def build_timeline_profile(chunks: list[AcousticChunk]) -> list[dict[str, Any]]:
    """Chronological hourly L_eq from first to last local hour in the selection."""
    if not chunks:
        return []

    def _hour_floor(dt: datetime) -> datetime:
        return dt.replace(minute=0, second=0, microsecond=0)

    start = _hour_floor(chunks[0].dt_local)
    end = _hour_floor(chunks[-1].dt_local)
    if end < start:
        return []

    buckets: dict[datetime, list[AcousticChunk]] = {}
    for c in chunks:
        key = _hour_floor(c.dt_local)
        buckets.setdefault(key, []).append(c)

    out: list[dict[str, Any]] = []
    cur = start
    while cur <= end:
        members = buckets.get(cur, [])
        hour = int(cur.hour)
        stats = _bucket_loudness_stats(members)
        out.append(
            {
                "t_local": cur.isoformat(timespec="seconds"),
                "label": cur.strftime("%d %b %H:%M"),
                "date_short": cur.strftime("%d %b"),
                "hour": hour,
                "period": "night" if hour in NIGHT_HOURS else "day",
                **stats,
            }
        )
        cur = cur + timedelta(hours=1)
    return out


def _bucket_loudness_stats(members: list[AcousticChunk]) -> dict[str, Any]:
    """Leq, within-hour L90/L10 spread, and gated quiet share for one hour bucket."""
    if not members:
        return {
            "leq_db": None,
            "l90_db": None,
            "l10_db": None,
            "gated_pct": None,
            "chunk_count": 0,
            "gated_count": 0,
        }
    dbas = [c.dba for c in members]
    percentiles = _level_percentiles(dbas)
    gated_n = sum(1 for c in members if c.gated)
    n = len(members)
    return {
        "leq_db": energetic_leq(dbas),
        "l90_db": percentiles["l90_db"],
        "l10_db": percentiles["l10_db"],
        "gated_pct": round(100.0 * gated_n / n, 1),
        "chunk_count": n,
        "gated_count": gated_n,
    }


def build_dashboard_report(
    recording_events: list[tuple[str, list[dict]]],
    *,
    timezone_name: str | None = None,
    site_label: str | None = None,
    label_map_path: str | None = None,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """
    Build Zone A/B/C payload.

    ``recording_events`` is a list of ``(filename, events)`` from selected JSONL files.
    """
    zone, tz_name, tz_warning = resolve_zone(timezone_name or DEFAULT_SITE_TIMEZONE)
    label_map = load_label_map(label_map_path)
    if min_confidence is not None:
        label_map = {**label_map, "min_confidence": float(min_confidence)}
    themes = load_display_theme_map()

    all_chunks: list[AcousticChunk] = []
    recording_summaries: list[dict[str, Any]] = []
    for name, events in recording_events:
        prepared = prepare_chunks(
            events,
            zone=zone,
            label_map=label_map,
            theme_by_label=themes,
            source_file=name,
        )
        all_chunks.extend(prepared)
        recording_summaries.append({"name": name, "chunks": len(prepared)})

    all_chunks.sort(key=lambda c: c.dt_utc)
    acoustic_events = segment_acoustic_events(
        all_chunks, label_map=label_map, theme_by_label=themes
    )

    # Zone A
    dbas = [c.dba for c in all_chunks]
    leq = energetic_leq(dbas)
    percentiles = _level_percentiles(dbas)
    peak_laf = None
    peak_at_local = None
    peak_at_utc = None
    for c in all_chunks:
        if c.lafmax is None:
            continue
        if peak_laf is None or c.lafmax > peak_laf:
            peak_laf = c.lafmax
            peak_at_local = format_local(c.dt_local)
            peak_at_utc = c.created_at

    device_counts = Counter(c.device_id for c in all_chunks if c.device_id)
    if site_label and site_label.strip():
        location_id = site_label.strip()
    elif device_counts:
        location_id = device_counts.most_common(1)[0][0]
    else:
        location_id = "Unknown site"

    if all_chunks:
        date_start = format_local_date(all_chunks[0].dt_local)
        date_end = format_local_date(all_chunks[-1].dt_local)
    else:
        date_start = date_end = "—"

    header_line = f"Site: {location_id} | {_header_date_span(date_start, date_end)}"

    zone_a = {
        "location_id": location_id,
        "timezone": tz_name,
        "date_start": date_start,
        "date_end": date_end,
        "header_line": header_line,
        "leq_db": leq,
        "leq_context": leq_context(leq),
        "l10_db": percentiles["l10_db"],
        "l50_db": percentiles["l50_db"],
        "l90_db": percentiles["l90_db"],
        "peak_lafmax_db": round(peak_laf, 1) if peak_laf is not None else None,
        "peak_lafmax_at_local": peak_at_local,
        "peak_lafmax_at_utc": peak_at_utc,
        "recordings": recording_summaries,
        "chunk_count": len(all_chunks),
    }

    typical_day = build_hourly_profile(all_chunks)
    timeline = build_timeline_profile(all_chunks)
    zone_b = {
        "timezone": tz_name,
        "timeline": timeline,
        "typical_day": typical_day,
        # Alias kept for takeaway / older clients.
        "hourly": typical_day,
        "default_view": "timeline",
        "night_hours": sorted(NIGHT_HOURS, key=lambda h: (h < 12, h)),
    }

    diet = sound_diet_from_events(acoustic_events)
    top_events = sorted(
        acoustic_events,
        key=lambda e: (-e.duration_s, -(e.max_dba or 0)),
    )[:12]
    zone_c = {
        "sound_diet": diet,
        "diet_basis": "duration_seconds",
        "acoustic_events_top": [
            {
                "resolved_label": e.resolved_label,
                "macro": e.macro,
                "start_local": e.start_local,
                "end_local": e.end_local,
                "duration_s": e.duration_s,
                "max_dba": e.max_dba,
                "leq_db": e.leq_db,
                "chunk_count": e.chunk_count,
            }
            for e in top_events
        ],
        "takeaway": "",
    }
    zone_c["takeaway"] = build_takeaway(zone_b=zone_b, zone_c=zone_c)

    meta = {
        "min_confidence": float(label_map.get("min_confidence", 0.25)),
        "timezone": tz_name,
        "timezone_warning": tz_warning,
        "segmentation": dict(label_map.get("segmentation") or {}),
        "tire_hiss": dict(label_map.get("tire_hiss") or {}),
        "acoustic_event_count": len(acoustic_events),
    }

    return {
        "ok": True,
        "zone_a": zone_a,
        "zone_b": zone_b,
        "zone_c": zone_c,
        "meta": meta,
    }
