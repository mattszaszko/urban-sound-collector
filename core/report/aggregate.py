"""Build three-zone dashboard report payload from JSONL events."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from core.audio_constants import YAMNET_CHUNK_DURATION_SECONDS
from core.recording_overview import percentile_nearest
from core.report.copy import build_takeaway
from core.report.macros import label_to_macro
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
from core.report.votes import MACRO_UNCLASSIFIED, load_display_theme_map, load_label_map

NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6}
DAY_HOURS = set(range(7, 22))
RELATIVE_SILENCE = "Relative silence"
DEFAULT_ACTIVE_DBA_THRESHOLD = 45.0
DEFAULT_L90_OFFSET_DB = 5.0


def leq_context(leq_db: float | None) -> str:
    """Map an A-weighted level (typically L50) to a layperson anchor band."""
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


def _clamp_nonneg_int(value: Any, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, n)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_report_segmentation_overrides(
    label_map: dict[str, Any],
    *,
    min_event_chunks: int | None = None,
    max_gap_chunks: int | None = None,
    threshold_mode: str | None = None,
    threshold_db: float | None = None,
    l90_offset_db: float | None = None,
    chunk_dbas: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Copy label_map with report-time segmentation overrides.

    Threshold modes:
    - absolute: use ``threshold_db`` (default 45)
    - l90_offset: use selection L90 + ``l90_offset_db`` (default +5; may be negative)
    """
    seg = dict(label_map.get("segmentation") or {})
    cfg_min = _clamp_nonneg_int(seg.get("min_event_chunks", 2), 2)
    cfg_gap = _clamp_nonneg_int(seg.get("max_gap_chunks", 2), 2)
    cfg_thr = _as_float(seg.get("active_dba_threshold", DEFAULT_ACTIVE_DBA_THRESHOLD), DEFAULT_ACTIVE_DBA_THRESHOLD)

    min_chunks = cfg_min if min_event_chunks is None else _clamp_nonneg_int(min_event_chunks, cfg_min)
    max_gap = cfg_gap if max_gap_chunks is None else _clamp_nonneg_int(max_gap_chunks, cfg_gap)

    mode = str(threshold_mode or "absolute").strip().lower()
    if mode not in {"absolute", "l90_offset"}:
        mode = "absolute"

    l90_used: float | None = None
    offset_used: float | None = None
    if mode == "l90_offset":
        offset_used = (
            DEFAULT_L90_OFFSET_DB
            if l90_offset_db is None
            else _as_float(l90_offset_db, DEFAULT_L90_OFFSET_DB)
        )
        percentiles = _level_percentiles(list(chunk_dbas or []))
        l90_used = percentiles["l90_db"]
        if l90_used is not None:
            resolved = float(l90_used) + float(offset_used)
        else:
            resolved = cfg_thr if threshold_db is None else _as_float(threshold_db, cfg_thr)
    else:
        resolved = cfg_thr if threshold_db is None else _as_float(threshold_db, cfg_thr)

    seg.update(
        {
            "min_event_chunks": min_chunks,
            "max_gap_chunks": max_gap,
            "active_dba_threshold": round(float(resolved), 2),
            "threshold_mode": mode,
        }
    )
    if offset_used is not None:
        seg["l90_offset_db"] = round(float(offset_used), 2)
    if l90_used is not None:
        seg["l90_db_used"] = round(float(l90_used), 1)

    out_map = {**label_map, "segmentation": seg}
    meta = {
        "min_event_chunks": min_chunks,
        "max_gap_chunks": max_gap,
        "threshold_mode": mode,
        "active_dba_threshold": seg["active_dba_threshold"],
        "l90_offset_db": seg.get("l90_offset_db"),
        "l90_db_used": seg.get("l90_db_used"),
    }
    return out_map, meta


def chunk_time_budget_category(
    chunk: AcousticChunk,
    *,
    label_map: dict[str, Any] | None = None,
    theme_by_label: dict[str, str] | None = None,
) -> str:
    """Map one chunk to a time-budget category (silence or macro)."""
    if chunk.gated:
        return RELATIVE_SILENCE
    if chunk.vote_label:
        return label_to_macro(
            chunk.vote_label,
            label_map=label_map,
            theme_by_label=theme_by_label,
        )
    return MACRO_UNCLASSIFIED


def _category_order(label_map: dict[str, Any], present: set[str]) -> list[str]:
    preferred = [RELATIVE_SILENCE]
    preferred.extend(str(m) for m in (label_map.get("macros") or []) if str(m) != RELATIVE_SILENCE)
    if MACRO_UNCLASSIFIED not in preferred:
        preferred.append(MACRO_UNCLASSIFIED)
    ordered = [c for c in preferred if c in present]
    extras = sorted(c for c in present if c not in ordered)
    return ordered + extras


def _budget_rows(seconds: Counter[str], *, label_map: dict[str, Any]) -> list[dict[str, Any]]:
    total = float(sum(seconds.values()))
    rows: list[dict[str, Any]] = []
    for cat in _category_order(label_map, set(seconds.keys())):
        secs = float(seconds[cat])
        pct = round(100.0 * secs / total, 1) if total > 0 else 0.0
        rows.append(
            {
                "category": cat,
                "pct": pct,
                "seconds": round(secs, 1),
            }
        )
    return rows


def build_time_budget(
    chunks: list[AcousticChunk],
    *,
    label_map: dict[str, Any] | None = None,
    theme_by_label: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Whole-selection time budget including Relative silence (gated chunks)."""
    cfg = label_map or load_label_map()
    themes = theme_by_label if theme_by_label is not None else load_display_theme_map()
    chunk_dur = float(YAMNET_CHUNK_DURATION_SECONDS)

    total_s: Counter[str] = Counter()
    day_s: Counter[str] = Counter()
    night_s: Counter[str] = Counter()
    hour_counters: dict[datetime, Counter[str]] = {}

    def _hour_floor(dt: datetime) -> datetime:
        return dt.replace(minute=0, second=0, microsecond=0)

    for chunk in chunks:
        cat = chunk_time_budget_category(
            chunk, label_map=cfg, theme_by_label=themes
        )
        total_s[cat] += chunk_dur
        if chunk.dt_local.hour in NIGHT_HOURS:
            night_s[cat] += chunk_dur
        else:
            day_s[cat] += chunk_dur
        key = _hour_floor(chunk.dt_local)
        hour_counters.setdefault(key, Counter())[cat] += chunk_dur

    hourly: list[dict[str, Any]] = []
    if chunks:
        start = _hour_floor(chunks[0].dt_local)
        end = _hour_floor(chunks[-1].dt_local)
        cur = start
        while cur <= end:
            counter = hour_counters.get(cur, Counter())
            hour = int(cur.hour)
            rows = _budget_rows(counter, label_map=cfg)
            hour_total = float(sum(counter.values()))
            hourly.append(
                {
                    "t_local": cur.isoformat(timespec="seconds"),
                    "label": cur.strftime("%d %b %H:%M"),
                    "date_short": cur.strftime("%d %b"),
                    "hour": hour,
                    "period": "night" if hour in NIGHT_HOURS else "day",
                    "seconds_total": round(hour_total, 1),
                    "shares": rows,
                }
            )
            cur = cur + timedelta(hours=1)

    categories = _category_order(
        cfg,
        set(total_s.keys()) | set(day_s.keys()) | set(night_s.keys()),
    )
    return {
        "chunk_duration_s": chunk_dur,
        "categories": categories,
        "total": _budget_rows(total_s, label_map=cfg),
        "day": _budget_rows(day_s, label_map=cfg),
        "night": _budget_rows(night_s, label_map=cfg),
        "hourly": hourly,
        "seconds_total": round(float(sum(total_s.values())), 1),
        "seconds_day": round(float(sum(day_s.values())), 1),
        "seconds_night": round(float(sum(night_s.values())), 1),
    }

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
    min_event_chunks: int | None = None,
    max_gap_chunks: int | None = None,
    threshold_mode: str | None = None,
    threshold_db: float | None = None,
    l90_offset_db: float | None = None,
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
        duration_s: float | None = None
        if prepared:
            span = (prepared[-1].dt_utc - prepared[0].dt_utc).total_seconds()
            if span > 0:
                duration_s = round(span + YAMNET_CHUNK_DURATION_SECONDS, 1)
            else:
                duration_s = round(len(prepared) * YAMNET_CHUNK_DURATION_SECONDS, 1)
        recording_summaries.append(
            {"name": name, "chunks": len(prepared), "duration_s": duration_s}
        )

    all_chunks.sort(key=lambda c: c.dt_utc)

    label_map, seg_meta = apply_report_segmentation_overrides(
        label_map,
        min_event_chunks=min_event_chunks,
        max_gap_chunks=max_gap_chunks,
        threshold_mode=threshold_mode,
        threshold_db=threshold_db,
        l90_offset_db=l90_offset_db,
        chunk_dbas=[c.dba for c in all_chunks],
    )

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

    total_duration_s = round(
        sum(float(r["duration_s"]) for r in recording_summaries if r.get("duration_s") is not None),
        1,
    )

    zone_a = {
        "location_id": location_id,
        "timezone": tz_name,
        "date_start": date_start,
        "date_end": date_end,
        "header_line": header_line,
        "leq_db": leq,
        "leq_context": leq_context(
            percentiles["l50_db"] if percentiles["l50_db"] is not None else leq
        ),
        "l10_db": percentiles["l10_db"],
        "l50_db": percentiles["l50_db"],
        "l90_db": percentiles["l90_db"],
        "peak_lafmax_db": round(peak_laf, 1) if peak_laf is not None else None,
        "peak_lafmax_at_local": peak_at_local,
        "peak_lafmax_at_utc": peak_at_utc,
        "recordings": recording_summaries,
        "chunk_count": len(all_chunks),
        "duration_s": total_duration_s if recording_summaries else None,
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
    time_budget = build_time_budget(
        all_chunks, label_map=label_map, theme_by_label=themes
    )
    top_events = sorted(
        acoustic_events,
        key=lambda e: (-e.duration_s, -(e.max_dba or 0)),
    )[:12]
    zone_c = {
        "time_budget": time_budget,
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
        "report_options": seg_meta,
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
