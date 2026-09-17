"""Energy-envelope acoustic event segmentation and label resolution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from core.audio_constants import YAMNET_CHUNK_DURATION_SECONDS
from core.report.macros import label_to_macro
from core.report.timeutil import format_local, parse_created_at_utc, to_local
from core.report.votes import (
    MACRO_UNCLASSIFIED,
    is_yamnet_gated,
    load_display_theme_map,
    load_label_map,
    vote_label_for_event,
)


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def energetic_leq(levels_db: list[float]) -> float | None:
    """Energetic average of dB values."""
    if not levels_db:
        return None
    import math

    linear = [10.0 ** (L / 10.0) for L in levels_db]
    return round(10.0 * math.log10(sum(linear) / len(linear)), 1)


@dataclass
class AcousticChunk:
    created_at: str
    dt_utc: datetime
    dt_local: datetime
    dba: float
    lafmax: float | None
    vote_label: str | None
    gated: bool = False
    device_id: str | None = None
    source_file: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcousticEvent:
    start_utc: str
    end_utc: str
    start_local: str
    end_local: str
    duration_s: float
    chunk_count: int
    max_dba: float
    max_lafmax: float | None
    leq_db: float | None
    dominant_label: str | None
    resolved_label: str
    macro: str
    vote_labels: list[str]
    bridged_gap_chunks: int = 0


def prepare_chunks(
    events: list[dict],
    *,
    zone: ZoneInfo,
    label_map: dict[str, Any] | None = None,
    theme_by_label: dict[str, str] | None = None,
    source_file: str | None = None,
) -> list[AcousticChunk]:
    """Build acoustic timeline chunks (keep quiet/gated rows with valid dBA)."""
    cfg = label_map or load_label_map()
    themes = theme_by_label if theme_by_label is not None else load_display_theme_map()
    out: list[AcousticChunk] = []
    for event in events:
        dba = _f(event.get("dBA_spl"))
        if dba is None:
            continue
        created = event.get("created_at")
        dt_utc = parse_created_at_utc(created if isinstance(created, str) else None)
        if dt_utc is None:
            continue
        dt_local = to_local(dt_utc, zone)
        vote = vote_label_for_event(event, label_map=cfg, theme_by_label=themes)
        device = event.get("device_id")
        out.append(
            AcousticChunk(
                created_at=str(created),
                dt_utc=dt_utc,
                dt_local=dt_local,
                dba=dba,
                lafmax=_f(event.get("LAFmax_dB")),
                vote_label=vote,
                gated=is_yamnet_gated(event),
                device_id=device if isinstance(device, str) else None,
                source_file=source_file,
                raw={},
            )
        )
    out.sort(key=lambda c: c.dt_utc)
    return out


def finalize_event(
    member_chunks: list[AcousticChunk],
    *,
    bridged_gap_chunks: int,
    label_map: dict[str, Any] | None = None,
    theme_by_label: dict[str, str] | None = None,
) -> AcousticEvent | None:
    if not member_chunks:
        return None
    cfg = label_map or load_label_map()
    themes = theme_by_label if theme_by_label is not None else load_display_theme_map()
    seg = cfg.get("segmentation") or {}
    min_chunks = int(seg.get("min_event_chunks", 2))
    if len(member_chunks) < min_chunks:
        return None

    tire = cfg.get("tire_hiss") or {}
    confused = {str(x) for x in (tire.get("source_labels") or [])}
    override_label = str(tire.get("target_label") or "Vehicle / Tire Rolling")
    confused_max = float(tire.get("confused_min_event_max_dba", 52.0))

    votes = [c.vote_label for c in member_chunks if c.vote_label]
    dominant: str | None = None
    if votes:
        dominant = Counter(votes).most_common(1)[0][0]

    max_dba = max(c.dba for c in member_chunks)
    lafmaxes = [c.lafmax for c in member_chunks if c.lafmax is not None]
    max_laf = max(lafmaxes) if lafmaxes else None

    if dominant in confused and max_dba > confused_max:
        resolved = override_label
    elif dominant:
        resolved = dominant
    else:
        resolved = MACRO_UNCLASSIFIED

    macro = label_to_macro(resolved, label_map=cfg, theme_by_label=themes)
    # Prefer timestamp span; fall back to nominal chunk duration.
    span = (member_chunks[-1].dt_utc - member_chunks[0].dt_utc).total_seconds()
    if span > 0:
        duration_s = round(span + YAMNET_CHUNK_DURATION_SECONDS, 2)
    else:
        duration_s = round(len(member_chunks) * YAMNET_CHUNK_DURATION_SECONDS, 2)

    return AcousticEvent(
        start_utc=member_chunks[0].created_at,
        end_utc=member_chunks[-1].created_at,
        start_local=format_local(member_chunks[0].dt_local),
        end_local=format_local(member_chunks[-1].dt_local),
        duration_s=duration_s,
        chunk_count=len(member_chunks),
        max_dba=round(max_dba, 1),
        max_lafmax=round(max_laf, 1) if max_laf is not None else None,
        leq_db=energetic_leq([c.dba for c in member_chunks]),
        dominant_label=dominant,
        resolved_label=resolved,
        macro=macro,
        vote_labels=votes,
        bridged_gap_chunks=bridged_gap_chunks,
    )


def segment_acoustic_events(
    chunks: list[AcousticChunk],
    *,
    label_map: dict[str, Any] | None = None,
    theme_by_label: dict[str, str] | None = None,
) -> list[AcousticEvent]:
    """Group continuous elevated-energy chunks into acoustic events."""
    cfg = label_map or load_label_map()
    themes = theme_by_label if theme_by_label is not None else load_display_theme_map()
    seg = cfg.get("segmentation") or {}
    threshold = float(seg.get("active_dba_threshold", 45.0))
    max_gap = int(seg.get("max_gap_chunks", 2))

    events: list[AcousticEvent] = []
    current: list[AcousticChunk] = []
    quiet_gap = 0
    bridged = 0

    def _close() -> None:
        nonlocal current, quiet_gap, bridged
        finished = finalize_event(
            current,
            bridged_gap_chunks=bridged,
            label_map=cfg,
            theme_by_label=themes,
        )
        if finished is not None:
            events.append(finished)
        current = []
        quiet_gap = 0
        bridged = 0

    for chunk in chunks:
        active = chunk.dba >= threshold
        if active:
            if current and quiet_gap:
                bridged += quiet_gap
            quiet_gap = 0
            current.append(chunk)
            continue

        # Below threshold
        if not current:
            continue
        quiet_gap += 1
        if quiet_gap <= max_gap:
            current.append(chunk)
            continue
        # Gap exceeded: finalize with already-bridged quiet inside; exclude this chunk.
        bridged += max_gap
        _close()

    if current:
        # Trailing quiet inside grace: trim them before finalize for cleaner end.
        trailing = 0
        for c in reversed(current):
            if c.dba < threshold:
                trailing += 1
            else:
                break
        if 0 < trailing <= max_gap:
            current = current[:-trailing]
            bridged += trailing
        _close()

    return events
