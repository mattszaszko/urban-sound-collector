"""Past-recording overview stats and slim chart series for the Data tab."""

from __future__ import annotations

import bisect
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from core.events import event_recording_id
from core.loudness import DEFAULT_CALIB_OFFSET

# Target chart points when downsampling long recordings.
CHART_MAX_POINTS = 1500
DEFAULT_LOUD_THRESHOLD_LAFMAX = 65.0
DEFAULT_SERIES_WINDOW_S = 120.0


def percentile_nearest(sorted_values: list[float], pct: float) -> float | None:
    """Percentile on a pre-sorted ascending list (nearest-rank)."""
    if not sorted_values:
        return None
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    # Nearest-rank: index = ceil(p/100 * n) - 1
    n = len(sorted_values)
    rank = max(1, int((pct / 100.0) * n + 0.999999999))
    return float(sorted_values[min(n, rank) - 1])


def loud_disturbance_stats(
    lafmax_values: list[float],
    *,
    threshold: float,
) -> tuple[int, float]:
    """Return (loud_chunk_count, loud_pct)."""
    if not lafmax_values:
        return 0, 0.0
    loud = sum(1 for v in lafmax_values if v >= threshold)
    return loud, round(100.0 * loud / len(lafmax_values), 1)


def slim_event_point(
    event: dict,
    *,
    calib_offset: float = DEFAULT_CALIB_OFFSET,
    display_db_offset: float = 0.0,
) -> dict | None:
    """Project a JSONL event into a chart/detail point.

    ``display_db_offset`` shifts ``dba`` / ``lafmax`` for UI/analysis only
    (does not mutate the underlying event). Gate L90 relative is unchanged.
    """
    created = event.get("created_at")
    if not isinstance(created, str):
        return None

    def _f(key: str) -> float | None:
        raw = event.get(key)
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    dba = _f("dBA_spl")
    lafmax = _f("LAFmax_dB")
    if display_db_offset:
        if dba is not None:
            dba = round(dba + float(display_db_offset), 1)
        if lafmax is not None:
            lafmax = round(lafmax + float(display_db_offset), 1)

    prep = event.get("yamnet_preprocess")
    l90_rel = None
    gated = False
    gate_open = False
    if isinstance(prep, dict):
        floor = prep.get("ambient_noise_floor_dbfs")
        try:
            if floor is not None:
                l90_rel = round(float(floor) + float(calib_offset), 1)
        except (TypeError, ValueError):
            l90_rel = None
        gated = bool(prep.get("gated"))
        gate_open = bool(prep.get("gate_open"))

    yamnet = event.get("top_label")
    if not isinstance(yamnet, str):
        yamnet = None
    if yamnet == "gated":
        gated = True
        gate_open = False

    clap_status = event.get("clap_status")
    if not isinstance(clap_status, str):
        clap_status = None
    clap_label = event.get("clap_top_label")
    if not isinstance(clap_label, str):
        clap_label = None
    clap_conf = _f("clap_top_confidence")
    yamnet_conf = _f("top_confidence")

    clap_event_seconds: float | None = None
    clap_meta = event.get("clap_meta")
    if isinstance(clap_meta, dict):
        for key in ("event_seconds", "window_seconds"):
            raw_sec = clap_meta.get(key)
            try:
                if raw_sec is not None:
                    clap_event_seconds = float(raw_sec)
                    break
            except (TypeError, ValueError):
                continue

    chunk_index = event.get("chunk_index")
    try:
        chunk_i = int(chunk_index) if chunk_index is not None else None
    except (TypeError, ValueError):
        chunk_i = None

    point: dict[str, Any] = {
        "t": created,
        "chunk_index": chunk_i,
        "dba": dba,
        "lafmax": lafmax,
        "l90_rel": l90_rel,
        "yamnet": yamnet,
        "yamnet_conf": yamnet_conf,
        "gated": gated,
        "gate_open": gate_open,
        "clap_status": clap_status,
        "clap": clap_label,
        "clap_conf": clap_conf,
        "clap_event_seconds": clap_event_seconds,
    }

    # Compact A-weighted spectrum for chart coloring / Spectrum widget (timbre, not ML labels).
    spectrum = event.get("spectrum")
    if isinstance(spectrum, dict):
        a_metrics = spectrum.get("a")
        if isinstance(a_metrics, dict):
            raw_centroid = a_metrics.get("centroid_hz")
            try:
                if raw_centroid is not None:
                    point["centroid_hz"] = float(raw_centroid)
            except (TypeError, ValueError):
                pass
            energy = a_metrics.get("energy_pct")
            if isinstance(energy, dict):
                bands: dict[str, float] = {}
                for src_key, dest_key in (("low", "dark"), ("mid", "mid"), ("high", "bright")):
                    raw_pct = energy.get(src_key)
                    try:
                        if raw_pct is not None:
                            bands[dest_key] = float(raw_pct)
                    except (TypeError, ValueError):
                        continue
                if bands:
                    point["spectrum_bands"] = bands

    return point


def _point_confidence(point: dict, key: str) -> float | None:
    raw = point.get(key)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def yamnet_label_spans(
    points: list[dict],
    *,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Consecutive non-gated same-label runs (for chart brackets).

    Points without ``yamnet_conf`` (older JSONL) are kept so labels still show.
    ``min_confidence`` hides runs whose confidence is known and below the floor.
    """
    out: list[dict] = []
    run_label: str | None = None
    run_start_t: str | None = None
    run_end_t: str | None = None
    run_dba: float | None = None
    run_conf: float | None = None

    def flush() -> None:
        nonlocal run_label, run_start_t, run_end_t, run_dba, run_conf
        if run_label and run_start_t and run_end_t and run_dba is not None:
            span: dict[str, Any] = {
                "t0": run_start_t,
                "t1": run_end_t,
                "dba": run_dba,
                "label": run_label,
            }
            if run_conf is not None:
                span["confidence"] = run_conf
            out.append(span)
        run_label = None
        run_start_t = None
        run_end_t = None
        run_dba = None
        run_conf = None

    for p in points:
        label = p.get("yamnet")
        conf = _point_confidence(p, "yamnet_conf")
        below = conf is not None and conf < float(min_confidence)
        if (
            p.get("gated")
            or not label
            or label == "gated"
            or p.get("dba") is None
            or below
        ):
            flush()
            continue
        if label != run_label:
            flush()
            run_label = str(label)
            run_start_t = p["t"]
            run_end_t = p["t"]
            run_dba = p["dba"]
            run_conf = conf
        else:
            run_end_t = p["t"]
            # Keep first-chunk dBA as the bracket height (stable along the run).
            if run_conf is None and conf is not None:
                run_conf = conf
            elif run_conf is not None and conf is not None:
                run_conf = min(run_conf, conf)
    flush()
    return out


def yamnet_change_markers(
    points: list[dict],
    *,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Start-of-run markers (compat); prefer ``yamnet_label_spans`` for charts."""
    return [
        {"t": s["t0"], "dba": s["dba"], "label": s["label"], **(
            {"confidence": s["confidence"]} if "confidence" in s else {}
        )}
        for s in yamnet_label_spans(points, min_confidence=min_confidence)
    ]


def clap_triggered_markers(
    points: list[dict],
    *,
    min_confidence: float = 0.0,
) -> list[dict]:
    out: list[dict] = []
    for p in points:
        if p.get("clap_status") != "triggered" or not p.get("clap") or p.get("dba") is None:
            continue
        conf = _point_confidence(p, "clap_conf")
        if conf is not None and conf < float(min_confidence):
            continue
        marker: dict[str, Any] = {"t": p["t"], "dba": p["dba"], "label": p["clap"]}
        if conf is not None:
            marker["confidence"] = conf
        secs = p.get("clap_event_seconds")
        try:
            if secs is not None and float(secs) > 0:
                marker["event_seconds"] = float(secs)
        except (TypeError, ValueError):
            pass
        out.append(marker)
    return out


def downsample_points(points: list[dict], max_points: int = CHART_MAX_POINTS) -> list[dict]:
    """Bucket points, keeping max dBA / LAFmax per bucket for chart visibility."""
    n = len(points)
    if n <= max_points or max_points < 2:
        return points
    out: list[dict] = []
    for i in range(max_points):
        start = int(i * n / max_points)
        end = int((i + 1) * n / max_points)
        bucket = points[start:end]
        if not bucket:
            continue
        best = bucket[0]
        best_dba = best.get("dba")
        best_laf = best.get("lafmax")
        for p in bucket[1:]:
            dba = p.get("dba")
            laf = p.get("lafmax")
            if dba is not None and (best_dba is None or dba > best_dba):
                best = p
                best_dba = dba
                best_laf = laf
            elif laf is not None and (best_laf is None or laf > (best_laf or -1e9)):
                # Prefer louder LAFmax when dBA tie/missing
                if best_dba is None or dba == best_dba:
                    best = p
                    best_laf = laf
        out.append(best)
    return out


def created_at_to_ms(created_at: object) -> float | None:
    """Parse an event/point ``created_at`` string to UTC epoch milliseconds."""
    if not isinstance(created_at, str) or not created_at:
        return None
    raw = created_at.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0


def slim_points_from_events(
    events: list[dict],
    *,
    calib_offset: float = DEFAULT_CALIB_OFFSET,
    display_db_offset: float = 0.0,
) -> list[dict]:
    """Project events to slim chart points (skip rows without ``created_at``)."""
    out: list[dict] = []
    for event in events:
        slim = slim_event_point(
            event,
            calib_offset=calib_offset,
            display_db_offset=display_db_offset,
        )
        if slim is not None:
            out.append(slim)
    return out


def build_overview_window_series(
    slim_points: list[dict],
    *,
    center_ms: float,
    window_s: float = DEFAULT_SERIES_WINDOW_S,
) -> dict[str, Any]:
    """Return full-resolution slim points for a time window around ``center_ms``."""
    window_s = max(30.0, min(900.0, float(window_s)))
    half_ms = window_s * 500.0
    lo = float(center_ms) - half_ms
    hi = float(center_ms) + half_ms

    # Timestamps in recording order (JSONL is chronological). Missing → +inf so they sort last.
    keys: list[float] = []
    for p in slim_points:
        ms = created_at_to_ms(p.get("t"))
        keys.append(ms if ms is not None else float("inf"))

    start_i = bisect.bisect_left(keys, lo)
    end_i = bisect.bisect_right(keys, hi)
    filtered = [
        p
        for p, ms in zip(slim_points[start_i:end_i], keys[start_i:end_i])
        if ms != float("inf") and lo <= ms <= hi
    ]

    return {
        "ok": True,
        "center_ms": float(center_ms),
        "window_s": window_s,
        "t_min": filtered[0].get("t") if filtered else None,
        "t_max": filtered[-1].get("t") if filtered else None,
        "points": filtered,
        "yamnet_spans": yamnet_label_spans(filtered),
        "yamnet_markers": yamnet_change_markers(filtered),
        "clap_markers": clap_triggered_markers(filtered),
    }


def build_recording_overview(
    events: list[dict],
    *,
    name: str,
    has_wav: bool,
    wav_name: str | None,
    recording_active: bool,
    loud_threshold: float = DEFAULT_LOUD_THRESHOLD_LAFMAX,
    calib_offset: float = DEFAULT_CALIB_OFFSET,
    display_db_offset: float = 0.0,
) -> dict[str, Any]:
    """Build overview payload from in-memory events (spectrum ignored)."""
    points_full: list[dict] = []
    dbas: list[float] = []
    lafmaxes: list[float] = []
    gate_open_n = 0
    gated_n = 0
    yamnet_counts: Counter[str] = Counter()
    clap_counts: Counter[str] = Counter()
    clap_triggered = 0
    device_id = None
    recording_id = None

    for event in events:
        slim = slim_event_point(
            event,
            calib_offset=calib_offset,
            display_db_offset=display_db_offset,
        )
        if slim is None:
            continue
        points_full.append(slim)
        if device_id is None and isinstance(event.get("device_id"), str):
            device_id = event["device_id"]
        if recording_id is None:
            recording_id = event_recording_id(event)
        if slim["dba"] is not None:
            dbas.append(float(slim["dba"]))
        if slim["lafmax"] is not None:
            lafmaxes.append(float(slim["lafmax"]))
        if slim["gated"]:
            gated_n += 1
        elif slim["gate_open"]:
            gate_open_n += 1
        label = slim.get("yamnet")
        if label and label != "gated" and not slim["gated"]:
            yamnet_counts[label] += 1
        if slim.get("clap_status") == "triggered":
            clap_triggered += 1
            clap_lab = slim.get("clap")
            if clap_lab:
                clap_counts[clap_lab] += 1

    chunks = len(points_full)
    started_at = points_full[0]["t"] if points_full else None
    ended_at = points_full[-1]["t"] if points_full else None
    duration_s = 0
    if chunks >= 2:
        # Prefer timestamp delta when parseable; else ~0.975 s/chunk.
        try:
            def _parse(t: str) -> datetime:
                raw = t[:-1] + "+00:00" if t.endswith("Z") else t
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt

            duration_s = max(
                0,
                int((_parse(ended_at) - _parse(started_at)).total_seconds()),
            )
        except Exception:  # noqa: BLE001
            duration_s = int(round((chunks - 1) * 0.975))
    elif chunks == 1:
        duration_s = 1

    dba_sorted = sorted(dbas)
    loud_n, loud_pct = loud_disturbance_stats(lafmaxes, threshold=loud_threshold)

    def _top(counter: Counter[str], n: int = 8) -> list[dict[str, Any]]:
        return [{"label": k, "count": int(v)} for k, v in counter.most_common(n)]

    points_chart = downsample_points(points_full)
    points_full_flag = len(points_chart) == len(points_full)

    return {
        "ok": True,
        "name": name,
        "has_wav": bool(has_wav),
        "wav_name": wav_name if has_wav else None,
        "recording_active": bool(recording_active),
        "calib_offset": float(calib_offset),
        "display_db_offset": float(display_db_offset),
        "summary": {
            "device_id": device_id,
            "recording_id": recording_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": duration_s,
            "chunks": chunks,
            "loudness": {
                "dba_mean": round(sum(dbas) / len(dbas), 1) if dbas else None,
                "dba_median": (
                    round(percentile_nearest(dba_sorted, 50) or 0.0, 1)
                    if dba_sorted
                    else None
                ),
                # Noise monitoring: L10 = level exceeded 10% of time (louder).
                "dba_l10": (
                    round(percentile_nearest(dba_sorted, 90) or 0.0, 1)
                    if dba_sorted
                    else None
                ),
                "dba_l50": (
                    round(percentile_nearest(dba_sorted, 50) or 0.0, 1)
                    if dba_sorted
                    else None
                ),
                "dba_l90": (
                    round(percentile_nearest(dba_sorted, 10) or 0.0, 1)
                    if dba_sorted
                    else None
                ),
                "lafmax_mean": (
                    round(sum(lafmaxes) / len(lafmaxes), 1) if lafmaxes else None
                ),
                "lafmax_max": round(max(lafmaxes), 1) if lafmaxes else None,
            },
            "disturbances": {
                "threshold_lafmax_db": float(loud_threshold),
                "loud_chunks": loud_n,
                "loud_pct": loud_pct,
                "gate_open_pct": (
                    round(100.0 * gate_open_n / chunks, 1) if chunks else 0.0
                ),
                "gated_pct": round(100.0 * gated_n / chunks, 1) if chunks else 0.0,
            },
            "yamnet": {
                "classified_chunks": sum(yamnet_counts.values()),
                "gated_chunks": gated_n,
                "top_labels": _top(yamnet_counts),
            },
            "clap": {
                "triggered": clap_triggered,
                "top_labels": _top(clap_counts),
            },
        },
        # Coarse series for the scrubber only; Inspect chart loads a dense window via /overview/series.
        "points": points_chart,
        "points_full": points_full_flag,
        "yamnet_spans": yamnet_label_spans(points_chart),
        "yamnet_markers": yamnet_change_markers(points_chart),
        "clap_markers": clap_triggered_markers(points_chart),
        "scrub_points": points_chart,
        "series_window_s": DEFAULT_SERIES_WINDOW_S,
    }
