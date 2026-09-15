"""Report timezone helpers (UTC JSONL → site-local civil time)."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_SITE_TIMEZONE = "Europe/Amsterdam"


def resolve_zone(name: str | None) -> tuple[ZoneInfo, str, str | None]:
    """Return (zone, resolved_name, warning). Invalid/missing → Amsterdam."""
    raw = (name or "").strip() or DEFAULT_SITE_TIMEZONE
    try:
        return ZoneInfo(raw), raw, None
    except ZoneInfoNotFoundError:
        return (
            ZoneInfo(DEFAULT_SITE_TIMEZONE),
            DEFAULT_SITE_TIMEZONE,
            f"Unknown timezone {raw!r}; using {DEFAULT_SITE_TIMEZONE}",
        )


def parse_created_at_utc(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_local(dt_utc: datetime, zone: ZoneInfo) -> datetime:
    return dt_utc.astimezone(zone)


def format_local(dt_local: datetime) -> str:
    return dt_local.strftime("%Y-%m-%d %H:%M:%S")


def format_local_date(dt_local: datetime) -> str:
    return dt_local.strftime("%Y-%m-%d")
