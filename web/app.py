"""FastAPI web interface for the Urban Sound Collector."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import psutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from jinja2 import Environment, FileSystemLoader

from core.audio_constants import (
    CAPTURE_CHUNK_SAMPLES,
    CAPTURE_SAMPLE_RATE,
    CAPTURE_STARTUP_DISCARD_SECONDS,
)
from core.audio_writer import (
    DEFAULT_AUDIO_FORMAT,
    ensure_flac_seekable,
    normalize_audio_format,
    read_flac_total_samples,
    sibling_audio_path,
)
from core.capture_alsa import CaptureDeviceError, check_capture_device_available
from core.events import event_recording_id
from core.export_filter import export_filename, iter_filtered_jsonl
from core.host_identity import default_device_id, hostname
from core.loudness import DEFAULT_CALIB_OFFSET
from core.report import DEFAULT_SITE_TIMEZONE, build_dashboard_report
from core.report.timeutil import resolve_zone, to_local
from core.recording_overview import (
    DEFAULT_LOUD_THRESHOLD_LAFMAX,
    DEFAULT_SERIES_WINDOW_S,
    build_overview_window_series,
    build_recording_overview,
    created_at_to_ms,
    slim_event_point,
    slim_points_from_events,
)
from core.yamnet_preprocess import DEFAULT_GATE_SENSITIVITY_DB

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
RECORDINGS_DIR = REPO_ROOT / "recordings"
LOGS_DIR = REPO_ROOT / "logs"
ENV_PATH = REPO_ROOT / ".env"
MAIN_PY = REPO_ROOT / "main.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_PYTHON_WIN = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

# Fall back to system python if venv not present (useful for dev on PC)
if VENV_PYTHON.exists():
    PYTHON = str(VENV_PYTHON)
elif VENV_PYTHON_WIN.exists():
    PYTHON = str(VENV_PYTHON_WIN)
else:
    PYTHON = sys.executable

CONFIG_DIR = REPO_ROOT / "config"
CLAP_PROMPTS_PATH = CONFIG_DIR / "clap_prompts.json"
CLAP_TRIGGERS_PATH = CONFIG_DIR / "clap_triggers.json"
YAMNET_CATALOG_PATH = CONFIG_DIR / "yamnet_label_catalog.json"

# ---------------------------------------------------------------------------
# Config from .env
# ---------------------------------------------------------------------------
PASSWORD = os.environ.get("USC_PASSWORD", "changeme")
VIEWER_PASSWORD = os.environ.get("USC_VIEWER_PASSWORD", "").strip()
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-please")
PORT = int(os.environ.get("PORT", "8080"))
DEFAULT_DEVICE_ID = default_device_id()
DEFAULT_ALSA_DEVICE = os.environ.get(
    "ALSA_DEVICE", "plughw:CARD=sndrpigooglevoi,DEV=0"
)
SITE_LABEL = os.environ.get("SITE_LABEL", "").strip()
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").strip()
SITE_TIMEZONE = os.environ.get("SITE_TIMEZONE", DEFAULT_SITE_TIMEZONE).strip() or DEFAULT_SITE_TIMEZONE


def _parse_display_db_offset(raw: object) -> float:
    try:
        return max(-40.0, min(40.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


DISPLAY_DB_OFFSET = _parse_display_db_offset(os.environ.get("DISPLAY_DB_OFFSET", "0"))
AUDIO_FORMAT = normalize_audio_format(
    os.environ.get("AUDIO_FORMAT", DEFAULT_AUDIO_FORMAT)
)

os.environ["SECRET_KEY"] = SECRET_KEY
os.environ["USC_PASSWORD"] = PASSWORD
os.environ["DISPLAY_DB_OFFSET"] = str(DISPLAY_DB_OFFSET)
os.environ["AUDIO_FORMAT"] = AUDIO_FORMAT
if VIEWER_PASSWORD:
    os.environ["USC_VIEWER_PASSWORD"] = VIEWER_PASSWORD
else:
    os.environ.pop("USC_VIEWER_PASSWORD", None)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
from web.auth import (  # noqa: E402  (after env setup)
    SESSION_COOKIE,
    get_role,
    is_admin,
    is_authenticated,
    login_page,
    make_session_cookie,
    verify_login,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Urban Sound Collector")
_jinja_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
    autoescape=True,
)


def _render(template_name: str, **ctx) -> HTMLResponse:
    t = _jinja_env.get_template(template_name)
    return HTMLResponse(t.render(**ctx))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%MZ")


def _default_recording_name() -> str:
    """Suggest a sensible prefix from the current UTC hour."""
    hour = datetime.now(timezone.utc).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "day"
    if 18 <= hour < 23:
        return "evening"
    return "night"


def _sanitize_recording_name(name: str) -> str:
    """Keep only safe filename characters; fall back to 'recording'."""
    cleaned = "".join(c for c in name.strip() if c.isalnum() or c in "-_")
    return cleaned or "recording"


def _hours_to_timeout(hours: float) -> str:
    """Convert hours to a GNU timeout duration string."""
    if hours <= 0:
        hours = 1.0
    # Prefer whole hours; otherwise use minutes for fractional values.
    if abs(hours - round(hours)) < 1e-9:
        return f"{int(round(hours))}h"
    minutes = max(1, int(round(hours * 60)))
    return f"{minutes}m"


def _read_jsonl_event(path: Path, *, last: bool) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        if last:
            with path.open("rb") as f:
                try:
                    f.seek(-4096, 2)
                except OSError:
                    f.seek(0)
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            if not lines:
                return None
            return json.loads(lines[-1])
        with path.open(encoding="utf-8") as f:
            line = f.readline()
        return json.loads(line) if line.strip() else None
    except (OSError, json.JSONDecodeError):
        return None


def _tail_jsonl_events(path: Path, max_lines: int = 320) -> list[dict]:
    """Return up to ``max_lines`` trailing JSONL events (oldest → newest)."""
    if max_lines <= 0 or not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []

    # Grow from the end until we have enough complete lines (spectrum rows are large).
    chunk = min(size, 256 * 1024)
    data = b""
    start = 0
    lines: list[bytes] = []
    try:
        with path.open("rb") as f:
            while True:
                start = max(0, size - chunk)
                f.seek(start)
                data = f.read(size - start)
                lines = [ln for ln in data.splitlines() if ln.strip()]
                if start == 0 or len(lines) >= max_lines + 1:
                    break
                chunk = min(size, chunk * 2)
    except OSError:
        return []

    if start > 0 and lines:
        # First line may be a partial record after seek.
        lines = lines[1:]
    lines = lines[-max_lines:]

    events: list[dict] = []
    for ln in lines:
        try:
            events.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return events


def _parse_event_time(created_at: object) -> datetime | None:
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
    return dt.astimezone(timezone.utc)


def _slim_series_point(event: dict, *, calib_offset: float) -> dict | None:
    """Project a JSONL event into a chart-friendly point."""
    return slim_event_point(
        event,
        calib_offset=calib_offset,
        display_db_offset=float(DISPLAY_DB_OFFSET),
    )


def _shift_slim_db_levels(points: list[dict], offset: float) -> list[dict]:
    """Return slim points with dba/lafmax shifted (raw cache stays unshifted)."""
    if not offset or not points:
        return points
    out: list[dict] = []
    for p in points:
        q = dict(p)
        if q.get("dba") is not None:
            try:
                q["dba"] = round(float(q["dba"]) + offset, 1)
            except (TypeError, ValueError):
                pass
        if q.get("lafmax") is not None:
            try:
                q["lafmax"] = round(float(q["lafmax"]) + offset, 1)
            except (TypeError, ValueError):
                pass
        out.append(q)
    return out


def _upsert_env_value(key: str, value: str) -> None:
    """Create or replace ``key=value`` in the repo ``.env`` file."""
    path = ENV_PATH
    lines: list[str] = []
    if path.is_file():
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError:
            lines = []
    prefix = f"{key}="
    found = False
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(prefix) or stripped.startswith(f"{key} ="):
            out.append(f"{key}={value}\n")
            found = True
        else:
            out.append(line if line.endswith("\n") else f"{line}\n")
    if not found:
        if out and not out[-1].endswith("\n"):
            out[-1] = f"{out[-1]}\n"
        if out and out[-1].strip():
            out.append("\n")
        out.append(f"{key}={value}\n")
    path.write_text("".join(out), encoding="utf-8")


def _set_display_db_offset(value: float) -> float:
    """Persist and activate a new display offset; returns the clamped value."""
    global DISPLAY_DB_OFFSET
    offset = _parse_display_db_offset(value)
    DISPLAY_DB_OFFSET = offset
    os.environ["DISPLAY_DB_OFFSET"] = str(offset)
    _upsert_env_value("DISPLAY_DB_OFFSET", str(offset))
    return offset


def _set_audio_format(value: object) -> str:
    """Persist and activate WAV/FLAC preference; returns normalized format."""
    global AUDIO_FORMAT
    fmt = normalize_audio_format(value, default=DEFAULT_AUDIO_FORMAT)
    AUDIO_FORMAT = fmt
    os.environ["AUDIO_FORMAT"] = fmt
    _upsert_env_value("AUDIO_FORMAT", fmt)
    return fmt


def _status_series(*, window_s: int = 120) -> dict:
    """Build a slim rolling series for the live loudness chart."""
    window_s = max(60, min(900, int(window_s)))
    calib = float(DEFAULT_CALIB_OFFSET)
    path = _active_output_path()
    if path is None:
        return {
            "ok": True,
            "recording": False,
            "window_s": window_s,
            "calib_offset": calib,
            "display_db_offset": float(DISPLAY_DB_OFFSET),
            "points": [],
        }

    # ~0.975 s/chunk → slightly over-fetch lines for the window.
    max_lines = max(80, int(window_s / 0.9) + 20)
    events = _tail_jsonl_events(path, max_lines=max_lines)
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - window_s
    points: list[dict] = []
    for event in events:
        dt = _parse_event_time(event.get("created_at"))
        if dt is None or dt.timestamp() < cutoff:
            continue
        slim = _slim_series_point(event, calib_offset=calib)
        if slim is not None:
            points.append(slim)
    return {
        "ok": True,
        "recording": True,
        "window_s": window_s,
        "calib_offset": calib,
        "display_db_offset": float(DISPLAY_DB_OFFSET),
        "points": points,
    }


def _stop_collector() -> bool:
    """Stop a running collector process. Returns True if one was stopped."""
    proc = _find_collector_process()
    if proc is None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    return True


def _run_poweroff() -> None:
    """Flush filesystems and power off the Pi (requires passwordless sudo)."""
    subprocess.run(["sudo", "sync"], check=False)
    result = subprocess.run(["sudo", "systemctl", "poweroff"], check=False)
    if result.returncode != 0:
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)


# Brief pause so the shutdown API response can reach the browser before poweroff.
_POWEROFF_RESPONSE_FLUSH_SEC = 1.5


def _schedule_immediate_poweroff() -> None:
    """Issue poweroff almost immediately after the HTTP response can flush."""

    def _worker() -> None:
        time.sleep(_POWEROFF_RESPONSE_FLUSH_SEC)
        _run_poweroff()

    threading.Thread(target=_worker, daemon=True).start()


def _find_collector_process() -> psutil.Process | None:
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info["cmdline"] or []
            if any("main.py" in c for c in cmd) and any(
                "python" in c.lower() for c in cmd
            ):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def _get_status() -> dict:
    """Return current collector status dict."""
    proc = _find_collector_process()
    if proc is None:
        return {
            "recording": False,
            "display_db_offset": float(DISPLAY_DB_OFFSET),
        }

    # Find the output file from the cmdline
    cmd = proc.cmdline()
    output_file: str | None = None
    for i, part in enumerate(cmd):
        if part in ("-o", "--output") and i + 1 < len(cmd):
            output_file = cmd[i + 1]
            break

    chunk_count = 0
    last_label = None
    last_dba = None
    last_centroid_hz = None
    last_spectrum_bands = None
    started_at = None
    recording_id = None

    if output_file:
        output_path = Path(output_file)
        last_event = _read_jsonl_event(output_path, last=True)
        if last_event:
            chunk_count = last_event.get("chunk_index", 0) + 1
            last_label = last_event.get("top_label")
            recording_id = event_recording_id(last_event)
            slim = slim_event_point(
                last_event,
                calib_offset=float(DEFAULT_CALIB_OFFSET),
                display_db_offset=float(DISPLAY_DB_OFFSET),
            )
            if slim:
                last_dba = slim.get("dba")
                last_centroid_hz = slim.get("centroid_hz")
                last_spectrum_bands = slim.get("spectrum_bands")
            else:
                raw_dba = last_event.get("dBA_spl")
                try:
                    last_dba = (
                        round(float(raw_dba) + float(DISPLAY_DB_OFFSET), 1)
                        if raw_dba is not None
                        else None
                    )
                except (TypeError, ValueError):
                    last_dba = raw_dba
        first_event = _read_jsonl_event(output_path, last=False)
        if first_event:
            started_at = first_event.get("created_at")
            if recording_id is None:
                recording_id = event_recording_id(first_event)
        output_file = str(output_path)

    elapsed_s = None
    if started_at:
        try:
            dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            elapsed_s = int(
                (datetime.now(timezone.utc) - dt).total_seconds()
            )
        except ValueError:
            pass

    return {
        "recording": True,
        "pid": proc.pid,
        "output_file": output_file,
        "chunk_count": chunk_count,
        "last_label": last_label,
        "last_dba": last_dba,
        "last_centroid_hz": last_centroid_hz,
        "last_spectrum_bands": last_spectrum_bands,
        "elapsed_s": elapsed_s,
        "started_at": started_at,
        "recording_id": recording_id,
        "display_db_offset": float(DISPLAY_DB_OFFSET),
    }


def _format_duration(seconds: float | int | None) -> str | None:
    """Human duration: minutes; hours + minutes; or days + hours + minutes."""
    if seconds is None:
        return None
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return None
    if total < 0:
        return None
    if total < 60:
        return "< 1 min" if total > 0 else "0 min"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts: list[str] = []
    if days:
        parts.append(f"{days} d")
    if hours or days:
        parts.append(f"{hours} h")
    parts.append(f"{minutes} min")
    return " ".join(parts)


def _jsonl_list_stats(path: Path) -> tuple[int, datetime | None, datetime | None]:
    """Return (event_count, first_created_at, last_created_at) cheaply.

    Reads only the first and last JSONL records (not the full file). Event count
    prefers ``chunk_index + 1`` from the last record; falls back to a binary
    newline count when that field is missing.
    """
    first_event = _read_jsonl_event(path, last=False)
    last_event = _read_jsonl_event(path, last=True)
    first_at = _parse_event_time(first_event.get("created_at")) if first_event else None
    last_at = _parse_event_time(last_event.get("created_at")) if last_event else None

    lines = 0
    if last_event is not None:
        raw_idx = last_event.get("chunk_index")
        try:
            if raw_idx is not None:
                lines = int(raw_idx) + 1
        except (TypeError, ValueError):
            lines = 0
    if lines <= 0:
        try:
            with path.open("rb") as f:
                lines = sum(1 for ln in f if ln.strip())
        except OSError:
            lines = 0
    return lines, first_at, last_at


def _list_recordings() -> list[dict]:
    """Return past JSONL recordings sorted newest first."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    zone, _, _ = resolve_zone(SITE_TIMEZONE)
    recordings = []
    for p in sorted(RECORDINGS_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        size = p.stat().st_size
        lines, first_at, last_at = _jsonl_list_stats(p) if size > 0 else (0, None, None)

        duration_s = None
        if first_at is not None and last_at is not None:
            duration_s = max(0, int((last_at - first_at).total_seconds()))
            # Single-chunk files: treat as ~1 s rather than 0.
            if duration_s == 0 and lines >= 1:
                duration_s = 1

        if first_at is not None:
            local = to_local(first_at, zone)
            started_label = local.strftime("%Y-%m-%d %H:%M")
            # Include short zone abbrev when available (CET/CEST).
            tz_name = local.tzname() or SITE_TIMEZONE
            started_display = f"{started_label} {tz_name}"
        else:
            # Fallback: file mtime in site timezone.
            mtime_utc = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            local = to_local(mtime_utc, zone)
            tz_name = local.tzname() or SITE_TIMEZONE
            started_display = f"{local.strftime('%Y-%m-%d %H:%M')} {tz_name}"

        audio_path = sibling_audio_path(p)
        has_audio = audio_path is not None
        audio_size = audio_path.stat().st_size if has_audio else 0
        audio_format = (
            audio_path.suffix.lower().lstrip(".") if has_audio and audio_path else None
        )
        entry: dict = {
            "name": p.name,
            "size_label": _format_bytes(size),
            "lines": lines,
            "started_display": started_display,
            "duration_label": _format_duration(duration_s),
            "has_audio": has_audio,
            "audio_name": audio_path.name if has_audio and audio_path else None,
            "audio_format": audio_format,
            "audio_size_label": _format_bytes(audio_size) if has_audio else None,
            # Legacy keys kept for older UI/clients.
            "has_wav": has_audio,
            "wav_name": audio_path.name if has_audio and audio_path else None,
            "wav_size_label": _format_bytes(audio_size) if has_audio else None,
        }
        recordings.append(entry)
    return recordings


def _format_bytes(n: int) -> str:
    """Human-readable byte size (binary units)."""
    size = float(max(0, int(n)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _disk_usage_summary(path: Path | None = None) -> dict:
    """Return free/total disk stats for the filesystem holding ``path``."""
    target = path or RECORDINGS_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target)
    except OSError:
        return {
            "ok": False,
            "free_bytes": 0,
            "total_bytes": 0,
            "used_bytes": 0,
            "free_pct": 0.0,
            "free_label": "—",
            "total_label": "—",
            "line": "Disk space unavailable",
            "level": "unknown",
        }

    free_pct = (100.0 * usage.free / usage.total) if usage.total else 0.0
    # Warn when free space is getting tight for WAV (~330 MB/h) + JSONL.
    if usage.free < 1 * 1024**3 or free_pct < 10.0:
        level = "critical"
    elif usage.free < 4 * 1024**3 or free_pct < 20.0:
        level = "warn"
    else:
        level = "ok"

    free_label = _format_bytes(usage.free)
    total_label = _format_bytes(usage.total)
    return {
        "ok": True,
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_pct": round(free_pct, 1),
        "free_label": free_label,
        "total_label": total_label,
        "line": f"{free_label} free of {total_label} ({free_pct:.0f}%)",
        "level": level,
    }


def _level_from_pct(pct: float | None, *, warn: float, critical: float) -> str:
    if pct is None:
        return "unknown"
    if pct >= critical:
        return "critical"
    if pct >= warn:
        return "warn"
    return "ok"


def _level_from_temp_c(temp_c: float | None) -> str:
    """Pi soft limit ~80°C; sustained ≥85°C is a hard throttle zone."""
    if temp_c is None:
        return "unknown"
    if temp_c >= 80.0:
        return "critical"
    if temp_c >= 70.0:
        return "warn"
    return "ok"


def _read_cpu_temp_c() -> float | None:
    """Best-effort CPU/SoC temperature (°C). Prefer psutil, then sysfs, then vcgencmd."""
    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False)
    except (AttributeError, OSError):
        sensors = None
    if sensors:
        preferred = (
            "cpu_thermal",
            "cpu-thermal",
            "soc_thermal",
            "rp1_adc",
            "coretemp",
            "k10temp",
        )
        for name in preferred:
            entries = sensors.get(name) or []
            for entry in entries:
                current = getattr(entry, "current", None)
                if current is not None:
                    try:
                        return round(float(current), 1)
                    except (TypeError, ValueError):
                        pass
        for entries in sensors.values():
            for entry in entries or []:
                current = getattr(entry, "current", None)
                if current is not None:
                    try:
                        return round(float(current), 1)
                    except (TypeError, ValueError):
                        pass

    thermal_root = Path("/sys/class/thermal")
    if thermal_root.is_dir():
        for zone in sorted(thermal_root.glob("thermal_zone*")):
            type_path = zone / "type"
            temp_path = zone / "temp"
            try:
                zone_type = type_path.read_text(encoding="utf-8").strip().lower()
            except OSError:
                zone_type = ""
            if zone_type and "thermal" not in zone_type and "cpu" not in zone_type and "soc" not in zone_type:
                # Still try cpu_thermal-style names; otherwise read first zone later.
                if zone_type not in {"cpu-thermal", "cpu_thermal", "soc-thermal", "soc_thermal"}:
                    continue
            try:
                milli = float(temp_path.read_text(encoding="utf-8").strip())
                if milli > 1000:
                    return round(milli / 1000.0, 1)
                return round(milli, 1)
            except (OSError, ValueError):
                continue
        # Fallback: first readable zone.
        for zone in sorted(thermal_root.glob("thermal_zone*")):
            try:
                milli = float((zone / "temp").read_text(encoding="utf-8").strip())
                if milli > 1000:
                    return round(milli / 1000.0, 1)
                return round(milli, 1)
            except (OSError, ValueError):
                continue

    vcgencmd = shutil.which("vcgencmd")
    if vcgencmd:
        try:
            result = subprocess.run(
                [vcgencmd, "measure_temp"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            raw = (result.stdout or "").strip()
            # e.g. temp=52.3'C
            if "temp=" in raw:
                num = raw.split("temp=", 1)[1].split("'", 1)[0]
                return round(float(num), 1)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return None


def _pi_throttle_status() -> dict[str, Any] | None:
    """Parse ``vcgencmd get_throttled`` when available (Raspberry Pi)."""
    vcgencmd = shutil.which("vcgencmd")
    if not vcgencmd:
        return None
    try:
        result = subprocess.run(
            [vcgencmd, "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    raw = (result.stdout or "").strip()
    # e.g. throttled=0x50000
    if "throttled=" not in raw:
        return None
    hex_part = raw.split("throttled=", 1)[1].strip()
    try:
        flags = int(hex_part, 0)
    except ValueError:
        return None

    def bit(n: int) -> bool:
        return bool(flags & (1 << n))

    now = {
        "under_voltage": bit(0),
        "freq_capped": bit(1),
        "throttled": bit(2),
        "soft_temp_limit": bit(3),
    }
    since_boot = {
        "under_voltage": bit(16),
        "freq_capped": bit(17),
        "throttled": bit(18),
        "soft_temp_limit": bit(19),
    }
    active = [k for k, v in now.items() if v]
    historical = [k for k, v in since_boot.items() if v]
    if active:
        level = "critical"
    elif historical:
        level = "warn"
    else:
        level = "ok"
    return {
        "raw": hex_part,
        "flags": flags,
        "now": now,
        "since_boot": since_boot,
        "active": active,
        "historical": historical,
        "level": level,
    }


def _default_network_iface() -> str | None:
    """Return the interface used by the default IPv4 route, if known."""
    try:
        result = subprocess.run(
            ["ip", "-4", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        match = re.search(r"\bdev\s+(\S+)", result.stdout or "")
        if match:
            return match.group(1)

    # /proc/net/route: destination 00000000 = default
    try:
        text = Path("/proc/net/route").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "00000000":
            return parts[0]
    return None


def _signal_pct_from_dbm(dbm: float) -> float:
    """Map typical Wi‑Fi RSSI (−90…−30 dBm) onto 0–100 for the meter bar."""
    return round(max(0.0, min(100.0, (float(dbm) + 90.0) / 60.0 * 100.0)), 1)


def _level_from_wifi_dbm(dbm: float | None) -> str:
    if dbm is None:
        return "unknown"
    if dbm < -70.0:
        return "critical"
    if dbm < -60.0:
        return "warn"
    return "ok"


def _level_from_ping_ms(ping_ms: float | None, *, ping_ok: bool) -> str:
    if not ping_ok or ping_ms is None:
        return "critical"
    if ping_ms >= 200.0:
        return "critical"
    if ping_ms >= 80.0:
        return "warn"
    return "ok"


def _wifi_link_info(iface: str) -> dict[str, Any]:
    """Best-effort SSID + RSSI for a wireless interface."""
    info: dict[str, Any] = {
        "ssid": None,
        "signal_dbm": None,
        "kind": "wifi",
    }
    iw = shutil.which("iw")
    if iw:
        try:
            result = subprocess.run(
                [iw, "dev", iface, "link"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            out = result.stdout or ""
            ssid_m = re.search(r"^\s*SSID:\s*(.+)\s*$", out, re.MULTILINE)
            if ssid_m:
                info["ssid"] = ssid_m.group(1).strip() or None
            sig_m = re.search(r"signal:\s*(-?\d+(?:\.\d+)?)\s*dBm", out, re.I)
            if sig_m:
                info["signal_dbm"] = round(float(sig_m.group(1)), 1)
            if "Not connected" in out:
                info["kind"] = "wifi"
            if info["signal_dbm"] is not None or info["ssid"] is not None:
                return info
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass

    # /proc/net/wireless — level is often dBm (negative) on modern kernels.
    try:
        text = Path("/proc/net/wireless").read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        if not line.strip().startswith(f"{iface}:"):
            continue
        # wlan0: 0000  60.  -50.  -256  ...
        nums = re.findall(r"-?\d+(?:\.\d+)?", line.split(":", 1)[1])
        if len(nums) >= 2:
            level = float(nums[1])
            # Some drivers report level as quality-like positive; prefer dBm.
            if level < 0:
                info["signal_dbm"] = round(level, 1)
            elif info["signal_dbm"] is None and level <= 100:
                # Treat 0–100 quality as a pseudo percentage later via signal_pct.
                info["signal_pct_hint"] = round(level, 1)
        break

    iwconfig = shutil.which("iwconfig")
    if iwconfig and info["ssid"] is None:
        try:
            result = subprocess.run(
                [iwconfig, iface],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            out = (result.stdout or "") + (result.stderr or "")
            essid_m = re.search(r'ESSID:"([^"]*)"', out)
            if essid_m:
                info["ssid"] = essid_m.group(1) or None
            if info["signal_dbm"] is None:
                sig_m = re.search(
                    r"Signal level[=:](-?\d+(?:\.\d+)?)\s*dBm", out, re.I
                )
                if sig_m:
                    info["signal_dbm"] = round(float(sig_m.group(1)), 1)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return info


def _iface_kind(iface: str | None) -> str:
    if not iface:
        return "unknown"
    name = iface.lower()
    if name.startswith(("wl", "wlan", "wifi")):
        return "wifi"
    if name.startswith(("en", "eth", "usb")):
        return "ethernet"
    # sysfs type: 1 = ethernet, 801 = wifi
    try:
        type_raw = Path(f"/sys/class/net/{iface}/type").read_text(encoding="utf-8")
        type_n = int(type_raw.strip())
        if type_n == 801:
            return "wifi"
        if type_n == 1:
            return "ethernet"
    except (OSError, ValueError):
        pass
    return "unknown"


def _ping_rtt_ms(host: str = "1.1.1.1") -> tuple[bool, float | None]:
    """One ICMP echo; returns (ok, rtt_ms)."""
    if os.name == "nt":
        cmd = ["ping", "-n", "1", "-w", "1000", host]
    else:
        cmd = ["ping", "-c", "1", "-W", "1", host]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    out = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"time[=<](\d+(?:\.\d+)?)\s*ms", out, re.I)
    if match:
        return True, round(float(match.group(1)), 1)
    return result.returncode == 0, None


def _network_status() -> dict[str, Any]:
    """Wi‑Fi RSSI when available, plus a quick public ping for internet reachability."""
    iface = _default_network_iface()
    kind = _iface_kind(iface)
    ssid = None
    signal_dbm = None
    signal_pct: float | None = None

    if iface and kind == "wifi":
        link = _wifi_link_info(iface)
        ssid = link.get("ssid")
        signal_dbm = link.get("signal_dbm")
        if signal_dbm is not None:
            signal_pct = _signal_pct_from_dbm(float(signal_dbm))
        elif link.get("signal_pct_hint") is not None:
            signal_pct = float(link["signal_pct_hint"])

    ping_ok, ping_ms = _ping_rtt_ms()

    wifi_level = _level_from_wifi_dbm(signal_dbm if signal_dbm is not None else None)
    ping_level = _level_from_ping_ms(ping_ms, ping_ok=ping_ok)

    # Prefer radio strength when on Wi‑Fi; otherwise judge by internet RTT.
    if kind == "wifi" and signal_dbm is not None:
        level = wifi_level
        # Offline internet still marks critical even with a strong AP signal.
        if not ping_ok:
            level = "critical"
        elif ping_level == "warn" and level == "ok":
            level = "warn"
    else:
        level = ping_level

    if signal_pct is None and ping_ok and ping_ms is not None:
        # Invert latency onto the bar (0 ms → full, 200 ms+ → empty).
        signal_pct = round(max(0.0, min(100.0, (1.0 - min(ping_ms, 200.0) / 200.0) * 100.0)), 1)
    elif signal_pct is None and not ping_ok:
        signal_pct = 0.0

    note_bits: list[str] = []
    if iface:
        note_bits.append(iface)
    if kind == "wifi" and ssid:
        note_bits.append(ssid)
    elif kind == "ethernet":
        note_bits.append("ethernet")
    if signal_dbm is not None:
        note_bits.append(f"{signal_dbm:.0f} dBm")
    if ping_ok and ping_ms is not None:
        note_bits.append(f"{ping_ms:.0f} ms to net")
    elif not ping_ok:
        note_bits.append("no internet reply")

    if signal_dbm is not None:
        value_text = f"{signal_dbm:.0f} dBm"
    elif ping_ok and ping_ms is not None:
        value_text = f"{ping_ms:.0f} ms"
    elif not ping_ok:
        value_text = "Offline"
    else:
        value_text = "—"

    return {
        "ok": bool(ping_ok or signal_dbm is not None),
        "iface": iface,
        "kind": kind,
        "ssid": ssid,
        "signal_dbm": signal_dbm,
        "signal_pct": signal_pct,
        "ping_ms": ping_ms,
        "ping_ok": ping_ok,
        "level": level,
        "value_text": value_text,
        "note": " · ".join(note_bits) if note_bits else "Network status unavailable",
    }


# Prime non-blocking CPU percent samples (first call is often 0.0).
try:
    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)
except Exception:  # noqa: BLE001
    pass


def _system_telemetry() -> dict[str, Any]:
    """Snapshot CPU / RAM / temp / disk / network / throttle for the Settings health card."""
    try:
        cpu_pct = float(psutil.cpu_percent(interval=None))
    except Exception:  # noqa: BLE001
        cpu_pct = None
    try:
        per_cpu = [float(x) for x in psutil.cpu_percent(interval=None, percpu=True)]
    except Exception:  # noqa: BLE001
        per_cpu = []

    mem_pct = None
    mem_used_label = None
    mem_total_label = None
    mem_available_label = None
    try:
        mem = psutil.virtual_memory()
        mem_pct = round(float(mem.percent), 1)
        mem_used_label = _format_bytes(int(mem.used))
        mem_total_label = _format_bytes(int(mem.total))
        mem_available_label = _format_bytes(int(mem.available))
    except Exception:  # noqa: BLE001
        pass

    load_1 = load_5 = load_15 = None
    try:
        load_1, load_5, load_15 = os.getloadavg()
    except (AttributeError, OSError):
        pass

    uptime_s = None
    try:
        uptime_s = max(0, int(time.time() - psutil.boot_time()))
    except Exception:  # noqa: BLE001
        pass

    temp_c = _read_cpu_temp_c()
    throttle = _pi_throttle_status()
    disk = _disk_usage_summary(RECORDINGS_DIR)
    network = _network_status()

    cpu_level = _level_from_pct(cpu_pct, warn=70.0, critical=90.0)
    mem_level = _level_from_pct(mem_pct, warn=75.0, critical=90.0)
    temp_level = _level_from_temp_c(temp_c)
    throttle_level = (throttle or {}).get("level", "unknown")
    network_level = network.get("level", "unknown")

    rank = {"ok": 0, "unknown": 0, "warn": 1, "critical": 2}
    overall = "ok"
    for lvl in (
        cpu_level,
        mem_level,
        temp_level,
        disk.get("level"),
        throttle_level,
        network_level,
    ):
        if rank.get(str(lvl), 0) > rank.get(overall, 0):
            overall = str(lvl)

    return {
        "ok": True,
        "hostname": hostname(),
        "cpu_percent": round(cpu_pct, 1) if cpu_pct is not None else None,
        "cpu_count": psutil.cpu_count() or None,
        "cpu_per_core": [round(x, 1) for x in per_cpu],
        "cpu_level": cpu_level,
        "memory_percent": mem_pct,
        "memory_used_label": mem_used_label,
        "memory_total_label": mem_total_label,
        "memory_available_label": mem_available_label,
        "memory_level": mem_level,
        "load_1": round(load_1, 2) if load_1 is not None else None,
        "load_5": round(load_5, 2) if load_5 is not None else None,
        "load_15": round(load_15, 2) if load_15 is not None else None,
        "uptime_s": uptime_s,
        "temp_c": temp_c,
        "temp_level": temp_level,
        "throttle": throttle,
        "disk": disk,
        "network": network,
        "overall_level": overall,
    }


def _active_output_path() -> Path | None:
    """Return the collector ``-o`` path if a recording is active."""
    proc = _find_collector_process()
    if proc is None:
        return None
    try:
        cmd = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    for i, part in enumerate(cmd):
        if part in ("-o", "--output") and i + 1 < len(cmd):
            return Path(cmd[i + 1]).resolve()
    return None


def _safe_recordings_path(filename: str) -> Path | None:
    """Resolve a basename under RECORDINGS_DIR, or None if unsafe/missing parent guard."""
    if not filename or "/" in filename or "\\" in filename or filename in {".", ".."}:
        return None
    path = (RECORDINGS_DIR / filename).resolve()
    if not str(path).startswith(str(RECORDINGS_DIR.resolve())):
        return None
    return path


def _iter_jsonl_events(path: Path) -> list[dict]:
    """Load all JSONL events from ``path`` (skip bad lines)."""
    events: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return events


# Soft cache of slim overview points keyed by resolved path + mtime (Inspect window series).
_slim_points_cache: dict[str, tuple[float, list[dict]]] = {}
_slim_points_cache_lock = threading.Lock()


def _cached_slim_points(path: Path, *, events: list[dict] | None = None) -> list[dict]:
    """Return slim chart points for ``path``, caching by mtime."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    key = str(path.resolve())
    with _slim_points_cache_lock:
        hit = _slim_points_cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    src = events if events is not None else _iter_jsonl_events(path)
    # Cache raw levels; display offset is applied at serve time so Settings changes apply immediately.
    slim = slim_points_from_events(
        src,
        calib_offset=float(DEFAULT_CALIB_OFFSET),
        display_db_offset=0.0,
    )
    with _slim_points_cache_lock:
        _slim_points_cache[key] = (mtime, slim)
        # Bound cache size on multi-recording Pis.
        if len(_slim_points_cache) > 8:
            oldest = next(iter(_slim_points_cache))
            if oldest != key:
                _slim_points_cache.pop(oldest, None)
    return slim


def _recording_overview_payload(path: Path, *, loud_threshold: float) -> dict:
    audio_path = sibling_audio_path(path)
    has_audio = audio_path is not None
    events = _iter_jsonl_events(path)
    chunk_seconds = float(CAPTURE_CHUNK_SAMPLES) / float(CAPTURE_SAMPLE_RATE)
    audio_duration_s = None
    if has_audio and audio_path is not None and audio_path.suffix.lower() == ".flac":
        fallback = len(events) * int(CAPTURE_CHUNK_SAMPLES) if events else None
        ensure_flac_seekable(audio_path, fallback_samples=fallback)
        total = read_flac_total_samples(audio_path)
        if total is not None and total > 0:
            audio_duration_s = float(total) / float(CAPTURE_SAMPLE_RATE)
    if audio_duration_s is None and has_audio and events:
        audio_duration_s = float(len(events)) * chunk_seconds
    active = _active_output_path()
    recording_active = active is not None and path.resolve() == active
    # Warm slim cache so the first Inspect window fetch is cheap.
    _cached_slim_points(path, events=events)
    return build_recording_overview(
        events,
        name=path.name,
        has_wav=has_audio,
        wav_name=audio_path.name if has_audio and audio_path else None,
        recording_active=recording_active,
        loud_threshold=loud_threshold,
        calib_offset=float(DEFAULT_CALIB_OFFSET),
        display_db_offset=float(DISPLAY_DB_OFFSET),
        audio_duration_s=audio_duration_s,
        audio_chunk_seconds=chunk_seconds,
    )


def _recording_overview_series_payload(
    path: Path,
    *,
    center_ms: float,
    window_s: float,
) -> dict:
    slim = _shift_slim_db_levels(_cached_slim_points(path), float(DISPLAY_DB_OFFSET))
    payload = build_overview_window_series(
        slim,
        center_ms=center_ms,
        window_s=window_s,
    )
    payload["display_db_offset"] = float(DISPLAY_DB_OFFSET)
    return payload


# Fields required for report aggregation (spectrum etc. are dropped to save RAM).
_REPORT_EVENT_KEYS = (
    "created_at",
    "device_id",
    "dBA_spl",
    "LAFmax_dB",
    "top_label",
    "top_confidence",
    "clap_status",
)


def _slim_event_for_report(event: dict) -> dict:
    """Keep only fields needed by dashboard aggregation."""
    out = {k: event[k] for k in _REPORT_EVENT_KEYS if k in event}
    prep = event.get("yamnet_preprocess")
    if isinstance(prep, dict) and "gated" in prep:
        out["yamnet_preprocess"] = {"gated": bool(prep.get("gated"))}
    return out


def _iter_jsonl_events_for_report(path: Path) -> list[dict]:
    """Load JSONL events for reports without bulky spectrum / CLAP payloads."""
    events: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, dict):
                    events.append(_slim_event_for_report(raw))
    except OSError:
        return []
    return events


def _report_error_payload(
    *,
    code: str,
    message: str,
    **extra: object,
) -> dict:
    return {
        "stage": "error",
        "ok": False,
        "code": code,
        "error": message,
        **extra,
    }


def _build_report_ndjson(
    files: list[str],
    *,
    min_confidence: float | None,
    min_event_chunks: int | None = None,
    max_gap_chunks: int | None = None,
    threshold_mode: str | None = None,
    threshold_db: float | None = None,
    l90_offset_db: float | None = None,
) -> Iterator[str]:
    """Yield NDJSON progress lines, then a final done/error object."""

    def _line(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    total = len(files)
    yield _line(
        {
            "stage": "start",
            "ok": True,
            "message": f"Preparing report for {total} recording(s)…",
            "total": total,
        }
    )

    recording_events: list[tuple[str, list]] = []
    total_bytes = 0
    try:
        for index, name in enumerate(files, start=1):
            path = _safe_recordings_path(name)
            if path is None or not path.exists() or not path.is_file():
                yield _line(
                    _report_error_payload(
                        code="not_found",
                        message=f"Recording not found: {name}",
                        file=name,
                    )
                )
                return
            try:
                size = path.stat().st_size
            except OSError as exc:
                yield _line(
                    _report_error_payload(
                        code="io_error",
                        message=f"Could not read {name}: {exc}",
                        file=name,
                    )
                )
                return
            total_bytes += size
            yield _line(
                {
                    "stage": "loading",
                    "ok": True,
                    "message": (
                        f"Loading {name} ({index}/{total}, "
                        f"{_format_bytes(size)})…"
                    ),
                    "file": name,
                    "index": index,
                    "total": total,
                    "bytes": size,
                }
            )
            try:
                events = _iter_jsonl_events_for_report(path)
            except MemoryError:
                yield _line(
                    _report_error_payload(
                        code="oom",
                        message=(
                            "The Pi ran out of memory while loading recordings. "
                            "Select fewer/shorter files and try again."
                        ),
                        file=name,
                    )
                )
                return
            recording_events.append((path.name, events))
            yield _line(
                {
                    "stage": "loaded",
                    "ok": True,
                    "message": f"Loaded {len(events):,} chunks from {name}",
                    "file": name,
                    "index": index,
                    "total": total,
                    "chunks": len(events),
                }
            )

        yield _line(
            {
                "stage": "aggregating",
                "ok": True,
                "message": "Computing Leq, timeline, events, and sound diet…",
                "bytes_total": total_bytes,
            }
        )

        try:
            payload = build_dashboard_report(
                recording_events,
                timezone_name=SITE_TIMEZONE,
                site_label=SITE_LABEL or None,
                min_confidence=min_confidence,
                min_event_chunks=min_event_chunks,
                max_gap_chunks=max_gap_chunks,
                threshold_mode=threshold_mode,
                threshold_db=threshold_db,
                l90_offset_db=l90_offset_db,
                display_db_offset=float(DISPLAY_DB_OFFSET),
            )
        except MemoryError:
            yield _line(
                _report_error_payload(
                    code="oom",
                    message=(
                        "The Pi ran out of memory while aggregating the report. "
                        "Select fewer/shorter files and try again."
                    ),
                )
            )
            return
        except Exception as exc:  # noqa: BLE001
            yield _line(
                _report_error_payload(
                    code="aggregate_failed",
                    message=f"Report aggregation failed: {exc}",
                )
            )
            return

        if not payload.get("ok", True):
            yield _line(
                _report_error_payload(
                    code="aggregate_failed",
                    message=str(payload.get("error") or "Report aggregation failed"),
                )
            )
            return

        yield _line(
            {
                "stage": "done",
                "ok": True,
                "message": "Report ready",
                "report": payload,
            }
        )
    except MemoryError:
        yield _line(
            _report_error_payload(
                code="oom",
                message=(
                    "The Pi ran out of memory while building the report. "
                    "Select fewer/shorter files and try again."
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        yield _line(
            _report_error_payload(
                code="internal",
                message=f"Unexpected error while building report: {exc}",
            )
        )


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def get_login():
    return login_page()


@app.post("/login")
async def post_login(
    password: str = Form(...),
    role: str = Form("admin"),
):
    matched = verify_login(role, password)
    if matched is None:
        if role == "viewer" and not VIEWER_PASSWORD:
            return login_page(
                error="Viewer login is not configured on this device.",
                role="viewer",
            )
        return login_page(error="Incorrect password.", role=role if role in ("admin", "viewer") else "admin")
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(
        SESSION_COOKIE,
        make_session_cookie(password, matched),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
    )
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def _index_context(
    request: Request,
    *,
    analyze_recording: str | None = None,
    report_recordings: list[str] | None = None,
) -> dict:
    status = _get_status()
    recordings = _list_recordings()
    error = request.query_params.get("error", "")
    return dict(
        request=request,
        status=status,
        recordings=recordings,
        disk=_disk_usage_summary(RECORDINGS_DIR),
        default_device_id=DEFAULT_DEVICE_ID,
        default_alsa_device=DEFAULT_ALSA_DEVICE,
        default_recording_name=_default_recording_name(),
        default_gate_sensitivity_db=DEFAULT_GATE_SENSITIVITY_DB,
        display_db_offset=DISPLAY_DB_OFFSET,
        audio_format=AUDIO_FORMAT,
        device_id=DEFAULT_DEVICE_ID,
        pi_hostname=hostname(),
        site_label=SITE_LABEL,
        public_url=PUBLIC_URL,
        site_timezone=SITE_TIMEZONE,
        user_role=get_role(request) or "viewer",
        error=error,
        analyze_recording=analyze_recording,
        report_recordings=report_recordings or [],
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    return _render("index.html", **_index_context(request))


@app.get("/analyze/{filename}", response_class=HTMLResponse)
async def analyze_recording_page(filename: str, request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    if not filename.lower().endswith(".jsonl"):
        return RedirectResponse("/?tab=data&error=bad_request", status_code=303)
    path = _safe_recordings_path(filename)
    if path is None or not path.exists() or not path.is_file():
        return RedirectResponse("/?tab=data&error=not_found", status_code=303)
    return _render("index.html", **_index_context(request, analyze_recording=path.name))


@app.get("/report", response_class=HTMLResponse)
async def report_page(request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    raw = request.query_params.get("recordings") or request.query_params.get("runs", "")
    names = [n.strip() for n in raw.split(",") if n.strip()]
    safe: list[str] = []
    for name in names[:32]:
        if not name.lower().endswith(".jsonl"):
            continue
        path = _safe_recordings_path(name)
        if path is not None and path.is_file():
            safe.append(path.name)
    if not safe:
        return RedirectResponse("/?tab=data&error=bad_request", status_code=303)
    return _render("index.html", **_index_context(request, report_recordings=safe))


# ---------------------------------------------------------------------------
# Control API
# ---------------------------------------------------------------------------

def _wants_json(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    return "application/json" in accept


def _require_admin_json(request: Request) -> JSONResponse | None:
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if not is_admin(request):
        return JSONResponse(
            {"ok": False, "error": "forbidden", "code": "viewer_readonly"},
            status_code=403,
        )
    return None


def _forbid_viewer_redirect(
    request: Request, *, tab: str | None = None
) -> RedirectResponse | None:
    """For form POSTs: send viewers back to the UI instead of mutating."""
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    if is_admin(request):
        return None
    q = "error=viewer_readonly"
    if tab:
        q = f"tab={tab}&{q}"
    return RedirectResponse(f"/?{q}", status_code=303)


@app.post("/api/start")
async def api_start(
    request: Request,
    hours: float = Form(8.0),
    recording_name: str = Form("recording"),
    device_id: str = Form(DEFAULT_DEVICE_ID),
    alsa_device: str = Form(DEFAULT_ALSA_DEVICE),
    gate_sensitivity_db: float = Form(DEFAULT_GATE_SENSITIVITY_DB),
    enable_clap: str = Form(""),
    record_audio: str = Form(""),
    audio_format: str = Form(""),
):
    wants_json = _wants_json(request)
    if not is_authenticated(request):
        if wants_json:
            return JSONResponse(
                {"ok": False, "code": "unauthorized", "error": "Sign in required"},
                status_code=401,
            )
        return RedirectResponse("/login", status_code=303)
    if not is_admin(request):
        if wants_json:
            return JSONResponse(
                {
                    "ok": False,
                    "code": "viewer_readonly",
                    "error": "Disabled for viewers",
                },
                status_code=403,
            )
        return RedirectResponse("/?error=viewer_readonly", status_code=303)

    if _find_collector_process():
        if wants_json:
            return JSONResponse(
                {
                    "ok": False,
                    "code": "already_recording",
                    "error": "A recording is already active. Stop it before starting a new one.",
                },
                status_code=409,
            )
        return RedirectResponse("/?error=already_recording", status_code=303)

    try:
        check_capture_device_available(alsa_device)
    except CaptureDeviceError as exc:
        message = str(exc)
        if wants_json:
            return JSONResponse(
                {
                    "ok": False,
                    "code": "no_capture_device",
                    "error": message,
                },
                status_code=400,
            )
        return RedirectResponse("/?error=no_capture_device", status_code=303)

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_recording_name(recording_name)
    output = RECORDINGS_DIR / f"{safe_name}-{_utc_now()}.jsonl"
    duration = _hours_to_timeout(hours)
    clap_on = enable_clap in {"1", "true", "on", "yes"}
    audio_on = record_audio in {"1", "true", "on", "yes"}
    fmt = normalize_audio_format(audio_format or AUDIO_FORMAT)
    cmd = [
        PYTHON, str(MAIN_PY),
        "--device-id", device_id,
        "--alsa-device", alsa_device,
        "--backend", "arecord",
        "--quiet",
        "-o", str(output),
        "--yamnet-gate-sensitivity-db", str(gate_sensitivity_db),
    ]
    if clap_on:
        cmd.append("--enable-clap")
    if audio_on:
        cmd.extend(["--record-audio", "--audio-format", fmt])
    full_cmd = ["timeout", duration] + cmd

    subprocess.Popen(
        full_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from web server process group
    )
    if wants_json:
        return JSONResponse(
            {
                "ok": True,
                "code": "started",
                "message": "Collector process started",
                "output_file": str(output),
                "enable_clap": clap_on,
                "record_audio": audio_on,
                "audio_format": fmt if audio_on else None,
                "startup_discard_s": float(CAPTURE_STARTUP_DISCARD_SECONDS),
            }
        )
    return RedirectResponse("/", status_code=303)


@app.post("/api/stop")
async def api_stop(request: Request):
    blocked = _forbid_viewer_redirect(request)
    if blocked:
        return blocked

    _stop_collector()
    return RedirectResponse("/", status_code=303)


@app.post("/api/shutdown")
async def api_shutdown(request: Request):
    denied = _require_admin_json(request)
    if denied:
        return denied

    stopped_recording = _stop_collector()
    _schedule_immediate_poweroff()

    return JSONResponse(
        {
            "ok": True,
            "stopped_recording": stopped_recording,
            "hostname": hostname(),
            "message": (
                "Shutdown started. Keep this page open until it reports the "
                "Pi is offline, then unplug power."
            ),
        }
    )


@app.get("/api/status")
async def api_status(request: Request):
    if not is_authenticated(request):
        return HTMLResponse("", status_code=401)
    return JSONResponse(_get_status())


@app.get("/api/system")
async def api_system(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return JSONResponse(_system_telemetry())


@app.get("/api/status/series")
async def api_status_series(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    raw = request.query_params.get("window_s", "120")
    try:
        window_s = int(raw)
    except (TypeError, ValueError):
        window_s = 120
    return JSONResponse(_status_series(window_s=window_s))


@app.get("/api/settings/display-db-offset")
async def api_get_display_db_offset(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return JSONResponse(
        {"ok": True, "display_db_offset": float(DISPLAY_DB_OFFSET)}
    )


@app.put("/api/settings/display-db-offset")
async def api_put_display_db_offset(request: Request):
    denied = _require_admin_json(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)
    raw = body.get("display_db_offset", body.get("offset"))
    try:
        offset = _set_display_db_offset(float(raw))
    except (TypeError, ValueError):
        return JSONResponse(
            {"ok": False, "error": "bad_request", "message": "Invalid offset"},
            status_code=400,
        )
    return JSONResponse({"ok": True, "display_db_offset": offset})


@app.get("/api/settings/audio-format")
async def api_get_audio_format(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return JSONResponse({"ok": True, "audio_format": AUDIO_FORMAT})


@app.put("/api/settings/audio-format")
async def api_put_audio_format(request: Request):
    denied = _require_admin_json(request)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)
    raw = body.get("audio_format", body.get("format"))
    text = str(raw or "").strip().lower()
    if text not in {"flac", "wav"}:
        return JSONResponse(
            {
                "ok": False,
                "error": "bad_request",
                "message": "audio_format must be flac or wav",
            },
            status_code=400,
        )
    fmt = _set_audio_format(text)
    return JSONResponse({"ok": True, "audio_format": fmt})


# ---------------------------------------------------------------------------
# Live log tail via Server-Sent Events
# ---------------------------------------------------------------------------

@app.get("/api/log-stream")
async def log_stream(request: Request):
    if not is_authenticated(request):
        return HTMLResponse("", status_code=401)

    async def event_generator() -> AsyncIterator[str]:
        import asyncio

        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        current_log: Path | None = None
        file_pos = 0

        while True:
            if await request.is_disconnected():
                break

            # Pick the newest log file
            logs = sorted(
                LOGS_DIR.glob("*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            newest = logs[0] if logs else None

            if newest != current_log:
                current_log = newest
                file_pos = 0

            if current_log and current_log.exists():
                with current_log.open(encoding="utf-8", errors="replace") as f:
                    f.seek(file_pos)
                    new_lines = f.readlines()
                    file_pos = f.tell()

                for line in new_lines:
                    line = line.rstrip()
                    if line:
                        yield f"data: {json.dumps(line)}\n\n"

            await asyncio.sleep(3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def _query_flag(request: Request, name: str, *, default: bool = True) -> bool:
    """Parse a boolean query flag (1/true/yes/on vs 0/false/no/off)."""
    raw = request.query_params.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@app.get("/recordings/{filename}")
async def download_recording(filename: str, request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    path = _safe_recordings_path(filename)
    if path is None or not path.exists() or not path.is_file():
        return HTMLResponse("Not found", status_code=404)

    # Audio (or any non-JSONL) — serve for playback or download.
    if path.suffix.lower() != ".jsonl":
        suffix = path.suffix.lower()
        if suffix == ".flac":
            # Unfinalized FLACs (hard stop) have total_samples=0 and won't seek in browsers.
            jsonl_sibling = path.with_suffix(".jsonl")
            fallback = None
            if jsonl_sibling.is_file():
                try:
                    # Cheap line count ≈ event/chunk count for sync repair.
                    with jsonl_sibling.open("rb") as f:
                        n_lines = sum(1 for line in f if line.strip())
                    if n_lines > 0:
                        fallback = int(n_lines) * int(CAPTURE_CHUNK_SAMPLES)
                except OSError:
                    fallback = None
            ensure_flac_seekable(path, fallback_samples=fallback)
            media = "audio/flac"
        elif suffix == ".wav":
            media = "audio/wav"
        else:
            media = "application/octet-stream"
        force_download = _query_flag(request, "download", default=False)
        # Omit filename= so browsers can stream/play inline (Range seeks work).
        if force_download:
            return FileResponse(
                path,
                filename=path.name,
                media_type=media,
            )
        return FileResponse(path, media_type=media)

    include_spectrum = _query_flag(request, "spectrum", default=True)
    include_yamnet_preprocess = _query_flag(
        request, "yamnet_preprocess", default=True
    )
    download_name = export_filename(
        filename,
        include_spectrum=include_spectrum,
        include_yamnet_preprocess=include_yamnet_preprocess,
    )

    # Full export: stream the original file unchanged.
    if include_spectrum and include_yamnet_preprocess:
        return FileResponse(
            path,
            filename=download_name,
            media_type="application/octet-stream",
        )

    def _stream() -> Iterator[str]:
        yield from iter_filtered_jsonl(
            path,
            include_spectrum=include_spectrum,
            include_yamnet_preprocess=include_yamnet_preprocess,
        )

    headers = {
        "Content-Disposition": f'attachment; filename="{download_name}"',
    }
    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers=headers,
    )


@app.get("/api/recordings/{filename}/overview")
async def api_recording_overview(filename: str, request: Request):
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if not filename.lower().endswith(".jsonl"):
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)
    path = _safe_recordings_path(filename)
    if path is None or not path.exists() or not path.is_file():
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)

    raw_thr = request.query_params.get("loud_threshold", str(DEFAULT_LOUD_THRESHOLD_LAFMAX))
    try:
        thr = float(raw_thr)
    except (TypeError, ValueError):
        thr = DEFAULT_LOUD_THRESHOLD_LAFMAX
    thr = max(50.0, min(80.0, thr))
    return JSONResponse(_recording_overview_payload(path, loud_threshold=thr))


@app.get("/api/recordings/{filename}/overview/series")
async def api_recording_overview_series(filename: str, request: Request):
    """Full-resolution slim points for a ~2 min window around ``t`` / ``t_ms``."""
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if not filename.lower().endswith(".jsonl"):
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)
    path = _safe_recordings_path(filename)
    if path is None or not path.exists() or not path.is_file():
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)

    raw_window = request.query_params.get("window_s", str(DEFAULT_SERIES_WINDOW_S))
    try:
        window_s = float(raw_window)
    except (TypeError, ValueError):
        window_s = float(DEFAULT_SERIES_WINDOW_S)
    window_s = max(30.0, min(900.0, window_s))

    center_ms: float | None = None
    raw_ms = request.query_params.get("t_ms")
    if raw_ms is not None and str(raw_ms).strip() != "":
        try:
            center_ms = float(raw_ms)
        except (TypeError, ValueError):
            center_ms = None
    if center_ms is None:
        raw_t = request.query_params.get("t")
        center_ms = created_at_to_ms(raw_t) if raw_t else None
    if center_ms is None or not (center_ms == center_ms):  # NaN guard
        return JSONResponse(
            {"ok": False, "error": "bad_request", "message": "Missing or invalid t / t_ms"},
            status_code=400,
        )

    payload = _recording_overview_series_payload(
        path,
        center_ms=center_ms,
        window_s=window_s,
    )
    return JSONResponse(payload)


@app.post("/api/recordings/report")
async def api_recordings_report(request: Request):
    if not is_authenticated(request):
        return JSONResponse(
            _report_error_payload(code="unauthorized", message="Sign in required"),
            status_code=401,
        )
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            _report_error_payload(code="bad_request", message="Invalid JSON body"),
            status_code=400,
        )
    if not isinstance(body, dict):
        return JSONResponse(
            _report_error_payload(code="bad_request", message="Invalid request body"),
            status_code=400,
        )

    files_raw = body.get("files") or []
    if not isinstance(files_raw, list) or not files_raw:
        return JSONResponse(
            _report_error_payload(code="no_files", message="No recordings selected"),
            status_code=400,
        )
    if len(files_raw) > 32:
        return JSONResponse(
            _report_error_payload(
                code="too_many_files",
                message="Too many recordings (max 32). Select fewer files.",
            ),
            status_code=400,
        )

    files: list[str] = []
    for raw_name in files_raw:
        if not isinstance(raw_name, str):
            return JSONResponse(
                _report_error_payload(
                    code="bad_request",
                    message="Each file name must be a string",
                ),
                status_code=400,
            )
        name = raw_name.strip()
        if not name.lower().endswith(".jsonl"):
            return JSONResponse(
                _report_error_payload(
                    code="bad_request",
                    message=f"Not a JSONL recording: {name}",
                    file=name,
                ),
                status_code=400,
            )
        files.append(name)

    min_conf = body.get("min_confidence")
    try:
        min_confidence = float(min_conf) if min_conf is not None else None
    except (TypeError, ValueError):
        min_confidence = None

    def _opt_nonneg_int(raw: Any) -> int | None:
        if raw is None or raw == "":
            return None
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return None

    def _opt_float(raw: Any) -> float | None:
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    min_event_chunks = _opt_nonneg_int(body.get("min_event_chunks"))
    max_gap_chunks = _opt_nonneg_int(body.get("max_gap_chunks"))
    threshold_mode_raw = body.get("threshold_mode")
    threshold_mode = (
        str(threshold_mode_raw).strip().lower()
        if isinstance(threshold_mode_raw, str) and threshold_mode_raw.strip()
        else None
    )
    if threshold_mode not in {None, "absolute", "l90_offset"}:
        threshold_mode = None
    threshold_db = _opt_float(body.get("threshold_db"))
    l90_offset_db = _opt_float(body.get("l90_offset_db"))

    return StreamingResponse(
        _build_report_ndjson(
            files,
            min_confidence=min_confidence,
            min_event_chunks=min_event_chunks,
            max_gap_chunks=max_gap_chunks,
            threshold_mode=threshold_mode,
            threshold_db=threshold_db,
            l90_offset_db=l90_offset_db,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _delete_recording_by_name(filename: str) -> dict[str, str]:
    """Delete a JSONL (+ sibling FLAC/WAV). Returns {name, status} or {name, status, error}."""
    name = (filename or "").strip()
    if not name.lower().endswith(".jsonl"):
        return {"name": name, "status": "error", "error": "bad_request"}
    path = _safe_recordings_path(name)
    if path is None:
        return {"name": name, "status": "error", "error": "bad_request"}
    if not path.exists() or not path.is_file():
        return {"name": name, "status": "error", "error": "not_found"}

    active = _active_output_path()
    if active is not None and path.resolve() == active:
        return {"name": name, "status": "error", "error": "recording_in_use"}

    audio_path = sibling_audio_path(path)
    # Also clean up the other extension if both somehow exist.
    extras = [
        path.with_suffix(".flac"),
        path.with_suffix(".wav"),
    ]
    try:
        path.unlink()
        deleted: set[Path] = set()
        if audio_path is not None and audio_path.is_file():
            audio_path.unlink()
            deleted.add(audio_path.resolve())
        for extra in extras:
            try:
                resolved = extra.resolve()
            except OSError:
                continue
            if resolved in deleted:
                continue
            if extra.is_file():
                extra.unlink()
    except OSError:
        return {"name": name, "status": "error", "error": "bad_request"}
    return {"name": name, "status": "deleted"}


@app.post("/api/recordings/delete")
async def api_recordings_delete(
    request: Request,
    filename: str = Form(...),
):
    blocked = _forbid_viewer_redirect(request, tab="data")
    if blocked:
        return blocked

    result = _delete_recording_by_name(filename)
    if result.get("status") == "deleted":
        return RedirectResponse("/?tab=data", status_code=303)
    err = result.get("error") or "bad_request"
    return RedirectResponse(f"/?tab=data&error={err}", status_code=303)


@app.post("/api/recordings/delete-bulk")
async def api_recordings_delete_bulk(request: Request):
    """Admin-only bulk delete; requires re-typing the admin password."""
    denied = _require_admin_json(request)
    if denied:
        return denied

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)

    password = body.get("password")
    if not isinstance(password, str) or verify_login("admin", password) is None:
        return JSONResponse(
            {"ok": False, "error": "bad_password", "message": "Incorrect admin password"},
            status_code=403,
        )

    files_raw = body.get("files") or []
    if not isinstance(files_raw, list) or not files_raw:
        return JSONResponse(
            {"ok": False, "error": "no_files", "message": "No recordings selected"},
            status_code=400,
        )
    if len(files_raw) > 64:
        return JSONResponse(
            {
                "ok": False,
                "error": "too_many_files",
                "message": "Too many recordings (max 64). Select fewer files.",
            },
            status_code=400,
        )

    deleted: list[str] = []
    failed: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_name in files_raw:
        if not isinstance(raw_name, str):
            failed.append({"name": str(raw_name), "error": "bad_request"})
            continue
        name = raw_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result = _delete_recording_by_name(name)
        if result.get("status") == "deleted":
            deleted.append(name)
        else:
            failed.append(
                {
                    "name": name,
                    "error": str(result.get("error") or "bad_request"),
                }
            )

    return JSONResponse(
        {
            "ok": len(failed) == 0,
            "deleted": deleted,
            "failed": failed,
            "deleted_count": len(deleted),
            "failed_count": len(failed),
        }
    )


# ---------------------------------------------------------------------------
# CLAP config APIs (prompts + triggers + catalog)
# ---------------------------------------------------------------------------

def _require_auth_json(request: Request) -> JSONResponse | None:
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return None


def _require_admin_for_clap_write(request: Request) -> JSONResponse | None:
    denied = _require_auth_json(request)
    if denied:
        return denied
    return _require_admin_json(request)


@app.get("/api/clap/prompts")
async def api_clap_prompts_get(request: Request):
    denied = _require_auth_json(request)
    if denied:
        return denied
    from core.clap_prompts import embedding_sync_status, load_prompt_pairs
    from core.clap_onnx import onnx_ready_status

    pairs = load_prompt_pairs(CLAP_PROMPTS_PATH) if CLAP_PROMPTS_PATH.exists() else []
    sync = embedding_sync_status(pairs, prompts_path=CLAP_PROMPTS_PATH)
    return JSONResponse(
        {
            "ok": True,
            "prompts": [{"label": p.label, "prompt": p.prompt} for p in pairs],
            "sync": sync,
            "onnx": onnx_ready_status(),
        }
    )


@app.put("/api/clap/prompts")
async def api_clap_prompts_put(request: Request):
    denied = _require_admin_for_clap_write(request)
    if denied:
        return denied
    from core.clap_prompts import ClapPromptPair, save_prompt_pairs

    body = await request.json()
    raw = body.get("prompts", [])
    try:
        pairs = [
            ClapPromptPair(label=str(x["label"]).strip(), prompt=str(x["prompt"]).strip())
            for x in raw
        ]
        save_prompt_pairs(pairs, CLAP_PROMPTS_PATH)
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, "count": len(pairs)})


@app.post("/api/clap/embeddings/rebuild")
async def api_clap_embeddings_rebuild(request: Request):
    denied = _require_admin_for_clap_write(request)
    if denied:
        return denied
    from core.classifier_clap import rebuild_text_embeddings

    try:
        result = rebuild_text_embeddings(prompts_path=CLAP_PROMPTS_PATH)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse(result)


@app.get("/api/clap/triggers")
async def api_clap_triggers_get(request: Request):
    denied = _require_auth_json(request)
    if denied:
        return denied
    from core.clap_trigger import load_trigger_config

    cfg = load_trigger_config(CLAP_TRIGGERS_PATH)
    return JSONResponse(
        {
            "ok": True,
            "cooldown_seconds": cfg.cooldown_seconds,
            "dba_threshold": cfg.dba_threshold,
            "lookback_seconds": cfg.lookback_seconds,
            "pre_onset_pad_ms": cfg.pre_onset_pad_ms,
            "end_settle_chunks": cfg.end_settle_chunks,
            "max_event_seconds": cfg.max_event_seconds,
            "onset_dba_margin_db": cfg.onset_dba_margin_db,
            "peak_decay_db": cfg.peak_decay_db,
            "trigger_labels": cfg.trigger_labels,
            "ambiguous_labels": cfg.ambiguous_labels,
        }
    )


@app.put("/api/clap/triggers")
async def api_clap_triggers_put(request: Request):
    denied = _require_admin_for_clap_write(request)
    if denied:
        return denied
    from core.clap_trigger import ClapTriggerConfig, save_trigger_config

    body = await request.json()
    try:
        cfg = ClapTriggerConfig(
            cooldown_seconds=float(body.get("cooldown_seconds", 5)),
            dba_threshold=float(body.get("dba_threshold", 55)),
            lookback_seconds=float(body.get("lookback_seconds", 4)),
            pre_onset_pad_ms=float(body.get("pre_onset_pad_ms", 150)),
            end_settle_chunks=int(body.get("end_settle_chunks", 2)),
            max_event_seconds=float(body.get("max_event_seconds", 5)),
            onset_dba_margin_db=float(body.get("onset_dba_margin_db", 3)),
            peak_decay_db=float(body.get("peak_decay_db", 5)),
            trigger_labels=list(body.get("trigger_labels", [])),
            ambiguous_labels=list(body.get("ambiguous_labels", [])),
        )
        save_trigger_config(cfg, CLAP_TRIGGERS_PATH)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True})


@app.get("/api/clap/yamnet-catalog")
async def api_yamnet_catalog(request: Request):
    denied = _require_auth_json(request)
    if denied:
        return denied
    if not YAMNET_CATALOG_PATH.exists():
        return JSONResponse(
            {"ok": False, "error": "catalog missing; run sync script"},
            status_code=404,
        )
    data = json.loads(YAMNET_CATALOG_PATH.read_text(encoding="utf-8"))
    return JSONResponse({"ok": True, **data})


@app.post("/api/clap/yamnet-catalog/sync")
async def api_yamnet_catalog_sync(request: Request):
    denied = _require_admin_for_clap_write(request)
    if denied:
        return denied
    script = REPO_ROOT / "scripts" / "sync_yamnet_label_catalog.py"
    # Prefer local CSV (always available); upstream optional via query.
    use_upstream = _query_flag(request, "upstream", default=False)
    cmd = [PYTHON, str(script)]
    if use_upstream:
        cmd.append("--from-upstream")
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    if completed.returncode != 0:
        return JSONResponse(
            {
                "ok": False,
                "error": completed.stderr.strip() or completed.stdout.strip() or "sync failed",
            },
            status_code=500,
        )
    data = json.loads(YAMNET_CATALOG_PATH.read_text(encoding="utf-8"))
    return JSONResponse(
        {
            "ok": True,
            "label_count": len(data.get("labels", [])),
            "themes": data.get("themes", []),
            "synced_at": data.get("synced_at"),
            "log": completed.stdout.strip(),
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=PORT, reload=False)
