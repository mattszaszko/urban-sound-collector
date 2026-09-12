"""FastAPI web interface for the Urban Sound Collector."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterator

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

from core.export_filter import export_filename, iter_filtered_jsonl
from core.host_identity import default_device_id, hostname
from core.loudness import DEFAULT_CALIB_OFFSET
from core.yamnet_preprocess import DEFAULT_GATE_SENSITIVITY_DB

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
RUNS_DIR = REPO_ROOT / "runs"
LOGS_DIR = REPO_ROOT / "logs"
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
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-please")
PORT = int(os.environ.get("PORT", "8080"))
DEFAULT_DEVICE_ID = default_device_id()
DEFAULT_ALSA_DEVICE = os.environ.get(
    "ALSA_DEVICE", "plughw:CARD=sndrpigooglevoi,DEV=0"
)
SITE_LABEL = os.environ.get("SITE_LABEL", "").strip()
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").strip()
SHUTDOWN_GRACE_SEC = max(15, int(os.environ.get("SHUTDOWN_GRACE_SEC", "60")))

os.environ["SECRET_KEY"] = SECRET_KEY

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
from web.auth import (  # noqa: E402  (after env setup)
    SESSION_COOKIE,
    is_authenticated,
    login_page,
    make_session_cookie,
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


def _default_run_name() -> str:
    """Suggest a sensible prefix from the current UTC hour."""
    hour = datetime.now(timezone.utc).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "day"
    if 18 <= hour < 23:
        return "evening"
    return "night"


def _sanitize_run_name(name: str) -> str:
    """Keep only safe filename characters; fall back to 'run'."""
    cleaned = "".join(c for c in name.strip() if c.isalnum() or c in "-_")
    return cleaned or "run"


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
    created = event.get("created_at")
    if not isinstance(created, str):
        return None
    dba = event.get("dBA_spl")
    try:
        dba_f = float(dba) if dba is not None else None
    except (TypeError, ValueError):
        dba_f = None

    prep = event.get("yamnet_preprocess")
    l90_rel = None
    gated = False
    if isinstance(prep, dict):
        floor = prep.get("ambient_noise_floor_dbfs")
        try:
            if floor is not None:
                l90_rel = round(float(floor) + float(calib_offset), 1)
        except (TypeError, ValueError):
            l90_rel = None
        gated = bool(prep.get("gated"))

    yamnet = event.get("top_label")
    if not isinstance(yamnet, str):
        yamnet = None
    if yamnet == "gated":
        gated = True

    clap_status = event.get("clap_status")
    if not isinstance(clap_status, str):
        clap_status = None
    clap_label = event.get("clap_top_label")
    if not isinstance(clap_label, str):
        clap_label = None

    return {
        "t": created,
        "dba": dba_f,
        "l90_rel": l90_rel,
        "yamnet": yamnet,
        "gated": gated,
        "clap_status": clap_status,
        "clap": clap_label,
    }


def _status_series(*, window_s: int = 300) -> dict:
    """Build a slim rolling series for the live loudness chart."""
    window_s = max(60, min(900, int(window_s)))
    calib = float(DEFAULT_CALIB_OFFSET)
    path = _active_output_path()
    if path is None:
        return {
            "ok": True,
            "running": False,
            "window_s": window_s,
            "calib_offset": calib,
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
        "running": True,
        "window_s": window_s,
        "calib_offset": calib,
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


def _schedule_poweroff_after(grace_seconds: int) -> None:
    """Power off after a delay so the UI can show a countdown."""

    def _worker() -> None:
        time.sleep(grace_seconds)
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
        return {"running": False}

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
    started_at = None
    run_id = None

    if output_file:
        output_path = Path(output_file)
        last_event = _read_jsonl_event(output_path, last=True)
        if last_event:
            chunk_count = last_event.get("chunk_index", 0) + 1
            last_label = last_event.get("top_label")
            last_dba = last_event.get("dBA_spl")
            run_id = last_event.get("run_id")
        first_event = _read_jsonl_event(output_path, last=False)
        if first_event:
            started_at = first_event.get("created_at")
            if run_id is None:
                run_id = first_event.get("run_id")
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
        "running": True,
        "pid": proc.pid,
        "output_file": output_file,
        "chunk_count": chunk_count,
        "last_label": last_label,
        "last_dba": last_dba,
        "elapsed_s": elapsed_s,
        "started_at": started_at,
        "run_id": run_id,
    }


def _list_runs() -> list[dict]:
    """Return past JSONL runs sorted newest first."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in sorted(RUNS_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        size = p.stat().st_size
        lines = 0
        if size > 0:
            with p.open("rb") as f:
                lines = sum(1 for _ in f)
        wav_path = p.with_suffix(".wav")
        has_wav = wav_path.is_file()
        entry: dict = {
            "name": p.name,
            "size_kb": round(size / 1024, 1),
            "lines": lines,
            "mtime": datetime.fromtimestamp(
                p.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
            "has_wav": has_wav,
            "wav_name": wav_path.name if has_wav else None,
            "wav_size_kb": (
                round(wav_path.stat().st_size / 1024, 1) if has_wav else None
            ),
        }
        runs.append(entry)
    return runs


def _active_output_path() -> Path | None:
    """Return the collector ``-o`` path if a run is active."""
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


def _safe_runs_path(filename: str) -> Path | None:
    """Resolve a basename under RUNS_DIR, or None if unsafe/missing parent guard."""
    if not filename or "/" in filename or "\\" in filename or filename in {".", ".."}:
        return None
    path = (RUNS_DIR / filename).resolve()
    if not str(path).startswith(str(RUNS_DIR.resolve())):
        return None
    return path


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def get_login():
    return login_page()


@app.post("/login")
async def post_login(password: str = Form(...)):
    if password == PASSWORD:
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(
            SESSION_COOKIE,
            make_session_cookie(password),
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 7,
        )
        return resp
    return login_page(error="Incorrect password.")


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    status = _get_status()
    runs = _list_runs()
    error = request.query_params.get("error", "")
    return _render(
        "index.html",
        request=request,
        status=status,
        runs=runs,
        default_device_id=DEFAULT_DEVICE_ID,
        default_alsa_device=DEFAULT_ALSA_DEVICE,
        default_run_name=_default_run_name(),
        default_gate_sensitivity_db=DEFAULT_GATE_SENSITIVITY_DB,
        device_id=DEFAULT_DEVICE_ID,
        pi_hostname=hostname(),
        site_label=SITE_LABEL,
        public_url=PUBLIC_URL,
        error=error,
    )


# ---------------------------------------------------------------------------
# Control API
# ---------------------------------------------------------------------------

@app.post("/api/start")
async def api_start(
    request: Request,
    hours: float = Form(8.0),
    run_name: str = Form("run"),
    device_id: str = Form(DEFAULT_DEVICE_ID),
    alsa_device: str = Form(DEFAULT_ALSA_DEVICE),
    gate_sensitivity_db: float = Form(DEFAULT_GATE_SENSITIVITY_DB),
    enable_clap: str = Form(""),
    record_audio: str = Form(""),
):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)

    if _find_collector_process():
        return RedirectResponse("/?error=already_running", status_code=303)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_run_name(run_name)
    output = RUNS_DIR / f"{safe_name}-{_utc_now()}.jsonl"
    duration = _hours_to_timeout(hours)
    cmd = [
        PYTHON, str(MAIN_PY),
        "--device-id", device_id,
        "--alsa-device", alsa_device,
        "--backend", "arecord",
        "--quiet",
        "-o", str(output),
        "--yamnet-gate-sensitivity-db", str(gate_sensitivity_db),
    ]
    if enable_clap in {"1", "true", "on", "yes"}:
        cmd.append("--enable-clap")
    if record_audio in {"1", "true", "on", "yes"}:
        cmd.append("--record-wav")
    full_cmd = ["timeout", duration] + cmd

    subprocess.Popen(
        full_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from web server process group
    )
    return RedirectResponse("/", status_code=303)


@app.post("/api/stop")
async def api_stop(request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)

    _stop_collector()
    return RedirectResponse("/", status_code=303)


@app.post("/api/shutdown")
async def api_shutdown(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    stopped_recording = _stop_collector()
    _schedule_poweroff_after(SHUTDOWN_GRACE_SEC)

    return JSONResponse(
        {
            "ok": True,
            "grace_seconds": SHUTDOWN_GRACE_SEC,
            "stopped_recording": stopped_recording,
            "hostname": hostname(),
            "message": (
                "Shutdown scheduled. Keep this page open until the countdown "
                "finishes, then unplug power."
            ),
        }
    )


@app.get("/api/status")
async def api_status(request: Request):
    if not is_authenticated(request):
        return HTMLResponse("", status_code=401)
    return JSONResponse(_get_status())


@app.get("/api/status/series")
async def api_status_series(request: Request):
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    raw = request.query_params.get("window_s", "300")
    try:
        window_s = int(raw)
    except (TypeError, ValueError):
        window_s = 300
    return JSONResponse(_status_series(window_s=window_s))


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


@app.get("/runs/{filename}")
async def download_run(filename: str, request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    path = _safe_runs_path(filename)
    if path is None or not path.exists() or not path.is_file():
        return HTMLResponse("Not found", status_code=404)

    # Audio (or any non-JSONL) — serve as attachment without filtering.
    if path.suffix.lower() != ".jsonl":
        media = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
        return FileResponse(
            path,
            filename=path.name,
            media_type=media,
        )

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


@app.post("/api/runs/delete")
async def api_runs_delete(
    request: Request,
    filename: str = Form(...),
):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)

    if not filename.lower().endswith(".jsonl"):
        return RedirectResponse("/?tab=data&error=bad_request", status_code=303)

    path = _safe_runs_path(filename)
    if path is None:
        return RedirectResponse("/?tab=data&error=bad_request", status_code=303)
    if not path.exists() or not path.is_file():
        return RedirectResponse("/?tab=data&error=not_found", status_code=303)

    active = _active_output_path()
    if active is not None and path.resolve() == active:
        return RedirectResponse("/?tab=data&error=run_in_use", status_code=303)

    wav_path = path.with_suffix(".wav")
    try:
        path.unlink()
        if wav_path.is_file():
            wav_path.unlink()
    except OSError:
        return RedirectResponse("/?tab=data&error=bad_request", status_code=303)

    return RedirectResponse("/?tab=data", status_code=303)


# ---------------------------------------------------------------------------
# CLAP config APIs (prompts + triggers + catalog)
# ---------------------------------------------------------------------------

def _require_auth_json(request: Request) -> JSONResponse | None:
    if not is_authenticated(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return None


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
    denied = _require_auth_json(request)
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
    denied = _require_auth_json(request)
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
            "pre_roll_seconds": cfg.pre_roll_seconds,
            "post_roll_seconds": cfg.post_roll_seconds,
            "trigger_labels": cfg.trigger_labels,
            "ambiguous_labels": cfg.ambiguous_labels,
        }
    )


@app.put("/api/clap/triggers")
async def api_clap_triggers_put(request: Request):
    denied = _require_auth_json(request)
    if denied:
        return denied
    from core.clap_trigger import ClapTriggerConfig, save_trigger_config

    body = await request.json()
    try:
        cfg = ClapTriggerConfig(
            cooldown_seconds=float(body.get("cooldown_seconds", 5)),
            dba_threshold=float(body.get("dba_threshold", 55)),
            pre_roll_seconds=float(body.get("pre_roll_seconds", 7)),
            post_roll_seconds=float(body.get("post_roll_seconds", 3)),
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
    denied = _require_auth_json(request)
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
