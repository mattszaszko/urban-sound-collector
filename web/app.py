"""FastAPI web interface for the Urban Sound Collector."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import psutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from jinja2 import Environment, FileSystemLoader

from core.host_identity import default_device_id, hostname

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
RUNS_DIR = REPO_ROOT / "runs"
LOGS_DIR = REPO_ROOT / "logs"
MAIN_PY = REPO_ROOT / "main.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Fall back to system python if venv not present (useful for dev on PC)
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
_OUTPUT_STAMP_RE = re.compile(r"-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}Z$")

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
DEFAULT_YAMNET_GAIN = float(os.environ.get("YAMNET_GAIN", "15.0"))
SITE_LABEL = os.environ.get("SITE_LABEL", "").strip()
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").strip()

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


def _output_prefix(path: Path) -> str:
    stripped = _OUTPUT_STAMP_RE.sub("", path.stem)
    return stripped or path.stem


def _newest_matching_jsonl(cmd_output: Path) -> Path:
    """Prefer the latest rotated file with the same name prefix."""
    prefix = _output_prefix(cmd_output)
    matches = [
        p
        for p in cmd_output.parent.glob(f"{prefix}-*.jsonl")
        if p.is_file()
    ]
    if not matches:
        return cmd_output
    return max(matches, key=lambda p: p.stat().st_mtime)


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


def _find_collector_process() -> psutil.Process | None:
    """Return the running main.py Process, or None."""
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
        first_path = Path(output_file)
        latest_path = _newest_matching_jsonl(first_path)
        last_event = _read_jsonl_event(latest_path, last=True)
        if last_event:
            chunk_count = last_event.get("chunk_index", 0) + 1
            last_label = last_event.get("top_label")
            last_dba = last_event.get("dBA_spl")
            run_id = last_event.get("run_id")
        first_event = _read_jsonl_event(first_path, last=False)
        if first_event:
            started_at = first_event.get("created_at")
            if run_id is None:
                run_id = first_event.get("run_id")
        output_file = str(latest_path)

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
        runs.append(
            {
                "name": p.name,
                "size_kb": round(size / 1024, 1),
                "lines": lines,
                "mtime": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
    return runs


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
        default_gain=DEFAULT_YAMNET_GAIN,
        default_run_name=_default_run_name(),
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
    rotate_hours: float = Form(0.0),
    run_name: str = Form("run"),
    device_id: str = Form(DEFAULT_DEVICE_ID),
    alsa_device: str = Form(DEFAULT_ALSA_DEVICE),
    yamnet_gain: float = Form(DEFAULT_YAMNET_GAIN),
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
        "--yamnet-gain", str(yamnet_gain),
        "--quiet",
        "-o", str(output),
        "--rotate-hours", str(max(0.0, rotate_hours)),
    ]
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

    proc = _find_collector_process()
    if proc:
        try:
            # Send SIGTERM to the process group so timeout + python both stop
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()

    return RedirectResponse("/", status_code=303)


@app.get("/api/status")
async def api_status(request: Request):
    if not is_authenticated(request):
        return HTMLResponse("", status_code=401)
    from fastapi.responses import JSONResponse
    return JSONResponse(_get_status())


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

@app.get("/runs/{filename}")
async def download_run(filename: str, request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    # Prevent path traversal
    path = (RUNS_DIR / filename).resolve()
    if not str(path).startswith(str(RUNS_DIR.resolve())):
        return HTMLResponse("Not found", status_code=404)
    if not path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(path, filename=filename, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=PORT, reload=False)
