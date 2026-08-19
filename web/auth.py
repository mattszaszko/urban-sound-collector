"""Session-based password authentication for the web UI."""

from __future__ import annotations

import os
from functools import wraps
from typing import Callable

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SESSION_COOKIE = "usc_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def _serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("SECRET_KEY", "change-me-please")
    return URLSafeTimedSerializer(secret)


def make_session_cookie(password: str) -> str:
    return _serializer().dumps({"authed": True, "pw_hash": hash(password)})


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return False
    try:
        _serializer().loads(token, max_age=SESSION_MAX_AGE)
        return True
    except (BadSignature, SignatureExpired):
        return False


LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Urban Sound Collector — Login</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#0f172a;color:#f1f5f9;
       display:flex;align-items:center;justify-content:center;min-height:100vh}}
  .card{{background:#1e293b;border-radius:12px;padding:2rem;width:min(90vw,360px);
        box-shadow:0 4px 24px #0008}}
  h1{{font-size:1.25rem;margin-bottom:1.5rem;color:#38bdf8}}
  input{{width:100%;padding:.75rem;border:1px solid #334155;border-radius:8px;
        background:#0f172a;color:#f1f5f9;font-size:1rem;margin-bottom:1rem}}
  button{{width:100%;padding:.75rem;background:#0ea5e9;border:none;border-radius:8px;
         color:#fff;font-size:1rem;cursor:pointer;font-weight:600}}
  button:hover{{background:#38bdf8}}
  .err{{color:#f87171;font-size:.875rem;margin-bottom:1rem}}
</style>
</head>
<body>
<div class="card">
  <h1>🎙 Urban Sound Collector</h1>
  {error}
  <form method="post" action="/login">
    <input type="password" name="password" placeholder="Password" autofocus>
    <button type="submit">Sign in</button>
  </form>
</div>
</body>
</html>"""


def login_page(error: str = "") -> HTMLResponse:
    err_html = f'<p class="err">{error}</p>' if error else ""
    return HTMLResponse(LOGIN_HTML.format(error=err_html))
