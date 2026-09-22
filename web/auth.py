"""Session-based role authentication for the web UI (admin | viewer)."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Literal

from fastapi import Request
from fastapi.responses import HTMLResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SESSION_COOKIE = "usc_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

Role = Literal["admin", "viewer"]
ROLES: tuple[Role, ...] = ("admin", "viewer")


def _serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("SECRET_KEY", "change-me-please")
    return URLSafeTimedSerializer(secret)


def _password_digest(password: str) -> str:
    """Stable digest for comparing rotated passwords (not for storage at rest)."""
    secret = os.environ.get("SECRET_KEY", "change-me-please").encode("utf-8")
    return hmac.new(secret, password.encode("utf-8"), hashlib.sha256).hexdigest()


def admin_password() -> str:
    return os.environ.get("USC_PASSWORD", "changeme")


def viewer_password() -> str:
    return os.environ.get("USC_VIEWER_PASSWORD", "").strip()


def password_for_role(role: Role) -> str:
    if role == "admin":
        return admin_password()
    return viewer_password()


def make_session_cookie(password: str, role: Role) -> str:
    if role not in ROLES:
        raise ValueError(f"invalid role: {role}")
    return _serializer().dumps(
        {"role": role, "pw_hash": _password_digest(password)}
    )


def _load_session(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        data = _serializer().loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict):
        return None
    role = data.get("role")
    pw_hash = data.get("pw_hash")
    # Legacy cookies had {"authed": True, "pw_hash": hash(password)} — reject them.
    if role not in ROLES or not isinstance(pw_hash, str):
        return None
    expected = password_for_role(role)  # type: ignore[arg-type]
    if not expected:
        return None
    if not hmac.compare_digest(pw_hash, _password_digest(expected)):
        return None
    return data


def is_authenticated(request: Request) -> bool:
    return _load_session(request) is not None


def get_role(request: Request) -> Role | None:
    data = _load_session(request)
    if not data:
        return None
    role = data.get("role")
    return role if role in ROLES else None  # type: ignore[return-value]


def is_admin(request: Request) -> bool:
    return get_role(request) == "admin"


def is_viewer(request: Request) -> bool:
    return get_role(request) == "viewer"


def verify_login(role: str, password: str) -> Role | None:
    """Return the role if credentials match, else None."""
    if role not in ROLES:
        return None
    expected = password_for_role(role)  # type: ignore[arg-type]
    if not expected:
        return None
    # Compare digests so length mismatch cannot raise from compare_digest.
    if not hmac.compare_digest(_password_digest(password), _password_digest(expected)):
        return None
    return role  # type: ignore[return-value]


LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Urban Sound Collector — Login</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#0f172a;color:#f1f5f9;
       display:flex;align-items:center;justify-content:center;min-height:100vh;
       padding:1rem}}
  .card{{background:#1e293b;border-radius:12px;padding:2rem;width:min(90vw,380px);
        box-shadow:0 4px 24px #0008}}
  h1{{font-size:1.25rem;margin-bottom:.35rem;color:#38bdf8}}
  .sub{{font-size:.8rem;color:#94a3b8;margin-bottom:1.25rem;line-height:1.4}}
  .role-toggle{{display:flex;gap:.35rem;margin-bottom:1rem;background:#0f172a;
               border:1px solid #334155;border-radius:8px;padding:.25rem}}
  .role-toggle label{{flex:1;text-align:center;padding:.55rem .4rem;border-radius:6px;
                     font-size:.85rem;font-weight:600;cursor:pointer;color:#94a3b8}}
  .role-toggle input{{position:absolute;opacity:0;pointer-events:none}}
  .role-toggle label:has(input:checked){{background:#0ea5e9;color:#fff}}
  input[type=password]{{width:100%;padding:.75rem;border:1px solid #334155;border-radius:8px;
        background:#0f172a;color:#f1f5f9;font-size:1rem;margin-bottom:1rem}}
  button[type=submit]{{width:100%;padding:.75rem;background:#0ea5e9;border:none;border-radius:8px;
         color:#fff;font-size:1rem;cursor:pointer;font-weight:600}}
  button[type=submit]:hover{{background:#38bdf8}}
  .err{{color:#f87171;font-size:.875rem;margin-bottom:1rem}}
  .hint{{font-size:.72rem;color:#64748b;margin-top:.85rem;line-height:1.4}}
</style>
</head>
<body>
<div class="card">
  <h1>🎙 Urban Sound Collector</h1>
  <p class="sub">Choose how you want to sign in.</p>
  {error}
  <form method="post" action="/login">
    <div class="role-toggle" role="radiogroup" aria-label="Sign-in mode">
      <label><input type="radio" name="role" value="admin" {admin_checked}> Admin</label>
      <label><input type="radio" name="role" value="viewer" {viewer_checked}> Viewer</label>
    </div>
    <input type="password" name="password" placeholder="Password" autofocus required>
    <button type="submit">Sign in</button>
  </form>
  <p class="hint">Admin can start/stop recordings and change settings.
  Viewer can browse, download, inspect, and generate reports (read-only).</p>
</div>
</body>
</html>"""


def login_page(error: str = "", *, role: str = "admin") -> HTMLResponse:
    err_html = f'<p class="err">{error}</p>' if error else ""
    selected = role if role in ROLES else "admin"
    return HTMLResponse(
        LOGIN_HTML.format(
            error=err_html,
            admin_checked="checked" if selected == "admin" else "",
            viewer_checked="checked" if selected == "viewer" else "",
        )
    )
