#!/usr/bin/env bash
# Urban Sound Collector — web UI + Cloudflare Tunnel setup
# Run once on the Pi: bash web/install.sh
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo ""
echo "=== Urban Sound Collector — web setup ==="
echo ""

# ── 1. Python deps ─────────────────────────────────────────────────────────
echo "Installing web UI Python dependencies..."
source .venv/bin/activate
pip install -q -r "$REPO/web/requirements-web.txt"
echo "Done."
echo ""

# ── 2. .env ────────────────────────────────────────────────────────────────
if [ ! -f "$REPO/.env" ]; then
  echo "Creating .env from template..."
  cp "$REPO/.env.example" "$REPO/.env"

  SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  # Use | as sed delimiter to avoid issues with / in paths
  sed -i "s|SECRET_KEY=change-me-please|SECRET_KEY=$SECRET|" "$REPO/.env"

  echo ""
  echo ">> Please set a password in .env now:"
  echo "   nano $REPO/.env"
  echo "   (Change USC_PASSWORD=changeme to something strong)"
  echo ""
  read -rp "Press Enter after you have saved the .env file..."
else
  echo ".env already exists — skipping."
fi
echo ""

# ── 3. cloudflared ─────────────────────────────────────────────────────────
echo "Checking for cloudflared..."
if ! command -v cloudflared &>/dev/null; then
  echo "Installing cloudflared..."
  ARCH=$(uname -m)
  if [ "$ARCH" = "aarch64" ]; then
    DEB_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb"
  else
    DEB_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm.deb"
  fi
  curl -fsSL "$DEB_URL" -o /tmp/cloudflared.deb
  sudo dpkg -i /tmp/cloudflared.deb
  rm /tmp/cloudflared.deb
  echo "cloudflared installed."
else
  echo "cloudflared already installed: $(cloudflared --version)"
fi
echo ""

# ── 4. Start web server in background ──────────────────────────────────────
echo "Starting web server on port 8080..."
source "$REPO/.venv/bin/activate"

# Kill any existing instance
pkill -f "uvicorn web.app:app" 2>/dev/null || true
sleep 1

nohup python -m uvicorn web.app:app --host 0.0.0.0 --port 8080 \
  > "$REPO/logs/webserver.log" 2>&1 &
disown
echo "Web server started (logs: $REPO/logs/webserver.log)"
echo ""

# ── 5. Cloudflare Tunnel ───────────────────────────────────────────────────
echo "Starting Cloudflare Tunnel..."
echo "(This gives you a public HTTPS URL — no router config needed)"
echo ""
echo "Your URL will appear below. Save it — it changes on restart unless you"
echo "set up a named tunnel (free at dash.cloudflare.com)."
echo ""
echo "Press Ctrl+C to stop the tunnel when done."
echo ""
cloudflared tunnel --url http://localhost:8080
