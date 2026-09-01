# Multi-Pi deployment

Run **multiple Raspberry Pis at the same time**, each with its own microphone,
data files, web UI, and public URL.

## Architecture

```text
noise-site-a.mattszaszko.com  →  tunnel A  →  Pi A  →  localhost:8080
noise-site-b.mattszaszko.com  →  tunnel B  →  Pi B  →  localhost:8080
```

| Rule | Why |
|---|---|
| **One tunnel per Pi** | Each tunnel token connects one device to Cloudflare |
| **One subdomain per Pi** | `noise-*` hostname must point at a single origin |
| **Never share a tunnel token** | Two Pis with the same token split traffic randomly |
| **Same Git repo on every Pi** | Code is identical; `.env` differs per device |
| **Unique `SITE_LABEL` + `PUBLIC_URL`** | So the web UI shows which Pi you are on |

Data never mixes: each Pi writes only to its own `runs/` and `logs/` folders.

---

## Per-Pi checklist

Use this table when adding Pi #2, #3, …

| Step | Pi A (existing) | Pi B (new) |
|---|---|---|
| Hostname | e.g. `raspberrypi` | e.g. `pi-site-b` (set in Imager) |
| Subdomain | `noise.mattszaszko.com` | `noise-site-b.mattszaszko.com` |
| Cloudflare tunnel | `urban-noise-pi` (keep) | **Create new** tunnel |
| `SITE_LABEL` in `.env` | e.g. `Amsterdam window` | e.g. `Site B balcony` |
| `PUBLIC_URL` in `.env` | `https://noise.mattszaszko.com` | `https://noise-site-b.mattszaszko.com` |
| `DEVICE_ID` in `.env` | leave empty → hostname | leave empty → hostname |
| `SECRET_KEY` | unique | **new** random key per Pi |

---

## Fresh Pi setup (full)

Assumes **Raspberry Pi OS 64-bit (Bookworm)**, user **`matt`**, Google Voice HAT.

### 1. OS and network

1. Flash with **Raspberry Pi Imager** (enable SSH, set user/password, Wi‑Fi).
2. Boot and SSH in: `ssh matt@<pi-ip>`

### 2. System packages

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev libasound2-dev alsa-utils curl
python3 --version
```

Prefer **Python 3.11** if neither TFLite package works on 3.13:

```bash
sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv .venv   # use instead of python3 -m venv
```

### 3. Clone the repo

```bash
cd ~
git clone https://github.com/mattszaszko/urban-sound-collector.git
cd urban-sound-collector
```

### 4. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy pyalsaaudio
pip install ai-edge-litert
# if that fails, try legacy runtime (Python 3.11 only):
# pip install tflite-runtime
# or use: python3.11 -m venv .venv
pip install -r web/requirements-web.txt
mkdir -p runs logs
```

### 5. Microphone test

```bash
arecord -l
arecord -D plughw:CARD=sndrpigooglevoi,DEV=0 -f S32_LE -r 48000 -c 1 -d 5 /tmp/mic-test.wav
ls -lh /tmp/mic-test.wav
```

### 6. Configure `.env` (unique per Pi)

```bash
cp .env.example .env
nano .env
```

Example for **second Pi**:

```env
USC_PASSWORD=your-shared-or-unique-password
SECRET_KEY=<run: python3 -c "import secrets; print(secrets.token_hex(32))">
SITE_LABEL=Site B balcony
PUBLIC_URL=https://noise-site-b.mattszaszko.com
DEVICE_ID=
ALSA_DEVICE=plughw:CARD=sndrpigooglevoi,DEV=0
```

Leave `DEVICE_ID` empty to use the Pi hostname in JSONL events.

### 7. Collector smoke test

```bash
source .venv/bin/activate
timeout 1m python main.py \
  --alsa-device plughw:CARD=sndrpigooglevoi,DEV=0 \
  --backend arecord \
  --quiet \
  -o "runs/test-$(date -u +%Y-%m-%dT%H-%MZ).jsonl"
wc -l runs/test-*.jsonl
```

### 8. Web UI systemd service

```bash
sudo cp ~/urban-sound-collector/web/urban-sound-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now urban-sound-web
sudo systemctl status urban-sound-web
```

### 9. Cloudflare named tunnel (new tunnel per Pi)

**Do not reuse the tunnel token from another Pi.**

1. [dash.cloudflare.com](https://dash.cloudflare.com) → **Zero Trust** → **Networks** → **Tunnels**
2. **Create a tunnel** → name e.g. `urban-noise-site-b`
3. Connector: **Debian / 64-bit** — copy the install command
4. On **this Pi only**:

```bash
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb
sudo cloudflared service install <TOKEN_FROM_CLOUDFLARE>
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
sudo systemctl status cloudflared
```

5. **Public hostname** (in tunnel config):
   - Subdomain: `noise-site-b` (pick a unique name)
   - Domain: `mattszaszko.com`
   - Service: `HTTP` → `localhost:8080`

6. Open `PUBLIC_URL` from `.env` in a browser → log in → start a test run.

### 10. Reboot test

```bash
sudo reboot
```

After ~2 minutes both services should be up without SSH:

```bash
sudo systemctl status urban-sound-web cloudflared
```

---

## Adding a second Pi while the first keeps running

1. **Leave Pi A alone** — its tunnel and `noise.mattszaszko.com` keep working.
2. Set up Pi B using the checklist above with a **new tunnel** and **new subdomain**.
3. Update Pi B's `.env` with `SITE_LABEL` and `PUBLIC_URL` for the new URL.
4. Bookmark both URLs — each controls only its own Pi.

---

## Updating code on all Pis

On each Pi:

```bash
cd ~/urban-sound-collector
git pull
source .venv/bin/activate
pip install -r web/requirements-web.txt
sudo systemctl restart urban-sound-web
```

`.env` is never overwritten by `git pull`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Wrong Pi responds at URL | Two Pis share one tunnel token — create separate tunnels |
| 502 on website | `sudo systemctl status urban-sound-web` — uvicorn not running |
| Mic not working | `arecord -l`; confirm `ALSA_DEVICE` uses `CARD=sndrpigooglevoi` |
| `tflite-runtime` / `ai-edge-litert` install fails | Try `pip install ai-edge-litert`; else Python 3.11 venv + `tflite-runtime` |
| Can't tell which Pi in UI | Set `SITE_LABEL` and `PUBLIC_URL` in `.env`, restart web service |

---

## Why `.env` is not in git

`.env` contains **passwords and secret keys**. Each Pi has its own copy on disk.
`.env.example` in the repo is the template only — copy it locally per device.
