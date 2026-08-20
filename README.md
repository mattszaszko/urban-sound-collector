# Urban Sound Collector

Raspberry Pi edge collector for live urban noise monitoring.

Captures audio from an **INMP441 I2S microphone**, aligns 24-bit PCM from ALSA,
computes **A-weighted loudness (relative dBA)**, classifies sound events with
**bundled YAMNet TFLite**, and streams results as **JSON Lines** for logging or
later dashboard ingestion.

---

## Refactor goals

The original prototype used `sounddevice` + TensorFlow Hub SavedModel with a
`x15` software gain hack to compensate for quiet INMP441 levels. That worked for
near-field sounds but corrupted loudness telemetry and made outdoor traffic hard
to classify reliably.

This refactor targets a **modular, Pi-native edge pipeline**:

| Before | After |
|---|---|
| `sounddevice` float capture | ALSA `S32_LE` int32 @ 48 kHz |
| Global digital gain | Correct 24-bit PCM alignment (`>> 8`, `/ 2^23`) |
| TensorFlow Hub SavedModel | Bundled **YAMNet TFLite** (lighter, offline) |
| Single RMS metric | **Dual branch**: loudness + classification |
| WAV/file testing path | Live capture only |

Classifier-only **`--yamnet-gain`** remains optional for quiet outdoor sources
(window traffic). Loudness metrics are always ungained.

---

## Architecture

Each ~0.975 s window flows through two parallel branches from one PCM buffer:

```text
INMP441 (I2S) → ALSA S32_LE @ 48 kHz
        │
        ▼
  int32 → PCM align → float32 [-1, 1]
        │
        ├─ Branch A (48 kHz) ──────────────────────────────┐
        │   A-weighting IIR (stateful)                     │
        │   rms_unweighted, rms_a_weighted, dBA_spl        │
        │                                                  │
        └─ Branch B (16 kHz) ──────────────────────────────┤
            resample_poly (48k → 16k)                      │
            optional --yamnet-gain (classifier only)       │
            YAMNet TFLite → top-3 predictions              │
                                                           ▼
                                              JSONL event per chunk
```

```mermaid
flowchart LR
    mic[INMP441 I2S]
    alsa[ALSA S32_LE 48kHz]
    pcm[PCM align]
    loud[LoudnessEngine]
    res[resample_poly]
    yam[YAMNet TFLite]
    json[JSONL event]

    mic --> alsa --> pcm
    pcm --> loud
    pcm --> res --> yam
    loud --> json
    yam --> json
```

---

## Project structure

```text
urban-sound-collector/
├── main.py                     # CLI entry: capture loop, JSONL output
├── requirements.txt            # collector deps (numpy, scipy, pyalsaaudio)
├── .env.example                # web UI config template
├── .gitattributes              # enforce LF line endings
├── README.md
├── core/
│   ├── audio_constants.py      # 48 kHz capture / 16 kHz YAMNet window sizes
│   ├── capture_alsa.py         # ALSA capture (pyalsa or arecord backend)
│   ├── pcm.py                  # int32 → aligned float32 normalization
│   ├── loudness.py             # A-weighting + relative dBA SPL
│   ├── resampler.py            # 48 kHz → 16 kHz for YAMNet
│   ├── classifier_tflite.py    # Bundled YAMNet TFLite inference
│   └── events.py               # JSONL event schema builder
├── web/
│   ├── app.py                  # FastAPI web server
│   ├── auth.py                 # Session-based password login
│   ├── requirements-web.txt    # web-only deps (fastapi, uvicorn, …)
│   ├── install.sh              # First-time setup script (deps + cloudflared)
│   ├── urban-sound-web.service # systemd unit — auto-start web UI on boot
│   └── templates/
│       └── index.html          # Mobile-friendly UI
├── models/
│   ├── yamnet.tflite           # Bundled classifier (~4 MB)
│   └── yamnet_class_map.csv
├── runs/                       # Local JSONL output (gitignored)
└── logs/                       # Collector + web server logs (gitignored)
```

### Module responsibilities

| Module | Role |
|---|---|
| `main.py` | Argument parsing, orchestrates capture → analysis → JSONL |
| `capture_alsa.py` | Opens ALSA device, yields fixed-size int32 chunks |
| `pcm.py` | INMP441 24-bit alignment into unit-scale float samples |
| `loudness.py` | IEC 61672-style A-weighting, RMS, relative `dBA_spl` |
| `resampler.py` | Polyphase resample + optional classifier gain + clip |
| `classifier_tflite.py` | Loads TFLite model, returns top-3 AudioSet labels |
| `events.py` | Builds one JSON object per chunk |

---

## JSONL event schema

One JSON object per line (~1 Hz):

```json
{
  "device_id": "pi-test-01",
  "created_at": "2026-08-12T14:14:34.317Z",
  "chunk_index": 0,
  "run_id": "2026-08-12T14-14-27Z",
  "rms_unweighted": 0.00851457,
  "rms_a_weighted": 0.0045808,
  "dBA_spl": 73.2,
  "top_label": "Vehicle",
  "top_confidence": 0.26171875,
  "predictions": [
    {"label": "Vehicle", "confidence": 0.26171875},
    {"label": "White noise", "confidence": 0.109375},
    {"label": "Car", "confidence": 0.08203125}
  ],
  "model_name": "yamnet",
  "model_version": "tflite/1"
}
```

- **`created_at`**: UTC timestamp for time-series use
- **`chunk_index`**: per-run counter (resets on restart)
- **`dBA_spl`**: relative SPL (not absolute; no calibrated mic yet)

---

## Requirements

- Raspberry Pi 4/5, 64-bit OS (Bookworm)
- INMP441 on I2S (L/R → GND for mono Left)
- Python 3.11 venv recommended (`tflite-runtime` wheels)

---

## Setup (Pi)

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libasound2-dev alsa-utils
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Find your ALSA device:

```bash
arecord -l
# typically plughw:3,0 for Google Voice HAT / INMP441
```

---

## Durability (crash / power loss)

JSONL is **append + flush + fsync** after every chunk (~1 Hz), so completed
events survive a process crash or power cut. At most the in-progress chunk is
lost.

Status/errors go to a **new file per run** under `logs/` (not `/tmp`, never
overwritten):

```text
logs/<device_id>_<run_id>.log
```

Heartbeat lines are written about once per minute so you can confirm progress
after a crash.

Do **not** redirect logs to `/tmp/urban-sound.log` — that path is wiped on
reboot and overwrites itself if reused.

---

## Run

```bash
mkdir -p runs logs
nohup timeout 3h python main.py \
  --device-id pi-test-01 \
  --alsa-device plughw:3,0 \
  --backend arecord \
  --yamnet-gain 15 \
  --quiet \
  -o "runs/evening-$(date -u +%Y-%m-%dT%H-%MZ).jsonl" \
  >/dev/null 2>&1 &
```

Collector logs still land in `logs/` (stderr is duplicated there). Check later:

```bash
pgrep -af "main.py"
ls -lh runs/ logs/
tail -n 20 logs/*.log
```

### Useful flags

| Flag | Default | Purpose |
|---|---|---|
| `--alsa-device` | `plughw:3,0` | ALSA capture device |
| `--backend` | `auto` | `pyalsa`, `arecord`, or `auto` |
| `--yamnet-gain` | `15.0` | Classifier-only boost; use `1` to disable |
| `--calib-offset` | `120.0` | Relative dBA offset |
| `--quiet` | off | Suppress JSON on stdout |
| `-o` | `runs/<device>_<run_id>.jsonl` | JSONL output (append + fsync) |
| `--log-dir` | `logs` | Per-run log directory |

Stop with **Ctrl+C**, or let `timeout` end the run.

---

## Web UI (remote control + monitoring)

A FastAPI web interface lets you start/stop runs and monitor status from any
device — phone, PC, anywhere on the internet — via a **Cloudflare Tunnel**
(free, no port forwarding, automatic HTTPS).

### Features

- Start a run with:
  - **Hours** input (e.g. `8`, or `0.5` for 30 minutes)
  - **Output file name** prefix (prefilled from time of day: `morning` /
    `day` / `evening` / `night`); UTC start stamp is always appended
    (`evening-2026-08-20T09-32Z.jsonl`)
  - Device ID, ALSA device, YAMNet gain
- Stop a running run
- Live status: chunk count, elapsed time, last label, dBA (polls every 10 s)
- Live log tail via Server-Sent Events (no page refresh needed)
- Download past JSONL files directly in the browser
- Password-protected login (session cookie, 7-day expiry)
- Optional **systemd** unit so the web UI auto-starts on Pi reboot
- Named Cloudflare Tunnel for a stable public URL (e.g. `https://noise.mattszaszko.com`)

---

### How Cloudflare Tunnel works

```
Your phone / browser (anywhere on the internet)
          ↓  HTTPS
  *.trycloudflare.com  ←  Cloudflare's global network
          ↓  encrypted outbound tunnel
  cloudflared process on the Pi
          ↓  localhost:8080
  FastAPI web server
```

The Pi makes an **outbound** connection to Cloudflare — no open router ports,
no static IP, no DNS setup required. Cloudflare relays browser traffic back
through the tunnel.

The free `trycloudflare.com` URL is **random and changes** every time
`cloudflared` restarts. For a permanent URL, set up a named tunnel tied to a
domain (free at [dash.cloudflare.com](https://dash.cloudflare.com)).

---

### First-time setup on the Pi

**1. Push the code** (from PowerShell on your PC):

```powershell
scp -r "C:\path\to\urban-sound-collector\web" matt@192.168.178.11:~/urban-sound-collector/
scp "C:\path\to\urban-sound-collector\.env.example" matt@192.168.178.11:~/urban-sound-collector/
```

**2. Run the install script** (on the Pi):

```bash
cd ~/urban-sound-collector
source .venv/bin/activate
# Strip Windows line endings if pushed from a Windows PC
sed -i 's/\r//' web/install.sh web/requirements-web.txt
bash web/install.sh
```

The script will:
- Install the 7 web Python packages
- Create `.env` from the template and prompt you to set a password (`nano .env`)
- Install `cloudflared` (ARM64 `.deb`)
- Start the web server in the background
- Start a Cloudflare Tunnel and print your public HTTPS URL

**3. Open the URL** on any device, enter your password, done.

---

### Named tunnel (stable URL)

Preferred setup: a named Cloudflare Tunnel on your domain, e.g.
`https://noise.mattszaszko.com` → `http://localhost:8080` on the Pi.

Install `cloudflared` as a systemd service from the Cloudflare Zero Trust
dashboard (Debian / 64-bit connector). Once that service is enabled, the tunnel
survives reboots automatically.

---

### Auto-start the web UI on boot (systemd)

Install the unit file so uvicorn starts after every reboot (no manual `nohup`):

```bash
# On the Pi — copy the unit and enable it
sudo cp ~/urban-sound-collector/web/urban-sound-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now urban-sound-web
sudo systemctl status urban-sound-web
```

Useful commands:

```bash
sudo systemctl status urban-sound-web   # is it running?
sudo systemctl restart urban-sound-web  # after code/config changes
sudo journalctl -u urban-sound-web -f   # live logs
```

Stop any old manual uvicorn first so port 8080 is free:

```bash
pkill -f "uvicorn web.app:app" 2>/dev/null
```

After reboot you should have:

| Component | Status |
|---|---|
| `cloudflared` | Auto-starts (named tunnel) |
| `urban-sound-web` | Auto-starts (uvicorn on :8080) |
| Collector run | Manual via web UI when you want |

---

### Environment variables (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `USC_PASSWORD` | `changeme` | Web UI login password |
| `SECRET_KEY` | `change-me-please` | Session cookie signing key (auto-generated by `install.sh`) |
| `PORT` | `8080` | Web server port |
| `DEVICE_ID` | `pi-test-01` | Default device ID shown in UI |
| `ALSA_DEVICE` | `plughw:CARD=sndrpigooglevoi,DEV=0` | Default ALSA device shown in UI |
| `YAMNET_GAIN` | `15.0` | Default YAMNet gain shown in UI |

The `.env` file is gitignored — never commit it.

---

## Crash durability

JSONL output uses **append + flush + `fsync`** after every chunk (~1 Hz), so
completed events survive a process crash or power cut. At most the current
~1 s chunk in progress is lost.

Logs go to `logs/<device>_<run_id>.log` — one file per run, never overwritten,
not in `/tmp`. A heartbeat line is written every ~60 chunks (~1 min) so you
can confirm progress after a crash.

Use `nohup ... & disown` when starting long runs so that closing SSH does not
kill the collector.

---

## Future work

- Firestore upload from the Pi (edge writer only)
- Optional overlap / shorter hop for faster label updates
