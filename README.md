# Urban Sound Collector

Quick-and-dirty Raspberry Pi edge collector for live urban noise monitoring.

Captures audio from an **INMP441 I2S microphone**, resamples to YAMNet's expected format, runs **on-device classification** with Google's YAMNet model, and emits structured JSON events for logging or later cloud ingestion.

## Features

- Live capture via `sounddevice` (typically 48 kHz hardware → 16 kHz for YAMNet)
- Software gain for quiet INMP441 levels
- Per-chunk JSON events with UTC timestamps, RMS, and top-k predictions
- JSONL file output for long-running field tests

## Requirements

- Raspberry Pi 4/5 (64-bit OS recommended)
- INMP441 wired for I2S mono (L/R → GND)
- Python 3.11+ venv

## Setup (Pi)

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libportaudio2 portaudio19-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download YAMNet once (example):

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.model_download('google/yamnet/TensorFlow2/yamnet/1'))"
```

## Run

```bash
python main.py \
  --model-path ~/.cache/kagglehub/models/google/yamnet/TensorFlow2/yamnet/1 \
  --device-id pi-test-01 \
  --quiet \
  -o runs/evening-test.jsonl
```

Stop with `Ctrl+C`.
