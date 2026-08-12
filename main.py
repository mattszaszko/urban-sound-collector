"""CLI entry point for the Urban IoT live INMP441 edge collector."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import List, Sequence, TextIO

import numpy as np
import sounddevice as sd
from scipy.signal import resample

from core.audio_constants import CHUNK_SAMPLES, TARGET_SAMPLE_RATE
from core.classifier import YAMNetClassifier
from core.events import (
    DEFAULT_MODEL_VERSION,
    build_noise_event,
    new_run_id,
)
from core.metrics import calculate_rms

# INMP441 raw levels are quiet; scale before RMS + YAMNet.
DIGITAL_GAIN = 15.0
# I2S/ALSA devices often reject 16 kHz; try common hardware rates first.
CAPTURE_RATE_CANDIDATES: Sequence[int] = (48_000, 44_100, 32_000, 22_050, 16_000)


def default_device_id() -> str:
    """Use the machine hostname as a stable-enough default device id."""
    return socket.gethostname().strip() or "pi-unknown"


def infer_model_version(model_path: str | None) -> str:
    """Derive a short model_version string from a local path or use default."""
    if not model_path:
        return DEFAULT_MODEL_VERSION
    path = Path(model_path)
    # .../google/yamnet/TensorFlow2/yamnet/1 → TensorFlow2/yamnet/1
    parts = path.parts
    if len(parts) >= 3 and parts[-3] == "TensorFlow2":
        return "/".join(parts[-3:])
    return path.name or DEFAULT_MODEL_VERSION


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Urban IoT Noise Classifier: "
            "stream live INMP441 audio with YAMNet and emit JSON events."
        )
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=default_device_id(),
        help=f"Logical device id for events (default: {default_device_id()})",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device index (default: system default).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=DIGITAL_GAIN,
        help=f"Digital gain multiplier for INMP441 (default: {DIGITAL_GAIN})",
    )
    parser.add_argument(
        "--capture-rate",
        type=int,
        default=None,
        help=(
            "Optional hardware capture sample rate in Hz. "
            "If omitted, auto-detect a rate the device accepts."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Optional local YAMNet SavedModel directory "
            "(skips remote download)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Write JSONL events to this file (one JSON object per line). "
            "Recommended for long runs; data is flushed after each chunk."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print JSON events to stdout (use with --output).",
    )
    return parser.parse_args(argv)


def default_output_path(device_id: str, run_id: str) -> Path:
    """Build a default JSONL path under runs/."""
    safe_device = device_id.replace("/", "-").replace(" ", "_")
    return Path("runs") / f"{safe_device}_{run_id}.jsonl"


def emit_event(
    event: dict,
    *,
    stdout: bool,
    output_handle: TextIO | None,
) -> None:
    """Print one JSON event to stdout and/or append to a JSONL file."""
    line = json.dumps(event, ensure_ascii=False)
    if stdout:
        print(line, flush=True)
    if output_handle is not None:
        output_handle.write(line + "\n")
        output_handle.flush()


def resolve_capture_rate(device: int | None, preferred: int | None = None) -> int:
    """Return a mono capture sample rate accepted by the input device."""
    try:
        info = sd.query_devices(device, kind="input")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not query input device: {exc}") from exc

    default_rate = int(round(float(info.get("default_sample_rate", 48_000))))
    candidates: List[int] = []
    for rate in (preferred, default_rate, *CAPTURE_RATE_CANDIDATES):
        if rate is None:
            continue
        rate_i = int(rate)
        if rate_i > 0 and rate_i not in candidates:
            candidates.append(rate_i)

    errors: List[str] = []
    for rate in candidates:
        try:
            sd.check_input_settings(
                device=device,
                channels=1,
                dtype="float32",
                samplerate=rate,
            )
            return rate
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rate} Hz: {exc}")

    details = "; ".join(errors) if errors else "no candidates tried"
    raise RuntimeError(
        "No supported mono capture sample rate found for this device. "
        f"Tried: {candidates}. Details: {details}"
    )


def to_yamnet_chunk(capture_audio: np.ndarray, capture_rate: int) -> np.ndarray:
    """Resample one capture window to YAMNet's 16 kHz / 15,600-sample format."""
    mono = np.asarray(capture_audio, dtype=np.float32).reshape(-1)
    if capture_rate == TARGET_SAMPLE_RATE and mono.shape[0] == CHUNK_SAMPLES:
        return mono

    # scipy resample → exactly CHUNK_SAMPLES at TARGET_SAMPLE_RATE.
    resampled = resample(mono, CHUNK_SAMPLES)
    return np.asarray(resampled, dtype=np.float32)


def stream_live(
    classifier: YAMNetClassifier,
    *,
    device_id: str,
    run_id: str,
    model_version: str,
    device: int | None = None,
    gain: float = DIGITAL_GAIN,
    capture_rate: int | None = None,
    output_path: Path | None = None,
    print_stdout: bool = True,
) -> int:
    """Capture live mono audio, classify each YAMNet window, emit JSON events.

    Captures at a hardware-supported rate (often 48 kHz on I2S), resamples each
    ~0.975 s window to 16 kHz / 15,600 samples for YAMNet, then applies gain.

    Returns:
        Number of events written.
    """
    chunk_duration = CHUNK_SAMPLES / TARGET_SAMPLE_RATE
    resolved_rate = resolve_capture_rate(device, preferred=capture_rate)
    capture_blocksize = int(round(chunk_duration * resolved_rate))
    chunk_index = 0
    events_written = 0

    print(
        f"Opening INMP441 stream: device_id={device_id}, run_id={run_id}, "
        f"capture={resolved_rate} Hz → YAMNet={TARGET_SAMPLE_RATE} Hz, "
        f"mono (Left via L/R=GND), capture_blocksize={capture_blocksize}, "
        f"gain={gain}",
        file=sys.stderr,
    )
    if output_path is not None:
        print(f"Recording JSONL to: {output_path.resolve()}", file=sys.stderr)
    if print_stdout:
        print("Streaming JSON to stdout. Press Ctrl+C to stop.", file=sys.stderr)
    elif output_path is not None:
        print("JSON stdout disabled (--quiet). Press Ctrl+C to stop.", file=sys.stderr)
    else:
        print("Press Ctrl+C to stop.", file=sys.stderr)

    stream: sd.InputStream | None = None
    output_handle: TextIO | None = None
    try:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8")

        # channels=1: hardware already routes Left when L/R is grounded.
        stream = sd.InputStream(
            samplerate=resolved_rate,
            channels=1,
            dtype="float32",
            blocksize=capture_blocksize,
            device=device,
        )
        stream.start()

        while True:
            frames, overflowed = stream.read(capture_blocksize)
            if overflowed:
                print(
                    f"  [WARN] Input overflow before chunk {chunk_index}",
                    file=sys.stderr,
                )

            try:
                yamnet_chunk = to_yamnet_chunk(frames, resolved_rate)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [WARN] Resample failed for chunk {chunk_index}: {exc}",
                    file=sys.stderr,
                )
                chunk_index += 1
                continue

            amplified = yamnet_chunk * np.float32(gain)

            try:
                rms = calculate_rms(amplified)
                predictions = classifier.predict(amplified)
            except (ValueError, RuntimeError) as exc:
                print(
                    f"  [WARN] Chunk {chunk_index} failed: {exc}",
                    file=sys.stderr,
                )
                chunk_index += 1
                continue

            result = build_noise_event(
                device_id=device_id,
                chunk_index=chunk_index,
                run_id=run_id,
                rms=rms,
                predictions=predictions,
                model_version=model_version,
            )
            emit_event(
                result,
                stdout=print_stdout,
                output_handle=output_handle,
            )
            events_written += 1
            chunk_index += 1

    except KeyboardInterrupt:
        print("\nStopping capture (KeyboardInterrupt).", file=sys.stderr)
    finally:
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # noqa: BLE001 — best-effort shutdown
                pass
            try:
                stream.close()
            except Exception:  # noqa: BLE001 — best-effort shutdown
                pass
            print("Audio stream closed.", file=sys.stderr)
        if output_handle is not None:
            output_handle.close()
            print(
                f"Wrote {events_written} event(s) to {output_path.resolve()}",
                file=sys.stderr,
            )

    return events_written


def main(argv: List[str] | None = None) -> int:
    """Run live INMP441 capture and classification.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    args = parse_args(argv)
    run_id = new_run_id()
    model_version = infer_model_version(args.model_path)
    output_path = args.output or default_output_path(args.device_id, run_id)
    print_stdout = not args.quiet

    if args.model_path:
        print(f"Loading YAMNet from local path: {args.model_path}", file=sys.stderr)
    else:
        print(
            "Loading YAMNet from Kaggle Models "
            "(first run may download ~18 MB, often silently)...",
            file=sys.stderr,
        )

    try:
        classifier = (
            YAMNetClassifier(args.model_path)
            if args.model_path
            else YAMNetClassifier()
        )
    except KeyboardInterrupt:
        print(
            "\nCancelled while loading YAMNet (KeyboardInterrupt).",
            file=sys.stderr,
        )
        return 130
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        events_written = stream_live(
            classifier,
            device_id=args.device_id,
            run_id=run_id,
            model_version=model_version,
            device=args.device,
            gain=args.gain,
            capture_rate=args.capture_rate,
            output_path=output_path,
            print_stdout=print_stdout,
        )
    except Exception as exc:  # noqa: BLE001 — surface device/open errors cleanly
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if events_written == 0:
        print("Warning: no events were recorded.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
