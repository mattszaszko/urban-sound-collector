"""CLI entry point for the Urban IoT live INMP441 edge collector."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import List, TextIO

from core.audio_constants import CAPTURE_CHUNK_SAMPLES, CAPTURE_SAMPLE_RATE
from core.capture_alsa import AlsAudioCapture
from core.classifier_tflite import (
    DEFAULT_MODEL_PATH,
    MODEL_VERSION,
    YamnetTFLiteClassifier,
)
from core.events import build_noise_event, new_run_id
from core.loudness import DEFAULT_CALIB_OFFSET, LoudnessEngine
from core.pcm import int32_frames_to_float32
from core.resampler import to_yamnet_waveform

DEFAULT_ALSA_DEVICE = "plughw:3,0"
# Classifier-only boost for quiet distant sources (window traffic, etc.).
# Does not affect loudness / dBA metrics.
DEFAULT_YAMNET_GAIN = 15.0


def default_device_id() -> str:
    """Use the machine hostname as a stable-enough default device id."""
    return socket.gethostname().strip() or "pi-unknown"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Urban IoT edge collector: capture INMP441 audio via ALSA, "
            "compute A-weighted loudness, classify with YAMNet TFLite, "
            "and emit JSONL events."
        )
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=default_device_id(),
        help=f"Logical device id for events (default: {default_device_id()})",
    )
    parser.add_argument(
        "--alsa-device",
        type=str,
        default=DEFAULT_ALSA_DEVICE,
        help=f"ALSA capture device (default: {DEFAULT_ALSA_DEVICE})",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "pyalsa", "arecord"),
        default="auto",
        help="ALSA capture backend (default: auto).",
    )
    parser.add_argument(
        "--calib-offset",
        type=float,
        default=DEFAULT_CALIB_OFFSET,
        help=f"Relative dBA calibration offset (default: {DEFAULT_CALIB_OFFSET})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to yamnet.tflite (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--yamnet-gain",
        type=float,
        default=DEFAULT_YAMNET_GAIN,
        help=(
            "Digital gain applied only to the YAMNet input waveform "
            f"(default: {DEFAULT_YAMNET_GAIN}). Loudness metrics stay ungained. "
            "Use 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Write JSONL events to this file (one JSON object per line). "
            "Data is flushed after each chunk."
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


def stream_live(
    *,
    classifier: YamnetTFLiteClassifier,
    device_id: str,
    run_id: str,
    alsa_device: str,
    backend: str,
    calib_offset: float,
    yamnet_gain: float,
    output_path: Path,
    print_stdout: bool,
) -> int:
    """Capture audio, run dual-branch analysis, emit JSONL events.

    Branch A: A-weighted loudness at 48 kHz (ungained).
    Branch B: YAMNet TFLite at 16 kHz (optional classifier-only gain).

    Returns:
        Number of events written.
    """
    chunk_index = 0
    events_written = 0
    capture: AlsAudioCapture | None = None
    output_handle: TextIO | None = None

    loudness = LoudnessEngine(
        sample_rate=float(CAPTURE_SAMPLE_RATE),
        calib_offset=calib_offset,
    )

    print(
        f"Opening INMP441 stream: device_id={device_id}, run_id={run_id}, "
        f"alsa={alsa_device}, backend={backend}, rate={CAPTURE_SAMPLE_RATE} Hz, "
        f"format=S32_LE, chunk_samples={CAPTURE_CHUNK_SAMPLES}, "
        f"calib_offset={calib_offset}, yamnet_gain={yamnet_gain}, "
        f"model={MODEL_VERSION}",
        file=sys.stderr,
    )
    print(f"Recording JSONL to: {output_path.resolve()}", file=sys.stderr)
    if print_stdout:
        print("Streaming JSON to stdout. Press Ctrl+C to stop.", file=sys.stderr)
    else:
        print("JSON stdout disabled (--quiet). Press Ctrl+C to stop.", file=sys.stderr)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")
        capture = AlsAudioCapture(
            alsa_device,
            CAPTURE_CHUNK_SAMPLES,
            backend=backend,
        )

        for raw_chunk in capture.iter_chunks():
            pcm = int32_frames_to_float32(raw_chunk)

            # Branch A — human loudness @ capture rate (no classifier gain).
            metrics = loudness.analyse(pcm)

            # Branch B — YAMNet classification @ 16 kHz (+ optional gain).
            try:
                waveform = to_yamnet_waveform(pcm, gain=yamnet_gain)
                predictions = classifier.predict(waveform)
            except (ValueError, RuntimeError) as exc:
                print(
                    f"  [WARN] Chunk {chunk_index} classification failed: {exc}",
                    file=sys.stderr,
                )
                chunk_index += 1
                continue

            result = build_noise_event(
                device_id=device_id,
                chunk_index=chunk_index,
                run_id=run_id,
                rms_unweighted=metrics["rms_unweighted"],
                rms_a_weighted=metrics["rms_a_weighted"],
                dba_spl=metrics["dBA_spl"],
                predictions=predictions,
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
        if capture is not None:
            capture.close()
            print("Audio stream closed.", file=sys.stderr)
        if output_handle is not None:
            output_handle.close()
            print(
                f"Wrote {events_written} event(s) to {output_path.resolve()}",
                file=sys.stderr,
            )

    return events_written


def main(argv: List[str] | None = None) -> int:
    """Run live INMP441 capture with loudness + YAMNet classification.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    args = parse_args(argv)
    run_id = new_run_id()
    output_path = args.output or default_output_path(args.device_id, run_id)
    print_stdout = not args.quiet

    print(f"Loading YAMNet TFLite from {args.model_path} ...", file=sys.stderr)
    try:
        classifier = YamnetTFLiteClassifier(model_path=args.model_path)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        events_written = stream_live(
            classifier=classifier,
            device_id=args.device_id,
            run_id=run_id,
            alsa_device=args.alsa_device,
            backend=args.backend,
            calib_offset=args.calib_offset,
            yamnet_gain=args.yamnet_gain,
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
