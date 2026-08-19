"""CLI entry point for the Urban IoT live INMP441 edge collector."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
DEFAULT_LOG_DIR = Path("logs")

logger = logging.getLogger("urban_sound_collector")


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
            "Each line is flushed and fsync'd so a crash keeps prior chunks."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory for per-run log files (default: {DEFAULT_LOG_DIR})",
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


def default_log_path(log_dir: Path, device_id: str, run_id: str) -> Path:
    """Build a unique per-run log path (never overwrite previous runs)."""
    safe_device = device_id.replace("/", "-").replace(" ", "_")
    return log_dir / f"{safe_device}_{run_id}.log"


def setup_logging(log_path: Path) -> None:
    """Log to stderr and a durable per-run file under logs/."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # Force UTC timestamps in the log file.
    formatter.converter = __import__("time").gmtime

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.INFO)

    root.addHandler(file_handler)
    root.addHandler(stderr_handler)


def emit_event(
    event: dict,
    *,
    stdout: bool,
    output_handle: TextIO | None,
) -> None:
    """Print one JSON event to stdout and/or append to a JSONL file.

    File writes flush + fsync so each chunk survives a hard crash/power loss
    (at the cost of more SD-card wear).
    """
    line = json.dumps(event, ensure_ascii=False)
    if stdout:
        print(line, flush=True)
    if output_handle is not None:
        output_handle.write(line + "\n")
        output_handle.flush()
        os.fsync(output_handle.fileno())


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

    logger.info(
        "Opening INMP441 stream: device_id=%s, run_id=%s, alsa=%s, backend=%s, "
        "rate=%s Hz, format=S32_LE, chunk_samples=%s, calib_offset=%s, "
        "yamnet_gain=%s, model=%s",
        device_id,
        run_id,
        alsa_device,
        backend,
        CAPTURE_SAMPLE_RATE,
        CAPTURE_CHUNK_SAMPLES,
        calib_offset,
        yamnet_gain,
        MODEL_VERSION,
    )
    logger.info("Recording JSONL to: %s", output_path.resolve())
    if print_stdout:
        logger.info("Streaming JSON to stdout. Press Ctrl+C to stop.")
    else:
        logger.info("JSON stdout disabled (--quiet). Press Ctrl+C to stop.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Append so a reused -o path never truncates prior data.
        output_handle = output_path.open("a", encoding="utf-8")
        capture = AlsAudioCapture(
            alsa_device,
            CAPTURE_CHUNK_SAMPLES,
            backend=backend,
        )

        for raw_chunk in capture.iter_chunks():
            if events_written == 0:
                logger.info("First audio chunk received; pipeline is live.")
            pcm = int32_frames_to_float32(raw_chunk)

            # Branch A — human loudness @ capture rate (no classifier gain).
            metrics = loudness.analyse(pcm)

            # Branch B — YAMNet classification @ 16 kHz (+ optional gain).
            try:
                waveform = to_yamnet_waveform(pcm, gain=yamnet_gain)
                predictions = classifier.predict(waveform)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Chunk %s classification failed: %s", chunk_index, exc)
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

            # Heartbeat every ~60 chunks (~1 min) so logs show progress.
            if events_written % 60 == 0:
                logger.info(
                    "Progress: %s event(s) written (chunk_index=%s, top=%s)",
                    events_written,
                    chunk_index - 1,
                    result.get("top_label"),
                )

    except KeyboardInterrupt:
        logger.info("Stopping capture (KeyboardInterrupt).")
    finally:
        if capture is not None:
            capture.close()
            logger.info("Audio stream closed.")
        if output_handle is not None:
            try:
                output_handle.flush()
                os.fsync(output_handle.fileno())
            except OSError:
                pass
            output_handle.close()
            logger.info(
                "Wrote %s event(s) to %s",
                events_written,
                output_path.resolve(),
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
    log_path = default_log_path(args.log_dir, args.device_id, run_id)
    print_stdout = not args.quiet

    setup_logging(log_path)
    logger.info("Logging to: %s", log_path.resolve())

    logger.info("Loading YAMNet TFLite from %s ...", args.model_path)
    try:
        classifier = YamnetTFLiteClassifier(model_path=args.model_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
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
        logger.exception("Fatal error: %s", exc)
        return 1

    if events_written == 0:
        logger.warning("No events were recorded.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
