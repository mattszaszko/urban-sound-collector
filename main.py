"""CLI entry point for the Urban IoT live INMP441 edge collector."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, TextIO

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from core.audio_constants import CAPTURE_CHUNK_SAMPLES, CAPTURE_SAMPLE_RATE
from core.capture_alsa import AlsAudioCapture
from core.classifier_tflite import (
    DEFAULT_MODEL_PATH,
    MODEL_VERSION,
    YamnetTFLiteClassifier,
)
from core.events import build_noise_event, new_run_id
from core.host_identity import default_device_id
from core.loudness import DEFAULT_CALIB_OFFSET, LoudnessEngine
from core.pcm import int32_frames_to_float32
from core.spectrum import SpectrumEngine
from core.yamnet_preprocess import (
    DEFAULT_AMBIENT_PERCENTILE,
    DEFAULT_AMBIENT_WINDOW_CHUNKS,
    DEFAULT_GAIN_SMOOTH_CHUNKS,
    DEFAULT_GATE_HYSTERESIS_DB,
    DEFAULT_GATE_SENSITIVITY_DB,
    DEFAULT_HPF_HZ,
    DEFAULT_TARGET_DBFS,
    YamnetPreprocessor,
    silence_predictions,
)

DEFAULT_ALSA_DEVICE = "plughw:3,0"
DEFAULT_LOG_DIR = Path("logs")

logger = logging.getLogger("urban_sound_collector")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Urban IoT edge collector: capture INMP441 audio via ALSA, "
            "compute A-weighted loudness, classify with YAMNet TFLite, "
            "analyse spectrum (Z + A 1/3-octave), and emit JSONL events."
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
        "--yamnet-hpf-hz",
        type=float,
        default=DEFAULT_HPF_HZ,
        help=f"YAMNet high-pass cutoff in Hz (default: {DEFAULT_HPF_HZ}).",
    )
    parser.add_argument(
        "--yamnet-target-dbfs",
        type=float,
        default=DEFAULT_TARGET_DBFS,
        help=(
            "YAMNet RMS normalization target in dBFS "
            f"(default: {DEFAULT_TARGET_DBFS})."
        ),
    )
    parser.add_argument(
        "--yamnet-gate-sensitivity-db",
        type=float,
        default=DEFAULT_GATE_SENSITIVITY_DB,
        help=(
            "Silence gate open offset above L90 ambient floor in dB "
            f"(default: {DEFAULT_GATE_SENSITIVITY_DB})."
        ),
    )
    parser.add_argument(
        "--yamnet-gate-hysteresis-db",
        type=float,
        default=DEFAULT_GATE_HYSTERESIS_DB,
        help=(
            "Silence gate close offset below open threshold in dB "
            f"(default: {DEFAULT_GATE_HYSTERESIS_DB})."
        ),
    )
    parser.add_argument(
        "--yamnet-gate-ambient-chunks",
        type=int,
        default=DEFAULT_AMBIENT_WINDOW_CHUNKS,
        help=(
            "Rolling window size for L90 ambient floor in chunks "
            f"(default: {DEFAULT_AMBIENT_WINDOW_CHUNKS}, ~5 min)."
        ),
    )
    parser.add_argument(
        "--yamnet-gate-percentile",
        type=float,
        default=DEFAULT_AMBIENT_PERCENTILE,
        help=(
            "Percentile for ambient noise floor (L90 = 10th percentile; "
            f"default: {DEFAULT_AMBIENT_PERCENTILE})."
        ),
    )
    parser.add_argument(
        "--yamnet-gain-smooth-chunks",
        type=int,
        default=DEFAULT_GAIN_SMOOTH_CHUNKS,
        help=(
            "Chunks over which to smooth YAMNet normalization gain "
            f"(default: {DEFAULT_GAIN_SMOOTH_CHUNKS})."
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
    parser.add_argument(
        "--no-spectrum",
        action="store_true",
        help="Disable Branch C spectral analysis (1/3-octave Z + A summaries).",
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


class JsonlWriter:
    """Append JSONL with flush + fsync after each line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle: TextIO = self.path.open("a", encoding="utf-8")

    def write_line(self, line: str) -> None:
        self.handle.write(line + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        except OSError:
            pass
        self.handle.close()


def emit_event(
    event: dict,
    *,
    stdout: bool,
    writer: JsonlWriter | None,
) -> None:
    """Print one JSON event to stdout and/or append to a JSONL file.

    File writes flush + fsync so each chunk survives a hard crash/power loss
    (at the cost of more SD-card wear).
    """
    line = json.dumps(event, ensure_ascii=False)
    if stdout:
        print(line, flush=True)
    if writer is not None:
        writer.write_line(line)


def stream_live(
    *,
    classifier: YamnetTFLiteClassifier,
    device_id: str,
    run_id: str,
    alsa_device: str,
    backend: str,
    calib_offset: float,
    yamnet_preprocessor: YamnetPreprocessor,
    output_path: Path,
    print_stdout: bool,
    enable_spectrum: bool = True,
) -> int:
    """Capture audio, run triple-branch analysis, emit JSONL events.

    Branch A: A-weighted loudness at 48 kHz (ungained).
    Branch B: YAMNet TFLite at 16 kHz (dynamic HPF + RMS normalize).
    Branch C: Z- and A-weighted 1/3-octave spectrum at 48 kHz (ungained).

    Returns:
        Number of events written.
    """
    chunk_index = 0
    events_written = 0
    gated_chunks = 0
    capture: AlsAudioCapture | None = None
    writer: JsonlWriter | None = None

    loudness = LoudnessEngine(
        sample_rate=float(CAPTURE_SAMPLE_RATE),
        calib_offset=calib_offset,
    )
    spectrum_engine: SpectrumEngine | None = None
    if enable_spectrum:
        spectrum_engine = SpectrumEngine(sample_rate=float(CAPTURE_SAMPLE_RATE))

    logger.info(
        "Opening INMP441 stream: device_id=%s, run_id=%s, alsa=%s, backend=%s, "
        "rate=%s Hz, format=S32_LE, chunk_samples=%s, calib_offset=%s, "
        "yamnet_hpf_hz=%s, yamnet_target_dbfs=%s, yamnet_gate_mode=dynamic_l90, "
        "yamnet_gate_sensitivity_db=%s, yamnet_gate_hysteresis_db=%s, "
        "yamnet_gate_ambient_chunks=%s, yamnet_gate_percentile=%s, "
        "yamnet_gain_smooth_chunks=%s, model=%s, spectrum=%s",
        device_id,
        run_id,
        alsa_device,
        backend,
        CAPTURE_SAMPLE_RATE,
        CAPTURE_CHUNK_SAMPLES,
        calib_offset,
        yamnet_preprocessor.hpf_hz,
        yamnet_preprocessor.target_dbfs,
        yamnet_preprocessor.gate_sensitivity_db,
        yamnet_preprocessor.gate_hysteresis_db,
        yamnet_preprocessor.ambient_window_chunks,
        yamnet_preprocessor.ambient_percentile,
        yamnet_preprocessor.gain_smooth_chunks,
        MODEL_VERSION,
        enable_spectrum,
    )
    logger.info("Recording JSONL to: %s", output_path.resolve())
    if print_stdout:
        logger.info("Streaming JSON to stdout. Press Ctrl+C to stop.")
    else:
        logger.info("JSON stdout disabled (--quiet). Press Ctrl+C to stop.")

    try:
        writer = JsonlWriter(output_path)
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

            # Branch C — Z- and A-weighted spectrum @ capture rate (no gain).
            spectrum = (
                spectrum_engine.analyse(pcm) if spectrum_engine is not None else None
            )

            # Branch B — YAMNet @ 16 kHz (HPF, dynamic normalize, silence gate).
            try:
                prep = yamnet_preprocessor.prepare(pcm)
                if prep.gated:
                    predictions = silence_predictions()
                    gated_chunks += 1
                else:
                    predictions = classifier.predict(prep.waveform_16k)
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
                spectrum=spectrum,
                yamnet_preprocess=prep.metadata,
            )
            emit_event(
                result,
                stdout=print_stdout,
                writer=writer,
            )
            events_written += 1
            chunk_index += 1

            # Heartbeat every ~60 chunks (~1 min) so logs show progress.
            if events_written % 60 == 0:
                logger.info(
                    "Progress: %s event(s) written (chunk_index=%s, top=%s, "
                    "gated=%s/%s, file=%s)",
                    events_written,
                    chunk_index - 1,
                    result.get("top_label"),
                    gated_chunks,
                    events_written,
                    writer.path.name,
                )

    except KeyboardInterrupt:
        logger.info("Stopping capture (KeyboardInterrupt).")
    finally:
        if capture is not None:
            capture.close()
            logger.info("Audio stream closed.")
        if writer is not None:
            final_path = writer.path
            writer.close()
            logger.info(
                "Wrote %s event(s); last file %s",
                events_written,
                final_path.resolve(),
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
        yamnet_preprocessor = YamnetPreprocessor(
            sample_rate=float(CAPTURE_SAMPLE_RATE),
            hpf_hz=args.yamnet_hpf_hz,
            target_dbfs=args.yamnet_target_dbfs,
            gain_smooth_chunks=max(1, args.yamnet_gain_smooth_chunks),
            ambient_window_chunks=max(1, args.yamnet_gate_ambient_chunks),
            gate_sensitivity_db=args.yamnet_gate_sensitivity_db,
            gate_hysteresis_db=args.yamnet_gate_hysteresis_db,
            ambient_percentile=args.yamnet_gate_percentile,
        )
        events_written = stream_live(
            classifier=classifier,
            device_id=args.device_id,
            run_id=run_id,
            alsa_device=args.alsa_device,
            backend=args.backend,
            calib_offset=args.calib_offset,
            yamnet_preprocessor=yamnet_preprocessor,
            output_path=output_path,
            print_stdout=print_stdout,
            enable_spectrum=not args.no_spectrum,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 — surface device/open errors cleanly
        logger.exception("Fatal error: %s", exc)
        return 1

    if events_written == 0:
        logger.warning("No events were recorded.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
