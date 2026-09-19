"""YAMNet→CLAP arming rules (dynamic energy capture; no carry)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.clap_capture import (
    DEFAULT_END_SETTLE_CHUNKS,
    DEFAULT_LOOKBACK_SECONDS,
    DEFAULT_MAX_EVENT_SECONDS,
    DEFAULT_ONSET_DBA_MARGIN_DB,
    DEFAULT_PEAK_DECAY_DB,
    DEFAULT_PRE_ONSET_PAD_MS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIGGERS_PATH = REPO_ROOT / "config" / "clap_triggers.json"

CLAP_STATUS_TRIGGERED = "triggered"
CLAP_STATUS_SCHEDULED = "scheduled"
CLAP_STATUS_PENDING = "pending"
CLAP_STATUS_SKIPPED = "skipped"
CLAP_STATUS_GATED = "gated"

# Environmental / broadband labels that must never arm CLAP.
DEFAULT_SUPPRESS_LABELS = [
    "Wind",
    "Rustling leaves",
    "White noise",
    "Outside, rural or natural",
]


@dataclass
class ClapTriggerConfig:
    cooldown_seconds: float = 5.0
    dba_threshold: float = 55.0
    trigger_labels: list[str] = field(default_factory=list)
    ambiguous_labels: list[str] = field(default_factory=list)
    suppress_labels: list[str] = field(
        default_factory=lambda: list(DEFAULT_SUPPRESS_LABELS)
    )
    lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS
    pre_onset_pad_ms: float = DEFAULT_PRE_ONSET_PAD_MS
    end_settle_chunks: int = DEFAULT_END_SETTLE_CHUNKS
    max_event_seconds: float = DEFAULT_MAX_EVENT_SECONDS
    onset_dba_margin_db: float = DEFAULT_ONSET_DBA_MARGIN_DB
    peak_decay_db: float = DEFAULT_PEAK_DECAY_DB

    def normalized(self) -> "ClapTriggerConfig":
        suppress = sorted({x.strip() for x in self.suppress_labels if x.strip()})
        suppress_set = {x.lower() for x in suppress}
        triggers = sorted(
            {
                x.strip()
                for x in self.trigger_labels
                if x.strip() and x.strip().lower() not in suppress_set
            }
        )
        ambiguous = sorted(
            {
                x.strip()
                for x in self.ambiguous_labels
                if x.strip()
                and x.strip().lower() not in suppress_set
                and x.strip() not in set(triggers)
            }
        )
        return ClapTriggerConfig(
            cooldown_seconds=float(self.cooldown_seconds),
            dba_threshold=float(self.dba_threshold),
            trigger_labels=triggers,
            ambiguous_labels=ambiguous,
            suppress_labels=suppress,
            lookback_seconds=float(self.lookback_seconds),
            pre_onset_pad_ms=float(self.pre_onset_pad_ms),
            end_settle_chunks=max(1, int(self.end_settle_chunks)),
            max_event_seconds=float(self.max_event_seconds),
            onset_dba_margin_db=float(self.onset_dba_margin_db),
            peak_decay_db=float(self.peak_decay_db),
        )


@dataclass
class ClapTriggerState:
    """Cooldown is measured from the last arm time (capture start)."""

    last_arm_monotonic: float | None = None


def load_trigger_config(path: Path = DEFAULT_TRIGGERS_PATH) -> ClapTriggerConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Legacy hybrid 7+3 keys are ignored when present; dynamic defaults apply.
    raw_suppress = data.get("suppress_labels")
    if raw_suppress is None:
        suppress_labels = list(DEFAULT_SUPPRESS_LABELS)
    else:
        suppress_labels = list(raw_suppress)
    return ClapTriggerConfig(
        cooldown_seconds=float(data.get("cooldown_seconds", 5.0)),
        dba_threshold=float(data.get("dba_threshold", 55.0)),
        trigger_labels=list(data.get("trigger_labels", [])),
        ambiguous_labels=list(data.get("ambiguous_labels", [])),
        suppress_labels=suppress_labels,
        lookback_seconds=float(
            data.get("lookback_seconds", DEFAULT_LOOKBACK_SECONDS)
        ),
        pre_onset_pad_ms=float(
            data.get("pre_onset_pad_ms", DEFAULT_PRE_ONSET_PAD_MS)
        ),
        end_settle_chunks=int(
            data.get("end_settle_chunks", DEFAULT_END_SETTLE_CHUNKS)
        ),
        max_event_seconds=float(
            data.get("max_event_seconds", DEFAULT_MAX_EVENT_SECONDS)
        ),
        onset_dba_margin_db=float(
            data.get("onset_dba_margin_db", DEFAULT_ONSET_DBA_MARGIN_DB)
        ),
        peak_decay_db=float(data.get("peak_decay_db", DEFAULT_PEAK_DECAY_DB)),
    ).normalized()


def save_trigger_config(
    config: ClapTriggerConfig, path: Path = DEFAULT_TRIGGERS_PATH
) -> None:
    cfg = config.normalized()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cooldown_seconds": cfg.cooldown_seconds,
        "dba_threshold": cfg.dba_threshold,
        "lookback_seconds": cfg.lookback_seconds,
        "pre_onset_pad_ms": cfg.pre_onset_pad_ms,
        "end_settle_chunks": cfg.end_settle_chunks,
        "max_event_seconds": cfg.max_event_seconds,
        "onset_dba_margin_db": cfg.onset_dba_margin_db,
        "peak_decay_db": cfg.peak_decay_db,
        "trigger_labels": cfg.trigger_labels,
        "ambiguous_labels": cfg.ambiguous_labels,
        "suppress_labels": cfg.suppress_labels,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _label_in_set(top_label: str, labels: list[str]) -> bool:
    needle = top_label.strip().lower()
    for item in labels:
        if item.strip().lower() == needle:
            return True
    return False


def evaluate_arm(
    *,
    gated: bool,
    top_label: str,
    dba_spl: float,
    preroll_ready: bool,
    capture_active: bool,
    config: ClapTriggerConfig,
    state: ClapTriggerState,
    now_monotonic: float | None = None,
) -> tuple[bool, str, str]:
    """Decide whether to arm a dynamic CLAP capture.

    Returns (should_arm, status_if_not, reason).
    When should_arm is True, status_if_not is unused.
    """
    now = time.monotonic() if now_monotonic is None else now_monotonic
    if capture_active:
        return False, CLAP_STATUS_PENDING, "capturing"
    if gated:
        return False, CLAP_STATUS_GATED, "gated"
    if not preroll_ready:
        return False, CLAP_STATUS_SKIPPED, "buffer_warming"

    if _label_in_set(top_label, config.suppress_labels):
        return False, CLAP_STATUS_SKIPPED, f"suppress:{top_label}"

    in_trigger = _label_in_set(top_label, config.trigger_labels)
    in_ambiguous = _label_in_set(top_label, config.ambiguous_labels)
    loud_enough = float(dba_spl) >= float(config.dba_threshold)

    if in_trigger:
        reason = f"label:{top_label}"
    elif in_ambiguous and loud_enough:
        reason = f"ambiguous:{top_label}@dBA>={config.dba_threshold}"
    else:
        return False, CLAP_STATUS_SKIPPED, "no_match"

    if state.last_arm_monotonic is not None:
        elapsed = now - state.last_arm_monotonic
        if elapsed < config.cooldown_seconds:
            return False, CLAP_STATUS_SKIPPED, "cooldown"

    return True, CLAP_STATUS_SCHEDULED, reason


# Back-compat alias used by older imports/tests during transition.
evaluate_trigger = evaluate_arm
