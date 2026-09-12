"""YAMNet→CLAP arming rules (hybrid 7+3 capture; no carry)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIGGERS_PATH = REPO_ROOT / "config" / "clap_triggers.json"

CLAP_STATUS_TRIGGERED = "triggered"
CLAP_STATUS_SCHEDULED = "scheduled"
CLAP_STATUS_PENDING = "pending"
CLAP_STATUS_SKIPPED = "skipped"
CLAP_STATUS_GATED = "gated"

# Hybrid window defaults (seconds)
CLAP_PRE_ROLL_SECONDS = 7.0
CLAP_POST_ROLL_SECONDS = 3.0


@dataclass
class ClapTriggerConfig:
    cooldown_seconds: float = 5.0
    dba_threshold: float = 55.0
    trigger_labels: list[str] = field(default_factory=list)
    ambiguous_labels: list[str] = field(default_factory=list)
    pre_roll_seconds: float = CLAP_PRE_ROLL_SECONDS
    post_roll_seconds: float = CLAP_POST_ROLL_SECONDS

    def normalized(self) -> "ClapTriggerConfig":
        return ClapTriggerConfig(
            cooldown_seconds=float(self.cooldown_seconds),
            dba_threshold=float(self.dba_threshold),
            trigger_labels=sorted({x.strip() for x in self.trigger_labels if x.strip()}),
            ambiguous_labels=sorted(
                {x.strip() for x in self.ambiguous_labels if x.strip()}
            ),
            pre_roll_seconds=float(self.pre_roll_seconds),
            post_roll_seconds=float(self.post_roll_seconds),
        )


@dataclass
class ClapTriggerState:
    """Cooldownoldown is measured from the last arm time (capture start)."""

    last_arm_monotonic: float | None = None


def load_trigger_config(path: Path = DEFAULT_TRIGGERS_PATH) -> ClapTriggerConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ClapTriggerConfig(
        cooldown_seconds=float(data.get("cooldown_seconds", 5.0)),
        dba_threshold=float(data.get("dba_threshold", 55.0)),
        trigger_labels=list(data.get("trigger_labels", [])),
        ambiguous_labels=list(data.get("ambiguous_labels", [])),
        pre_roll_seconds=float(
            data.get("pre_roll_seconds", CLAP_PRE_ROLL_SECONDS)
        ),
        post_roll_seconds=float(
            data.get("post_roll_seconds", CLAP_POST_ROLL_SECONDS)
        ),
    ).normalized()


def save_trigger_config(
    config: ClapTriggerConfig, path: Path = DEFAULT_TRIGGERS_PATH
) -> None:
    cfg = config.normalized()
    ambiguous = [x for x in cfg.ambiguous_labels if x not in set(cfg.trigger_labels)]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cooldown_seconds": cfg.cooldown_seconds,
        "dba_threshold": cfg.dba_threshold,
        "pre_roll_seconds": cfg.pre_roll_seconds,
        "post_roll_seconds": cfg.post_roll_seconds,
        "trigger_labels": cfg.trigger_labels,
        "ambiguous_labels": ambiguous,
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
    """Decide whether to arm a hybrid CLAP capture.

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
