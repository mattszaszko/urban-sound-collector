"""YAMNet→CLAP trigger rules and cooldown state."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIGGERS_PATH = REPO_ROOT / "config" / "clap_triggers.json"

CLAP_STATUS_TRIGGERED = "triggered"
CLAP_STATUS_CARRIED = "carried"
CLAP_STATUS_SKIPPED = "skipped"
CLAP_STATUS_GATED = "gated"


@dataclass
class ClapTriggerConfig:
    cooldown_seconds: float = 5.0
    carry_ttl_seconds: float = 30.0
    dba_threshold: float = 55.0
    trigger_labels: list[str] = field(default_factory=list)
    ambiguous_labels: list[str] = field(default_factory=list)

    def normalized(self) -> "ClapTriggerConfig":
        return ClapTriggerConfig(
            cooldown_seconds=float(self.cooldown_seconds),
            carry_ttl_seconds=float(self.carry_ttl_seconds),
            dba_threshold=float(self.dba_threshold),
            trigger_labels=sorted({x.strip() for x in self.trigger_labels if x.strip()}),
            ambiguous_labels=sorted(
                {x.strip() for x in self.ambiguous_labels if x.strip()}
            ),
        )


@dataclass
class ClapTriggerState:
    last_trigger_monotonic: float | None = None
    last_result: dict[str, Any] | None = None
    last_result_monotonic: float | None = None


def load_trigger_config(path: Path = DEFAULT_TRIGGERS_PATH) -> ClapTriggerConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ClapTriggerConfig(
        cooldown_seconds=float(data.get("cooldown_seconds", 5.0)),
        carry_ttl_seconds=float(data.get("carry_ttl_seconds", 30.0)),
        dba_threshold=float(data.get("dba_threshold", 55.0)),
        trigger_labels=list(data.get("trigger_labels", [])),
        ambiguous_labels=list(data.get("ambiguous_labels", [])),
    ).normalized()


def save_trigger_config(
    config: ClapTriggerConfig, path: Path = DEFAULT_TRIGGERS_PATH
) -> None:
    cfg = config.normalized()
    # Keep ambiguous labels out of the direct trigger set.
    ambiguous = [x for x in cfg.ambiguous_labels if x not in set(cfg.trigger_labels)]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cooldown_seconds": cfg.cooldown_seconds,
        "carry_ttl_seconds": cfg.carry_ttl_seconds,
        "dba_threshold": cfg.dba_threshold,
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


def evaluate_trigger(
    *,
    gated: bool,
    top_label: str,
    dba_spl: float,
    buffer_ready: bool,
    config: ClapTriggerConfig,
    state: ClapTriggerState,
    now_monotonic: float | None = None,
) -> tuple[bool, str, str]:
    """Return (should_run_clap, status_if_not, reason).

    When should_run_clap is True, status_if_not is unused and reason explains
    why CLAP should fire. When False, status_if_not is gated/skipped/carried.
    """
    now = time.monotonic() if now_monotonic is None else now_monotonic
    if gated:
        return False, CLAP_STATUS_GATED, "gated"
    if not buffer_ready:
        return False, CLAP_STATUS_SKIPPED, "buffer_warming"

    in_trigger = _label_in_set(top_label, config.trigger_labels)
    in_ambiguous = _label_in_set(top_label, config.ambiguous_labels)
    loud_enough = float(dba_spl) >= float(config.dba_threshold)

    if in_trigger:
        reason = f"label:{top_label}"
    elif in_ambiguous and loud_enough:
        reason = f"ambiguous:{top_label}@dBA>={config.dba_threshold}"
    else:
        # Maybe carry previous result.
        if (
            state.last_result is not None
            and state.last_result_monotonic is not None
            and (now - state.last_result_monotonic) <= config.carry_ttl_seconds
        ):
            return False, CLAP_STATUS_CARRIED, "carry"
        return False, CLAP_STATUS_SKIPPED, "no_match"

    if state.last_trigger_monotonic is not None:
        elapsed = now - state.last_trigger_monotonic
        if elapsed < config.cooldown_seconds:
            if (
                state.last_result is not None
                and state.last_result_monotonic is not None
                and (now - state.last_result_monotonic) <= config.carry_ttl_seconds
            ):
                return False, CLAP_STATUS_CARRIED, "cooldown"
            return False, CLAP_STATUS_SKIPPED, "cooldown"

    return True, CLAP_STATUS_TRIGGERED, reason
