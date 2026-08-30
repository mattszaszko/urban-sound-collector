"""Hostname and device identity helpers for multi-Pi deployments."""

from __future__ import annotations

import os
import socket


def hostname() -> str:
    """Return the machine hostname."""
    return socket.gethostname().strip() or "pi-unknown"


def default_device_id() -> str:
    """Return DEVICE_ID from the environment, or the Pi hostname."""
    explicit = os.environ.get("DEVICE_ID", "").strip()
    return explicit or hostname()
