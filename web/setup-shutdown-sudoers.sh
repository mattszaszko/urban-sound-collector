#!/usr/bin/env bash
# Allow the web UI user to shut down the Pi without a password prompt.
# Run once on each Pi: sudo bash web/setup-shutdown-sudoers.sh
set -eu

USER_NAME="${SUDO_USER:-matt}"
RULES_FILE="/etc/sudoers.d/urban-sound-shutdown"

SHUTDOWN="$(command -v shutdown || true)"
SYSTEMCTL="$(command -v systemctl || true)"
SYNC="$(command -v sync || true)"

if [[ -z "$SHUTDOWN" || -z "$SYSTEMCTL" ]]; then
  echo "Could not find shutdown or systemctl on PATH." >&2
  exit 1
fi

{
  echo "# Urban Sound Collector — web UI shutdown (managed by setup-shutdown-sudoers.sh)"
  echo "${USER_NAME} ALL=(ALL) NOPASSWD: ${SHUTDOWN}"
  echo "${USER_NAME} ALL=(ALL) NOPASSWD: ${SYSTEMCTL} poweroff"
  if [[ -n "$SYNC" ]]; then
    echo "${USER_NAME} ALL=(ALL) NOPASSWD: ${SYNC}"
  fi
} > "$RULES_FILE"

chmod 0440 "$RULES_FILE"
visudo -cf "$RULES_FILE"
echo "Installed ${RULES_FILE} for user ${USER_NAME}."
