#!/usr/bin/env bash
set -euo pipefail

# Install and load the flight recorder agent for Bad Apple.
# The agent tails the audit ledger and the engine intent drop-stream into a
# bounded hash-chained ring, and freezes a signed incident bundle (frames +
# state snapshot) on kill-switch events, engine death, or manual freeze.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.tape.plist"
AGENT_DIR="${HOME}/Library/LaunchAgents"
PLIST_DST="${AGENT_DIR}/com.badapple.tape.plist"

mkdir -p "${AGENT_DIR}" /var/lib/bad_apple/tape

# Keep heavy/regenerable state out of incident-bundle snapshots (and out of
# daily respawn checkpoints — same .fabricignore mechanism).
FABRICIGNORE="/var/lib/bad_apple/.fabricignore"
for pat in kv_cache generated_images install_backups tape/ grounded_index .tmp; do
    grep -qxF "${pat}" "${FABRICIGNORE}" 2>/dev/null || echo "${pat}" >> "${FABRICIGNORE}"
done

sed -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"

chmod 644 "${PLIST_DST}"

if launchctl list com.badapple.tape >/dev/null 2>&1; then
    launchctl unload "${PLIST_DST}" || true
fi
launchctl load -w "${PLIST_DST}"

echo "com.badapple.tape loaded from ${PLIST_DST}"
echo "Logs: /var/lib/bad_apple/tape.log"
