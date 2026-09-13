#!/usr/bin/env bash
set -euo pipefail

# Install and load the IFY watchdog agent for Bad Apple.
# The agent tails /var/lib/bad_apple/ledger.jsonl, verifies each new line
# against the hash chain, learns behavioral baselines, and surfaces
# findings through proposals and notifications. See IFY.md.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.ify.plist"
AGENT_DIR="${HOME}/Library/LaunchAgents"
PLIST_DST="${AGENT_DIR}/com.badapple.ify.plist"

mkdir -p "${AGENT_DIR}"

# Substitute placeholder tokens so the agent points at the native binary
# and the console user's home directory.
sed -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"

chmod 644 "${PLIST_DST}"

if launchctl list com.badapple.ify >/dev/null 2>&1; then
    launchctl unload "${PLIST_DST}" || true
fi
launchctl load -w "${PLIST_DST}"

echo "com.badapple.ify loaded from ${PLIST_DST}"
echo "Logs: /var/lib/bad_apple/ify.log"
echo "State: ${HOME}/.bad_apple/ify/"
