#!/usr/bin/env bash
set -euo pipefail

# Install and load the daily sovereign-ledger checkpoint agent for Bad Apple.
# The agent re-verifies /var/lib/bad_apple/ledger.jsonl once a day, rewrites
# the hardened copy, and re-signs both tips through the identity agent.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.checkpoint.plist"
AGENT_DIR="${HOME}/Library/LaunchAgents"
PLIST_DST="${AGENT_DIR}/com.badapple.checkpoint.plist"

mkdir -p "${AGENT_DIR}"

# Substitute placeholder tokens so the agent points at the native binary
# and the console user's home directory.
sed -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"

chmod 644 "${PLIST_DST}"

if launchctl list com.badapple.checkpoint >/dev/null 2>&1; then
    launchctl unload "${PLIST_DST}" || true
fi
launchctl load -w "${PLIST_DST}"

echo "com.badapple.checkpoint loaded from ${PLIST_DST}"
echo "Logs: /var/lib/bad_apple/checkpoint.log"
