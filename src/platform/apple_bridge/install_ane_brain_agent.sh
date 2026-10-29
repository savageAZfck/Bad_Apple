#!/usr/bin/env bash
set -euo pipefail

# Install and load the ANE brain fetch agent. Runs once at login to pull
# the 4B Neural Engine artifact into ~/.bad_apple/ane/ when downloads are
# allowed. The fetch is idempotent — it no-ops if the manifest exists.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_bridge/com.badapple.ane_brain.plist"
AGENT_DIR="${HOME}/Library/LaunchAgents"
PLIST_DST="${AGENT_DIR}/com.badapple.ane_brain.plist"

mkdir -p "${AGENT_DIR}"

sed -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"

chmod 644 "${PLIST_DST}"

if launchctl list com.badapple.ane_brain >/dev/null 2>&1; then
    launchctl unload "${PLIST_DST}" || true
fi
launchctl load -w "${PLIST_DST}"

echo "com.badapple.ane_brain loaded from ${PLIST_DST}"
echo "Logs: /tmp/badapple_ane_brain.log"
