#!/usr/bin/env bash
set -euo pipefail

# Install and load the daily respawn state-snapshot agent for Bad Apple.
# The agent versions /var/lib/bad_apple once a day into a content-addressed
# store so platform state (ledgers, checkpoints, IFY state, cache) can be
# reverted atomically to any previous snapshot.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.respawn.plist"
AGENT_DIR="${HOME}/Library/LaunchAgents"
PLIST_DST="${AGENT_DIR}/com.badapple.respawn.plist"

mkdir -p "${AGENT_DIR}"

sed -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"

chmod 644 "${PLIST_DST}"

if launchctl list com.badapple.respawn >/dev/null 2>&1; then
    launchctl unload "${PLIST_DST}" || true
fi
launchctl load -w "${PLIST_DST}"

echo "com.badapple.respawn loaded from ${PLIST_DST}"
echo "Logs: /var/lib/bad_apple/respawn.log"
