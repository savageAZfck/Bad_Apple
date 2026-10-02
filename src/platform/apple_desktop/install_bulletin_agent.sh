#!/usr/bin/env bash
set -euo pipefail

# Install and load the daily oversight-bulletin agent for Bad Apple. The
# agent classifies new ledger entries by authority class, summarizes council
# dissent and IFY findings, and writes a signed bulletin artifact to
# /var/lib/bad_apple/bulletins/.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.bulletin.plist"
AGENT_DIR="${HOME}/Library/LaunchAgents"
PLIST_DST="${AGENT_DIR}/com.badapple.bulletin.plist"

mkdir -p "${AGENT_DIR}"

# Substitute placeholder tokens so the agent points at the native binary
# and the console user's home directory.
sed -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"

chmod 644 "${PLIST_DST}"

if launchctl list com.badapple.bulletin >/dev/null 2>&1; then
    launchctl unload "${PLIST_DST}" || true
fi
launchctl load -w "${PLIST_DST}"

echo "com.badapple.bulletin loaded from ${PLIST_DST}"
echo "Logs: /var/lib/bad_apple/bulletin.log"
