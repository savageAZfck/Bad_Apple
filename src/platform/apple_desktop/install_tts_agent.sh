#!/usr/bin/env bash
set -euo pipefail

# Install and load the local neural TTS agent for Bad Apple.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.tts.plist"
AGENT_DIR="${HOME}/Library/LaunchAgents"
PLIST_DST="${AGENT_DIR}/com.badapple.tts.plist"

mkdir -p "${AGENT_DIR}"

# Substitute the current repo root and home directory into the plist, so the
# agent points at the right venv and voices directory on this machine.
sed -e "s|/Users/savag3/bad_apple|${REPO_ROOT}|g" \
    -e "s|/Users/savag3|${HOME}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"

chmod 644 "${PLIST_DST}"

if launchctl list com.badapple.tts >/dev/null 2>&1; then
    launchctl unload "${PLIST_DST}" || true
fi
launchctl load -w "${PLIST_DST}"

echo "com.badapple.tts loaded from ${PLIST_DST}"
echo "Logs: /tmp/badapple_tts.log"
echo "Socket: /tmp/badapple_tts.sock"
