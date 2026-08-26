#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.menubar.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.badapple.menubar.plist"
APP_EXECUTABLE="/Applications/Bad Apple.app/Contents/MacOS/BadApple"

if [[ ! -x "${APP_EXECUTABLE}" ]]; then
    echo "Bad Apple is not installed at /Applications/Bad Apple.app" >&2
    exit 1
fi

mkdir -p "${HOME}/Library/LaunchAgents"
install -m 644 "${PLIST_SRC}" "${PLIST_DST}"
plutil -lint "${PLIST_DST}"

osascript -e 'tell application "Bad Apple" to quit' 2>/dev/null || true
launchctl unload "${PLIST_DST}" 2>/dev/null || true
launchctl load -w "${PLIST_DST}"

echo "Bad Apple menu bar is installed as a persistent login agent."
echo "LaunchAgent: ${PLIST_DST}"
