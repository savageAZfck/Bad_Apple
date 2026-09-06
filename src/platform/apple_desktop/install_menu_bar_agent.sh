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

if [[ "$(id -u)" -eq 0 ]]; then
    CONSOLE_USER="${SUDO_USER:-$(logname 2>/dev/null || id -un)}"
    CONSOLE_HOME=$(eval echo "~${CONSOLE_USER}")
    PLIST_DST="${CONSOLE_HOME}/Library/LaunchAgents/com.badapple.menubar.plist"
    CONSOLE_UID=$(id -u "${CONSOLE_USER}")
    RUN_AS="launchctl asuser ${CONSOLE_UID} sudo -u ${CONSOLE_USER} -E HOME=${CONSOLE_HOME}"
else
    CONSOLE_USER="$(id -un)"
    CONSOLE_HOME="${HOME}"
    CONSOLE_UID="$(id -u)"
    RUN_AS=""
fi

mkdir -p "${CONSOLE_HOME}/Library/LaunchAgents"
install -m 644 "${PLIST_SRC}" "${PLIST_DST}"
plutil -lint "${PLIST_DST}"

if [[ -n "${RUN_AS}" ]]; then
    ${RUN_AS} osascript -e 'tell application "Bad Apple" to quit' 2>/dev/null || true
    ${RUN_AS} launchctl unload "${PLIST_DST}" 2>/dev/null || true
    ${RUN_AS} launchctl load -w "${PLIST_DST}"
else
    osascript -e 'tell application "Bad Apple" to quit' 2>/dev/null || true
    launchctl unload "${PLIST_DST}" 2>/dev/null || true
    launchctl load -w "${PLIST_DST}"
fi

echo "Bad Apple menu bar is installed as a persistent login agent."
echo "LaunchAgent: ${PLIST_DST}"
