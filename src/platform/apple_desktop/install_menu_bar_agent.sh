#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.menubar.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.badapple.menubar.plist"
APP_EXECUTABLE="/Applications/Bad Apple.app/Contents/MacOS/BadApple"

# Tune memory-governor thresholds to the machine's total unified memory.
RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
if [[ "$RAM_GB" -ge 24 ]]; then
    BADAPPLE_MEMORY_ELEVATED=0.80
    BADAPPLE_MEMORY_UNHEALTHY=0.90
    BADAPPLE_MEMORY_CRITICAL=0.95
elif [[ "$RAM_GB" -ge 12 ]]; then
    BADAPPLE_MEMORY_ELEVATED=0.75
    BADAPPLE_MEMORY_UNHEALTHY=0.85
    BADAPPLE_MEMORY_CRITICAL=0.90
else
    BADAPPLE_MEMORY_ELEVATED=0.70
    BADAPPLE_MEMORY_UNHEALTHY=0.80
    BADAPPLE_MEMORY_CRITICAL=0.85
fi

if [[ ! -x "${APP_EXECUTABLE}" ]]; then
    echo "error: Bad Apple.app is not installed at /Applications/Bad Apple.app" >&2
    echo "       Run the installer first: sudo ./install.sh" >&2
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
sed -e "s|__BADAPPLE_ROOT__|${REPO_ROOT}|g" \
    -e "s|__BADAPPLE_MEMORY_ELEVATED__|${BADAPPLE_MEMORY_ELEVATED}|g" \
    -e "s|__BADAPPLE_MEMORY_UNHEALTHY__|${BADAPPLE_MEMORY_UNHEALTHY}|g" \
    -e "s|__BADAPPLE_MEMORY_CRITICAL__|${BADAPPLE_MEMORY_CRITICAL}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"
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
