#!/usr/bin/env bash
set -euo pipefail

# Install and load the local web dashboard (Control Center) LaunchAgent.
# The dashboard binds to loopback only and serves the bundled web assets.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_desktop/com.badapple.dashboard.plist"
APP_BUNDLE="/Applications/Bad Apple.app"
DASHBOARD_BIN="${APP_BUNDLE}/Contents/Helpers/badapple-dashboard"
WEB_ROOT="${APP_BUNDLE}/Contents/Resources/web"

if [[ ! -x "${DASHBOARD_BIN}" ]]; then
    echo "error: dashboard binary not found at ${DASHBOARD_BIN}" >&2
    exit 1
fi

if [[ ! -d "${WEB_ROOT}" ]]; then
    echo "error: dashboard web assets not found at ${WEB_ROOT}" >&2
    exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
    CONSOLE_USER="${SUDO_USER:-$(logname 2>/dev/null || id -un)}"
    CONSOLE_HOME="$(eval echo "~${CONSOLE_USER}")"
    PLIST_DST="${CONSOLE_HOME}/Library/LaunchAgents/com.badapple.dashboard.plist"
    CONSOLE_UID="$(id -u "${CONSOLE_USER}")"
    RUN_AS="launchctl asuser ${CONSOLE_UID} sudo -u ${CONSOLE_USER} -E HOME=${CONSOLE_HOME}"
else
    CONSOLE_USER="$(id -un)"
    CONSOLE_HOME="${HOME}"
    PLIST_DST="${CONSOLE_HOME}/Library/LaunchAgents/com.badapple.dashboard.plist"
    RUN_AS=""
fi

mkdir -p "${CONSOLE_HOME}/Library/LaunchAgents"

sed -e "s|__APP_BUNDLE__|${APP_BUNDLE}|g" \
    -e "s|__CONSOLE_HOME__|${CONSOLE_HOME}|g" \
    -e "s|__CONSOLE_USER__|${CONSOLE_USER}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"
chmod 644 "${PLIST_DST}"
plutil -lint "${PLIST_DST}"

if [[ -n "${RUN_AS}" ]]; then
    ${RUN_AS} launchctl unload "${PLIST_DST}" 2>/dev/null || true
    ${RUN_AS} launchctl load -w "${PLIST_DST}"
else
    launchctl unload "${PLIST_DST}" 2>/dev/null || true
    launchctl load -w "${PLIST_DST}"
fi

echo "Bad Apple dashboard is installed as a persistent login agent."
echo "LaunchAgent: ${PLIST_DST}"
echo "URL: http://127.0.0.1:8787/control"
echo "Log: /tmp/badapple_dashboard.log"
