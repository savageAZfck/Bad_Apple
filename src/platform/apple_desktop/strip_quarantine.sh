#!/usr/bin/env bash
set -euo pipefail

# Strip the Gatekeeper quarantine flag from the unsigned Bad Apple.app bundle.
# This lets the app launch with a normal double-click instead of requiring
# Control-click -> Open on first launch. No Apple Developer ID required.

APP="/Applications/Bad Apple.app"

if [[ ! -d "${APP}" ]]; then
    echo "error: ${APP} not found. Copy the unsigned zip to /Applications first." >&2
    exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Stripping quarantine requires root. Re-run with sudo." >&2
    exit 1
fi

xattr -dr com.apple.quarantine "${APP}"
echo "Quarantine flag removed from ${APP}. Double-click to open."
