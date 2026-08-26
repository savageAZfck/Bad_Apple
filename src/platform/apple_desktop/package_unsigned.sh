#!/usr/bin/env bash
set -euo pipefail

# Build Bad Apple.app unsigned and package it into a consumer .zip.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

VERSION="$(awk -F'"' '/^\[package\]/{p=1} p && /^version = /{print $2; exit}' Cargo.toml)"
BUILD_DIR="${REPO_ROOT}/target/release"
APP_DIR="${BUILD_DIR}/Bad Apple.app"
ZIP_PATH="${BUILD_DIR}/Bad_Apple-${VERSION}-unsigned.zip"

echo "Building release binaries..."
cargo build --release

echo "Building Bad Apple.app without code signing..."
BADAPPLE_NO_SIGN=1 "${REPO_ROOT}/src/platform/apple_desktop/build_bad_apple_menu_bar.sh"

[[ -d "${APP_DIR}" ]] || { echo "error: Bad Apple.app not found at ${APP_DIR}" >&2; exit 1; }

STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT

PKG_DIR="${STAGING}/pkg"
mkdir -p "${PKG_DIR}"

ditto "${APP_DIR}" "${PKG_DIR}/Bad Apple.app"

cat > "${PKG_DIR}/README.txt" <<'EOF'
Welcome to Bad Apple!

This is an unsigned build of Bad Apple.app. It does not require an Apple
Developer ID, but macOS Gatekeeper will warn you the first time you open it.

To run Bad Apple.app:
1. Unzip this archive if you have not already.
2. In Finder, locate "Bad Apple.app".
3. Right-click (or Control-click) "Bad Apple.app" and choose "Open".
4. If a Gatekeeper dialog appears, click "Open".
   On macOS Sequoia or later, you may instead need to open System Settings >
   Privacy & Security > Security and click "Open Anyway" after your first
   launch attempt.

To install the full Bad Apple platform (LaunchDaemons, supervisor, etc.),
place Bad Apple.app in /Applications, then run the platform installer from
your Bad Apple repository checkout:

    osascript -e 'do shell script "cd /Users/savag3/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'

Replace /Users/savag3/bad_apple with the path to your checkout if you moved it.
EOF

rm -f "${ZIP_PATH}"
( cd "${PKG_DIR}" && zip -r -y "${ZIP_PATH}" "Bad Apple.app" README.txt )

echo "Packaged: ${ZIP_PATH}"
