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
cp -f "${REPO_ROOT}/src/platform/apple_desktop/strip_quarantine.sh" "${PKG_DIR}/"
cp -f "${REPO_ROOT}/src/platform/apple_desktop/install_badapple.sh" "${PKG_DIR}/"
chmod +x "${PKG_DIR}/install_badapple.sh"

# Double-clickable installer for people who do not want to use Terminal.
cat > "${PKG_DIR}/Install Bad Apple.command" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./install_badapple.sh
EOF
chmod +x "${PKG_DIR}/Install Bad Apple.command"

cat > "${PKG_DIR}/README.txt" <<'EOF'
Welcome to Bad Apple!

This is an unsigned build of Bad Apple.app. It does not require an Apple
Developer ID and never phones home to Apple for notarization. macOS Gatekeeper
will flag it the first time you open it, so follow one of the two paths below.

Quick path (recommended):
1. Unzip this archive if you have not already.
2. Copy "Bad Apple.app" into /Applications.
3. Double-click "Install Bad Apple" to finish setup. It will ask for your
   Mac password once so it can install the small background helper.
4. Launch "Bad Apple.app" from /Applications normally.

What happens when you double-click "Install Bad Apple":
- It checks that Bad Apple.app is in /Applications.
- It finds your Bad Apple source folder (set BADAPPLE_ROOT if you moved it).
- It runs the platform installer with administrator privileges.
- It shows a friendly "Installing Bad Apple..." dialog, then "Done." when
  the background helper is ready, or a clear error with the next step.

Manual path:
1. Copy "Bad Apple.app" into /Applications.
2. Control-click "Bad Apple.app" and choose "Open".
3. If prompted, click "Open Anyway" in System Settings > Privacy & Security > Security.

To install the full Bad Apple platform (LaunchDaemons, supervisor, etc.)
from a Bad Apple repository checkout instead, run:

    osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'

Replace /path/to/bad_apple with the path to your checkout if you moved it.

New in this build:
- One-click "Install Bad Apple" setup for non-technical users.
- First-run onboarding window from the menu bar.
- Plain-English Status window and friendly menu-bar tool tips.
- Self-hosting model registry: manage local models with `badapple model <list|scan|info|use|verify|add|remove|recommend>`.
- P2P edge mesh: discover peers and share signed model manifests with
  `badapple p2p <peers|sync|models|pull>`.
- Swift menu-bar Troubleshooting menu (Restart Daemon, Open Log, Copy MCP Socket).
- Local web Control Center served by the bundled `badapple-dashboard` on
  http://127.0.0.1:8787/control.
- All local IPC is authenticated with SLICKS v2 (Secure Enclave) with HMAC fallback.

To update later, use Bad Apple > Check for Updates in the menu bar, or run:

    sudo /Applications/Bad\ Apple.app/Contents/Resources/update_bad_apple.sh
EOF

rm -f "${ZIP_PATH}"
( cd "${PKG_DIR}" && zip -r -y "${ZIP_PATH}" "Bad Apple.app" README.txt strip_quarantine.sh install_badapple.sh "Install Bad Apple.command" )

echo "Packaged: ${ZIP_PATH}"
