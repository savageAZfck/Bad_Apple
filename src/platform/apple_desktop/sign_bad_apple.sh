#!/usr/bin/env bash
# Sign Bad Apple.app with a stable self-signed cert so Accessibility/TCC grants
# survive rebuilds. Creates the cert if it does not exist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
APP_DIR="${REPO_ROOT}/target/release/Bad Apple.app"
CERT_NAME="Bad Apple Dev (savag3)"
KEYCHAIN="${HOME}/Library/Keychains/badapple.keychain-db"

# Unlock the dev keychain and allow codesign to use it without prompting.
security unlock-keychain -p "" "${KEYCHAIN}" || true
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "" "${KEYCHAIN}" || true

if [[ ! -d "${APP_DIR}" ]]; then
    echo "Bad Apple.app not found at ${APP_DIR}; build it first." >&2
    exit 1
fi

# Create the dev cert if missing.
if ! security find-identity -v -p codesigning 2>/dev/null | grep -qF "${CERT_NAME}"; then
    echo "Developer certificate not found; creating..."
    "${REPO_ROOT}/src/platform/apple_desktop/create_dev_signing_cert.sh"
fi

# Sign the main app bundle and all embedded binaries.
codesign --deep --force --verify --verbose --sign "${CERT_NAME}" "${APP_DIR}"

echo "Signed: ${APP_DIR}"
