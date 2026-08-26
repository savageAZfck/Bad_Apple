#!/usr/bin/env bash
set -euo pipefail

# Build and package a signed, notarized release of Bad Apple.
# Requires:
#   - Apple Developer ID certificate installed in a keychain (CODESIGN_ID)
#   - Notarization credentials (APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD)
#
# Example:
#   CODESIGN_ID="Developer ID Application: Your Name (TEAMID)" \
#   APPLE_ID="you@example.com" \
#   APPLE_TEAM_ID="TEAMID" \
#   APPLE_APP_PASSWORD="abcd-1234-abcd-1234" \
#   src/platform/apple_desktop/package_signed_release.sh

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

VERSION="$(cd "${REPO_ROOT}" && grep '^version' Cargo.toml | head -n1 | sed -e 's/.*= *"//' -e 's/".*//')"
BUILD_DIR="${REPO_ROOT}/target/release"
APP_DIR="${BUILD_DIR}/Bad Apple.app"
PKG_DIR="${BUILD_DIR}/Bad_Apple-${VERSION}-full-signed"
ZIP_PATH="${BUILD_DIR}/Bad_Apple-${VERSION}-full-signed.zip"
SUBMIT_LOG="${BUILD_DIR}/notarytool-submit.log"

CODESIGN_ID="${CODESIGN_ID:-}"
APPLE_ID="${APPLE_ID:-}"
APPLE_TEAM_ID="${APPLE_TEAM_ID:-}"
APPLE_APP_PASSWORD="${APPLE_APP_PASSWORD:-}"

if [[ -z "${CODESIGN_ID}" ]]; then
    echo "ERROR: CODESIGN_ID is not set." >&2
    echo "Set it to your Apple Developer ID Application identity." >&2
    exit 1
fi

echo "Building release binaries..."
cargo build --release

echo "Building Bad Apple.app..."
"${REPO_ROOT}/src/platform/apple_desktop/build_bad_apple_menu_bar.sh"

[[ -d "${APP_DIR}" ]] || { echo "error: Bad Apple.app not found at ${APP_DIR}" >&2; exit 1; }

echo "Signing app bundle, binaries, and libraries..."
find "${APP_DIR}/Contents" -type f \( -name "BadApple" -o -name "BadAppleUI" -o -name "BadAppleScreenCapture" -o -name "*.dylib" \) -exec \
    codesign --force --options runtime --timestamp --sign "${CODESIGN_ID}" {} \;

codesign --deep --force --options runtime --timestamp --sign "${CODESIGN_ID}" "${APP_DIR}"

codesign --verify --deep --strict "${APP_DIR}"

rm -rf "${PKG_DIR}"
mkdir -p "${PKG_DIR}"

# Re-use the full-release packager, then re-sign anything it adds.
BADAPPLE_SIGN=0 "${REPO_ROOT}/src/platform/apple_desktop/package_full_release.sh"

# package_full_release.sh produces an unsigned zip; replace the .app with the signed one.
rm -rf "${BUILD_DIR}/Bad_Apple-${VERSION}-full/Bad Apple.app"
ditto "${APP_DIR}" "${BUILD_DIR}/Bad_Apple-${VERSION}-full/Bad Apple.app"

# Re-sign the embedded repo binaries.
for bin in badapple gatekeeper badapple-identity; do
    if [[ -x "${BUILD_DIR}/Bad_Apple-${VERSION}-full/bad_apple/target/release/${bin}" ]]; then
        codesign --force --options runtime --timestamp --sign "${CODESIGN_ID}" \
            "${BUILD_DIR}/Bad_Apple-${VERSION}-full/bad_apple/target/release/${bin}"
    fi
done

rm -f "${ZIP_PATH}"
( cd "${BUILD_DIR}" && ditto -c -k --keepParent "Bad_Apple-${VERSION}-full" "${ZIP_PATH}" )

if [[ -n "${APPLE_ID}" && -n "${APPLE_TEAM_ID}" && -n "${APPLE_APP_PASSWORD}" ]]; then
    echo "Submitting to Apple Notary Service..."
    xcrun notarytool submit "${ZIP_PATH}" \
        --apple-id "${APPLE_ID}" \
        --team-id "${APPLE_TEAM_ID}" \
        --password "${APPLE_APP_PASSWORD}" \
        --wait > "${SUBMIT_LOG}" 2>&1
    cat "${SUBMIT_LOG}"
    REQUEST_ID="$(grep -oE 'id: [a-f0-9-]+' "${SUBMIT_LOG}" | head -n1 | awk '{print $2}')"
    if [[ -n "${REQUEST_ID}" ]]; then
        xcrun notarytool log "${REQUEST_ID}" \
            --apple-id "${APPLE_ID}" \
            --team-id "${APPLE_TEAM_ID}" \
            --password "${APPLE_APP_PASSWORD}" \
            "${BUILD_DIR}/notarytool-log.json"
        echo "Notarization log saved to ${BUILD_DIR}/notarytool-log.json"
    fi
    xcrun stapler staple "${APP_DIR}"
else
    echo "WARNING: notarization skipped; set APPLE_ID, APPLE_TEAM_ID, and APPLE_APP_PASSWORD to notarize." >&2
fi

echo "Packaged: ${ZIP_PATH}"
