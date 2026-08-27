#!/usr/bin/env bash
set -euo pipefail

# Build a drag-to-Applications DMG installer for Bad Apple.
# This produces a consumer-friendly .dmg that:
#   - contains Bad Apple.app
#   - contains the full bad_apple platform directory
#   - includes an Applications symlink
#   - includes an Install.command that strips quarantine and runs install.sh
#
# The DMG is unsigned and un-notarized. It is intended for users who have
# agreed to run unsigned software (same as the .zip). The Install.command
# removes the Gatekeeper quarantine flag from the app bundle before copying
# it to /Applications and installing the platform LaunchDaemons.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

VERSION="$(cd "${REPO_ROOT}" && grep '^version' Cargo.toml | head -n1 | sed -e 's/.*= *"//' -e 's/".*//')"
BUILD_DIR="${REPO_ROOT}/target/release"
PKG_NAME="Bad_Apple-${VERSION}-full"
DMG_NAME="Bad_Apple-${VERSION}.dmg"
STAGING="${BUILD_DIR}/${PKG_NAME}-dmg"
MOUNT_DIR="/Volumes/${PKG_NAME}"

# Make sure we have a full unsigned release package first.
"${REPO_ROOT}/src/platform/apple_desktop/package_full_release.sh"

rm -rf "${STAGING}"
mkdir -p "${STAGING}"

# The full package staging is the canonical source.
PKG_STAGING="${BUILD_DIR}/${PKG_NAME}"
if [[ ! -d "${PKG_STAGING}" ]]; then
    echo "error: full package staging not found at ${PKG_STAGING}" >&2
    exit 1
fi

cp -R "${PKG_STAGING}/Bad Apple.app" "${STAGING}/"
cp -R "${PKG_STAGING}/bad_apple" "${STAGING}/"
cp "${PKG_STAGING}/README.txt" "${STAGING}/README.txt"
cp "${PKG_STAGING}/strip_quarantine.sh" "${STAGING}/strip_quarantine.sh" || true

# Applications alias for drag-to-install.
ln -s /Applications "${STAGING}/Applications"

# Create the Install.command launcher.
cat > "${STAGING}/Install.command" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

DMG_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="${DMG_DIR}/Bad Apple.app"

echo "Stripping Gatekeeper quarantine from Bad Apple.app..."
/usr/bin/xattr -dr com.apple.quarantine "${APP}" 2>/dev/null || true

echo "Installing Bad Apple (this will prompt for your administrator password)..."
exec osascript -e "do shell script \"cp -Rf '${APP}' /Applications/ && '${DMG_DIR}/bad_apple/src/platform/apple_bridge/install_badapple_platform.sh' --install --unsigned-install\" with administrator privileges"
EOF
chmod +x "${STAGING}/Install.command"

# Optional background image and .DS_Store layout. Keep it simple for now.
# A future enhancement can add a custom .DS_Store background canvas.

echo "Creating DMG..."
rm -f "${BUILD_DIR}/${DMG_NAME}"

# Create a temporary read-write DMG.
SIZE_KB="$(du -sk "${STAGING}" | cut -f1)"
PADDING_KB=$((SIZE_KB + 20480))

tmp_dmg="$(mktemp -u /tmp/bad_apple_XXXXXX).dmg"
hdiutil create -srcfolder "${STAGING}" -volname "${PKG_NAME}" -fs HFS+J \
    -format UDRW -size "${PADDING_KB}k" -o "${tmp_dmg}" >/dev/null

# Compress to a read-only, internet-enabled DMG.
hdiutil convert "${tmp_dmg}" -format UDZO -o "${BUILD_DIR}/${DMG_NAME}" >/dev/null
rm -f "${tmp_dmg}"

echo "DMG: ${BUILD_DIR}/${DMG_NAME}"
