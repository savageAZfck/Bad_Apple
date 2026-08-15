#!/usr/bin/env bash
set -euo pipefail

# Build the Bad Apple menu-bar host app that exposes BadAppleIntent to Siri.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BUILD_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}/release"
SCRATCH_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}/swiftpm-desktop"
APP_DIR="${BUILD_DIR}/Bad Apple.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
FRAMEWORKS_DIR="${CONTENTS_DIR}/Frameworks"

resolve_sdk() {
    local active_sdk
    active_sdk=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)
    if [[ -d "${active_sdk}/System/Library/Frameworks/FoundationModels.framework" ]]; then
        echo "${active_sdk}"
        return
    fi
    for sdk in \
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk" \
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
    do
        if [[ -d "${sdk}/System/Library/Frameworks/FoundationModels.framework" ]]; then
            echo "${sdk}"
            return
        fi
    done
    echo "No macOS SDK with FoundationModels.framework found." >&2
    exit 1
}

SDK_PATH=$(resolve_sdk)
mkdir -p "${BUILD_DIR}" "${MACOS_DIR}" "${FRAMEWORKS_DIR}"

cd "${REPO_ROOT}/src/platform/apple_desktop"
SDKROOT="${SDK_PATH}" swift build \
    -c release \
    --scratch-path "${SCRATCH_DIR}" \
    --triple arm64-apple-macosx26.0
BIN_DIR=$(SDKROOT="${SDK_PATH}" swift build \
    -c release \
    --scratch-path "${SCRATCH_DIR}" \
    --triple arm64-apple-macosx26.0 \
    --show-bin-path)

install -m 755 "${BIN_DIR}/BadAppleMenuBar" "${MACOS_DIR}/BadApple"
install -m 755 "${BIN_DIR}/BadAppleMenuBar" "${BUILD_DIR}/BadAppleMenuBar"
if [[ -f "${BUILD_DIR}/libBadAppleBridge.dylib" ]]; then
    install -m 755 "${BUILD_DIR}/libBadAppleBridge.dylib" "${FRAMEWORKS_DIR}/libBadAppleBridge.dylib"
elif [[ -f "${BIN_DIR}/libBadAppleBridge.dylib" ]]; then
    install -m 755 "${BIN_DIR}/libBadAppleBridge.dylib" "${FRAMEWORKS_DIR}/libBadAppleBridge.dylib"
fi
install_name_tool -add_rpath "@executable_path/../Frameworks" "${MACOS_DIR}/BadApple" 2>/dev/null || true

cat > "${CONTENTS_DIR}/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Bad Apple</string>
    <key>CFBundleExecutable</key>
    <string>BadApple</string>
    <key>CFBundleIdentifier</key>
    <string>com.badapple.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Bad Apple</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>26.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSSiriUsageDescription</key>
    <string>Bad Apple uses Siri to send spoken requests to its local cognitive substrate.</string>
</dict>
</plist>
PLIST

plutil -lint "${CONTENTS_DIR}/Info.plist"
codesign --force --deep --sign - "${APP_DIR}"

echo "Built: ${APP_DIR}"
echo "Install it in /Applications and launch it once so macOS indexes Execute Bad Apple."
