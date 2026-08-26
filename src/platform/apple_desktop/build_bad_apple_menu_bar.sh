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
mkdir -p "${BUILD_DIR}" "${MACOS_DIR}" "${FRAMEWORKS_DIR}" "${SCRATCH_DIR}/native"

# Compile directly so this user-session host can be built even when the active
# SwiftPM ManifestAPI predates macOS 26. The bridge module and dylib are emitted
# together, ensuring the AppIntent and voice host use one SLICKS implementation.
rm -rf "${SCRATCH_DIR}/native"
mkdir -p "${SCRATCH_DIR}/native"
SWIFTC=$(xcrun --find swiftc)
TARGET="arm64-apple-macosx26.0"
FRAMEWORK_SEARCH="${SDK_PATH}/System/Library/Frameworks"

"${SWIFTC}" \
    -parse-as-library -swift-version 5 -O \
    -target "${TARGET}" -sdk "${SDK_PATH}" \
    -emit-library -emit-module \
    -module-name BadAppleBridge \
    -emit-module-path "${SCRATCH_DIR}/native/BadAppleBridge.swiftmodule" \
    -o "${BUILD_DIR}/libBadAppleBridge.dylib" \
    "${REPO_ROOT}/src/platform/apple_bridge/BadAppleBridge.swift" \
    "${REPO_ROOT}/src/platform/apple_bridge/BadAppleIntent.swift" \
    -F "${FRAMEWORK_SEARCH}" \
    -framework AppIntents -framework CoreML -framework CryptoKit \
    -framework Foundation -framework FoundationModels -framework Security \
    -Xlinker -undefined -Xlinker dynamic_lookup \
    -Xlinker -install_name -Xlinker "@rpath/libBadAppleBridge.dylib"

# On macOS 26, TCC loads privacy usage descriptions from the binary's embedded
# __info_plist section before the bundle's Info.plist is fully mounted. Embed a
# minimal plist now so microphone/speech prompts fire reliably.
EMBED_PLIST="${SCRATCH_DIR}/native/BadAppleMenuBar-Info.plist"
cat > "${EMBED_PLIST}" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.badapple.app</string>
    <key>CFBundleName</key>
    <string>Bad Apple</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Bad Apple listens locally for “hey bad apple” and captures the following prompt.</string>
    <key>NSSpeechRecognitionUsageDescription</key>
    <string>Bad Apple uses only Apple’s on-device speech recognizer to transcribe local voice requests.</string>
    <key>NSSiriUsageDescription</key>
    <string>Bad Apple uses Siri to send spoken requests to its local cognitive substrate.</string>
</dict>
</plist>
PLIST
plutil -lint "${EMBED_PLIST}"

"${SWIFTC}" \
    -parse-as-library -swift-version 5 -O \
    -target "${TARGET}" -sdk "${SDK_PATH}" \
    -I "${SCRATCH_DIR}/native" -L "${BUILD_DIR}" \
    -o "${SCRATCH_DIR}/native/BadAppleMenuBar" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleMenuBar.swift" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleUIAccess.swift" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleMenuBarUIResponder.swift" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleControlCenter.swift" \
    -lBadAppleBridge -ldl \
    -framework AppKit -framework AVFoundation -framework Speech -framework AudioToolbox -framework ServiceManagement \
    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "${EMBED_PLIST}"

install -m 755 "${SCRATCH_DIR}/native/BadAppleMenuBar" "${MACOS_DIR}/BadApple"
install -m 755 "${SCRATCH_DIR}/native/BadAppleMenuBar" "${BUILD_DIR}/BadAppleMenuBar"
install -m 755 "${BUILD_DIR}/libBadAppleBridge.dylib" "${FRAMEWORKS_DIR}/libBadAppleBridge.dylib"
install -d "${CONTENTS_DIR}/Helpers"
install -m 755 "${BUILD_DIR}/badapple" "${CONTENTS_DIR}/Helpers/badapple" 2>/dev/null || true

# Screen capture helper runs as a child of the Bad Apple bundle so it uses
# Bad Apple's Screen Recording permission instead of the Python Aqua helper.
"${SWIFTC}" \
    -parse-as-library -swift-version 5 -O \
    -target "${TARGET}" -sdk "${SDK_PATH}" \
    -o "${SCRATCH_DIR}/native/BadAppleScreenCapture" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleScreenCapture.swift" \
    -framework AppKit -framework Foundation -framework ScreenCaptureKit
install -m 755 "${SCRATCH_DIR}/native/BadAppleScreenCapture" "${CONTENTS_DIR}/Helpers/BadAppleScreenCapture"

# UI automation helper runs as a child of the Bad Apple bundle so it uses
# Bad Apple's Accessibility permission to drive other apps via System Events.
"${SWIFTC}" \
    -parse-as-library -swift-version 5 -O \
    -target "${TARGET}" -sdk "${SDK_PATH}" \
    -o "${SCRATCH_DIR}/native/BadAppleUI" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleUI.swift" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleUIAccess.swift" \
    -framework Foundation
install -m 755 "${SCRATCH_DIR}/native/BadAppleUI" "${CONTENTS_DIR}/Helpers/BadAppleUI"
if [[ -f "${BUILD_DIR}/libbad_apple.dylib" ]]; then
    install -m 755 "${BUILD_DIR}/libbad_apple.dylib" "${FRAMEWORKS_DIR}/libbad_apple.dylib"
    install_name_tool -id "@rpath/libbad_apple.dylib" "${FRAMEWORKS_DIR}/libbad_apple.dylib" 2>/dev/null || true
fi
install_name_tool -add_rpath "@executable_path/../Frameworks" "${MACOS_DIR}/BadApple" 2>/dev/null || true

cat > "${CONTENTS_DIR}/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Bad Apple</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>BadApple</string>
    <key>CFBundleIdentifier</key>
    <string>com.badapple.app</string>
    <key>CFBundleLocalizations</key>
    <array>
        <string>en</string>
        <string>es-MX</string>
    </array>
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
    <key>NSMicrophoneUsageDescription</key>
    <string>Bad Apple listens locally for “hey bad apple” and captures the following prompt.</string>
    <key>NSSpeechRecognitionUsageDescription</key>
    <string>Bad Apple uses only Apple’s on-device speech recognizer to transcribe local voice requests.</string>
    <key>NSSiriUsageDescription</key>
    <string>Bad Apple uses Siri to send spoken requests to its local cognitive substrate.</string>
    <key>NSServices</key>
    <array>
        <dict>
            <key>NSMenuItem</key>
            <dict>
                <key>default</key>
                <string>Bad Apple: Rewrite</string>
            </dict>
            <key>NSMessage</key>
            <string>rewriteWithBadApple</string>
            <key>NSSendTypes</key>
            <array>
                <string>public.utf8-plain-text</string>
            </array>
            <key>NSReturnTypes</key>
            <array>
                <string>public.utf8-plain-text</string>
            </array>
        </dict>
        <dict>
            <key>NSMenuItem</key>
            <dict>
                <key>default</key>
                <string>Bad Apple: Summarize</string>
            </dict>
            <key>NSMessage</key>
            <string>summarizeWithBadApple</string>
            <key>NSSendTypes</key>
            <array>
                <string>public.utf8-plain-text</string>
            </array>
            <key>NSReturnTypes</key>
            <array>
                <string>public.utf8-plain-text</string>
            </array>
        </dict>
        <dict>
            <key>NSMenuItem</key>
            <dict>
                <key>default</key>
                <string>Bad Apple: Proofread</string>
            </dict>
            <key>NSMessage</key>
            <string>proofreadWithBadApple</string>
            <key>NSSendTypes</key>
            <array>
                <string>public.utf8-plain-text</string>
            </array>
            <key>NSReturnTypes</key>
            <array>
                <string>public.utf8-plain-text</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
PLIST

plutil -lint "${CONTENTS_DIR}/Info.plist"
codesign --force --deep --sign - "${APP_DIR}"

echo "Built: ${APP_DIR}"
echo "Install it in /Applications and launch it once so macOS indexes Execute Bad Apple."
