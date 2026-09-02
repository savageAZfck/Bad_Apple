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

# Derive the bundle version from Cargo.toml so the updater can detect new
# releases.  Falls back to 0.1.0 if parsing fails.
BUNDLE_VERSION="$(awk -F'"' '/^\[package\]/{p=1} p && /^version = /{print $2; exit}' "${REPO_ROOT}/Cargo.toml")"
BUNDLE_VERSION="${BUNDLE_VERSION:-0.1.0}"

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
# Clean the app bundle so stale code signatures from prior signed runs do not
# leak into unsigned builds.
rm -rf "${APP_DIR}"
mkdir -p "${BUILD_DIR}" "${MACOS_DIR}" "${FRAMEWORKS_DIR}" "${SCRATCH_DIR}/native"

# Build the Apple Intelligence bridge (including the badapple-identity Secure
# Enclave helper) so the resulting app bundle and release package always contain
# the binaries that install_badapple_platform.sh hard-requires.
echo "Building Apple bridge (libBadAppleBridge.dylib + badapple-identity)..."
bash "${REPO_ROOT}/src/platform/apple_bridge/build_apple_bridge.sh"

# Compile directly so this user-session host can be built even when the active
# SwiftPM ManifestAPI predates macOS 26. The bridge module and dylib are emitted
# together, ensuring the AppIntent and voice host use one SLICKS implementation.
rm -rf "${SCRATCH_DIR}/native"
mkdir -p "${SCRATCH_DIR}/native"
SWIFTC=$(xcrun --find swiftc)
TARGET="arm64-apple-macosx26.0"
FRAMEWORK_SEARCH="${SDK_PATH}/System/Library/Frameworks"

# Build the Swift MLX inference module and acquire its matching Metal shaders.
MLX_INFERENCE_DIR="${REPO_ROOT}/src/platform/apple_desktop/MLXInference"
MLX_BUILD_DIR="${MLX_INFERENCE_DIR}/.build/arm64-apple-macosx/release"
MLX_DYLIB="${MLX_BUILD_DIR}/libBadAppleMLX.dylib"
MLX_MODULE_PATH="${MLX_BUILD_DIR}/Modules"
MLX_METAL_VERSION="0.31.1"
MLX_METAL_SHA256="198488eb61359e953580a9c4530400feee1a06dd2f28a930a6ffa58aec66a597"
MLX_METAL_CACHE="${HOME}/.cache/badapple/mlx-metal-${MLX_METAL_VERSION}/mlx.metallib"

echo "Building BadAppleMLX inference module..."
(cd "${MLX_INFERENCE_DIR}" && swift build -c release)

if [[ -f "${MLX_METAL_CACHE}" ]] && [[ "$(shasum -a 256 "${MLX_METAL_CACHE}" | awk '{print $1}')" != "${MLX_METAL_SHA256}" ]]; then
    rm -f "${MLX_METAL_CACHE}"
fi

if [[ ! -f "${MLX_METAL_CACHE}" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
    [[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python3)"
    TMP_METAL_DIR="$(mktemp -d)"
    "${PYTHON_BIN}" -m pip download --no-deps "mlx-metal==${MLX_METAL_VERSION}" -d "${TMP_METAL_DIR}"
    METAL_WHEEL=("${TMP_METAL_DIR}"/mlx_metal-*.whl)
    mkdir -p "${TMP_METAL_DIR}/extracted" "$(dirname "${MLX_METAL_CACHE}")"
    "${PYTHON_BIN}" -m zipfile -e "${METAL_WHEEL[0]}" "${TMP_METAL_DIR}/extracted"
    install -m 644 "${TMP_METAL_DIR}/extracted/mlx/lib/mlx.metallib" "${MLX_METAL_CACHE}"
    rm -rf "${TMP_METAL_DIR}"
fi

[[ "$(shasum -a 256 "${MLX_METAL_CACHE}" | awk '{print $1}')" == "${MLX_METAL_SHA256}" ]] || {
    echo "MLX Metal shader integrity verification failed." >&2
    exit 1
}

echo "Using BadAppleMLX dylib: ${MLX_DYLIB}"
echo "Using MLX Metal shaders: ${MLX_METAL_CACHE}"

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

# Build the menu bar app. Logic layer files (Security, Tools, Conversation, RAG)
# are pure Foundation and always included. BadAppleEngine requires the MLX
# dylib, so it's only included when MLX is available.
LOGIC_SOURCES=""
for src in BadAppleSecurity.swift BadAppleTools.swift BadAppleConversation.swift BadAppleRAG.swift BadAppleNativeRuntime.swift BadAppleAgent.swift; do
    [[ -f "${REPO_ROOT}/src/platform/apple_desktop/${src}" ]] && LOGIC_SOURCES="${LOGIC_SOURCES} ${REPO_ROOT}/src/platform/apple_desktop/${src}"
done

MLX_SOURCES=""
MLX_FLAGS=""
if [[ -n "${MLX_DYLIB}" && -f "${MLX_DYLIB}" ]]; then
    MLX_SOURCES="${REPO_ROOT}/src/platform/apple_desktop/BadAppleEngine.swift"
    # Swift modules are in Modules/. C module maps are in *.build/include/
    # and in the source checkouts (for C targets like _NumericsShims).
    # Filter out -tool duplicates and Swift module re-exports.
    MLX_INCLUDE_DIRS=$(
        {
            echo "${MLX_BUILD_DIR}/Modules"
            find "${MLX_BUILD_DIR}" -name "module.modulemap" -exec dirname {} \; \
                | grep -v -- "-tool" \
                | grep -v "ArgumentParser" \
                | grep -v "ArgumentParserToolInfo" \
                | sort -u
            # C module source include paths (for _NumericsShims, etc.)
            find "${MLX_INFERENCE_DIR}/.build/checkouts" -name "module.modulemap" -exec dirname {} \; \
                | sort -u
        } | sort -u | tr '\n' ':'
    )
    MLX_FLAGS=""
    IFS=':' read -ra MLX_DIRS <<< "${MLX_INCLUDE_DIRS}"
    for dir in "${MLX_DIRS[@]}"; do
        [[ -n "$dir" ]] && MLX_FLAGS="${MLX_FLAGS} -I ${dir}"
    done
    MLX_FLAGS="${MLX_FLAGS} -L ${MLX_BUILD_DIR} -lBadAppleMLX -Xlinker -rpath -Xlinker @executable_path/../Libraries"
    echo "Building with native MLX inference support."
fi

"${SWIFTC}" \
    -parse-as-library -swift-version 5 -O \
    -target "${TARGET}" -sdk "${SDK_PATH}" \
    -I "${SCRATCH_DIR}/native" -L "${BUILD_DIR}" \
    ${MLX_FLAGS} \
    -o "${SCRATCH_DIR}/native/BadAppleMenuBar" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleMenuBar.swift" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleUIAccess.swift" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleMenuBarUIResponder.swift" \
    "${REPO_ROOT}/src/platform/apple_desktop/BadAppleControlCenter.swift" \
    ${LOGIC_SOURCES} \
    ${MLX_SOURCES} \
    -lBadAppleBridge -ldl \
    -framework AppKit -framework AVFoundation -framework Speech -framework AudioToolbox -framework ServiceManagement \
    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "${EMBED_PLIST}"

# Copy the MLX runtime and its matching Metal shaders beside one another.
install -d "${CONTENTS_DIR}/Libraries"
install -m 755 "${MLX_DYLIB}" "${CONTENTS_DIR}/Libraries/libBadAppleMLX.dylib"
install -m 644 "${MLX_METAL_CACHE}" "${CONTENTS_DIR}/Libraries/mlx.metallib"
echo "Installed BadAppleMLX runtime and Metal shaders into app bundle Libraries."

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
install -d "${CONTENTS_DIR}/Resources"
install -m 755 "${REPO_ROOT}/src/platform/apple_desktop/update_bad_apple.sh" "${CONTENTS_DIR}/Resources/update_bad_apple.sh"
install -m 755 "${REPO_ROOT}/src/platform/apple_desktop/strip_quarantine.sh" "${CONTENTS_DIR}/Resources/strip_quarantine.sh"
install -m 755 "${REPO_ROOT}/src/platform/apple_desktop/install_badapple.sh" "${CONTENTS_DIR}/Resources/install_badapple.sh"
install -m 755 "${REPO_ROOT}/src/platform/apple_bridge/install_badapple_platform.sh" "${CONTENTS_DIR}/Resources/install_badapple_platform.sh"
install -m 644 "${REPO_ROOT}/badapple_aqua_helper.py" "${CONTENTS_DIR}/Resources/badapple_aqua_helper.py"

if [[ -f "${BUILD_DIR}/libbad_apple.dylib" ]]; then
    install -m 755 "${BUILD_DIR}/libbad_apple.dylib" "${FRAMEWORKS_DIR}/libbad_apple.dylib"
    install_name_tool -id "@rpath/libbad_apple.dylib" "${FRAMEWORKS_DIR}/libbad_apple.dylib" 2>/dev/null || true
fi
install_name_tool -add_rpath "@executable_path/../Frameworks" "${MACOS_DIR}/BadApple" 2>/dev/null || true

cat > "${CONTENTS_DIR}/Info.plist" <<PLIST
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
    <string>${BUNDLE_VERSION}</string>
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

SIGN_SCRIPT="${REPO_ROOT}/src/platform/apple_desktop/sign_bad_apple.sh"
if [[ "${BADAPPLE_NO_SIGN:-0}" == "1" ]]; then
    codesign --force --deep --sign - "${APP_DIR}"
    echo "Applied local ad-hoc signature (BADAPPLE_NO_SIGN=1)."
elif [[ -x "${SIGN_SCRIPT}" ]]; then
    if "${SIGN_SCRIPT}"; then
        echo "Signed: ${APP_DIR}"
    else
        rm -rf "${CONTENTS_DIR}/_CodeSignature"
        echo "Warning: code signing failed or no signing identity was available; continuing with unsigned .app bundle." >&2
    fi
else
    rm -rf "${CONTENTS_DIR}/_CodeSignature"
    echo "Warning: signing script not found at ${SIGN_SCRIPT}; continuing with unsigned .app bundle." >&2
fi

echo "Built: ${APP_DIR}"
echo "Install it in /Applications and launch it once so macOS indexes Execute Bad Apple."
