#!/usr/bin/env bash
set -euo pipefail

# Build the optional Apple Intelligence bridge as a dynamic library.
# This script is invoked manually after `cargo build --release`; it is not
# part of the Rust cargo pipeline because Swift/Objective-C cannot be built
# by cargo directly.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}"
BUILD_DIR="${CARGO_TARGET_DIR}/release"
mkdir -p "${BUILD_DIR}"

# Pick an SDK that actually contains FoundationModels.  The active
# `xcrun --sdk macosx --show-sdk-path` may be the CommandLineTools SDK, which
# is often missing the framework.  Fall back to the Xcode macOS SDK if needed.
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
FRAMEWORK_SEARCH="${SDK_PATH}/System/Library/Frameworks"

echo "Using SDK: ${SDK_PATH}"
echo "Building libBadAppleBridge.dylib..."

swiftc \
    -emit-library \
    -sdk "${SDK_PATH}" \
    -o "${BUILD_DIR}/libBadAppleBridge.dylib" \
    "${REPO_ROOT}/src/platform/apple_bridge/BadAppleBridge.swift" \
    "${REPO_ROOT}/src/platform/apple_bridge/BadAppleIntent.swift" \
    -module-name BadAppleBridge \
    -framework Foundation \
    -F "${FRAMEWORK_SEARCH}" \
    -framework AppIntents \
    -framework CoreML \
    -framework CryptoKit \
    -framework FoundationModels \
    -framework Security \
    -target arm64-apple-macos26.0 \
    -Xlinker -undefined \
    -Xlinker dynamic_lookup \
    -Xlinker -install_name \
    -Xlinker "@rpath/libBadAppleBridge.dylib"

echo "Built: ${BUILD_DIR}/libBadAppleBridge.dylib"

swiftc \
    -O \
    -sdk "${SDK_PATH}" \
    -o "${BUILD_DIR}/badapple-identity" \
    "${REPO_ROOT}/src/platform/apple_bridge/BadAppleIdentity.swift" \
    -framework CryptoKit \
    -framework Foundation \
    -framework LocalAuthentication \
    -framework Security \
    -target arm64-apple-macos26.0
codesign --force --sign - "${BUILD_DIR}/badapple-identity"
echo "Built: ${BUILD_DIR}/badapple-identity"

echo "Building badapple-identity-agent..."

swiftc \
    -O \
    -sdk "${SDK_PATH}" \
    -o "${BUILD_DIR}/badapple-identity-agent" \
    "${REPO_ROOT}/src/platform/apple_bridge/BadAppleIdentityAgent.swift" \
    -framework CryptoKit \
    -framework Foundation \
    -framework LocalAuthentication \
    -framework Security \
    -target arm64-apple-macos26.0
codesign --force --sign - "${BUILD_DIR}/badapple-identity-agent"
echo "Built: ${BUILD_DIR}/badapple-identity-agent"
