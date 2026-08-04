#!/usr/bin/env bash
set -euo pipefail

# Build the optional Apple Intelligence bridge as a dynamic library.
# This script is invoked manually after `cargo build --release`; it is not
# part of the Rust cargo pipeline because Swift/Objective-C cannot be built
# by cargo directly.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/target/release"
mkdir -p "${BUILD_DIR}"

# The FoundationModels framework lives in the macOS SDK; locate it so the
# compiler can resolve `import FoundationModels`.
SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
FRAMEWORK_SEARCH="${SDK_PATH}/System/Library/PrivateFrameworks"

echo "Building libFireflySiriBridge.dylib..."

swiftc \
    -emit-library \
    -o "${BUILD_DIR}/libFireflySiriBridge.dylib" \
    "${REPO_ROOT}/src/platform/apple_bridge/FireflySiriBridge.swift" \
    -module-name FireflySiriBridge \
    -framework Foundation \
    -F "${FRAMEWORK_SEARCH}" \
    -framework FoundationModels \
    -target arm64-apple-macos26.0 \
    -Xlinker -install_name \
    -Xlinker "@rpath/libFireflySiriBridge.dylib"

echo "Built: ${BUILD_DIR}/libFireflySiriBridge.dylib"
