#!/usr/bin/env bash
set -euo pipefail

# Build the FireflyMenuBar macOS status-bar control app.
# This is a separate Swift/ObjC runtime that dynamically loads
# libsapient_soul.dylib at launch and uses the C FFI from firefly_core.h.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BUILD_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}/release"
mkdir -p "${BUILD_DIR}"

cd "${REPO_ROOT}/src/platform/apple_desktop"

echo "Building FireflyMenuBar..."
swift build -c release

# Locate the produced executable.
EXECUTABLE=$(find .build -type f -name FireflyMenuBar -perm +111 2>/dev/null | head -n 1)
if [[ -z "${EXECUTABLE}" ]]; then
    echo "Could not find built FireflyMenuBar executable" >&2
    exit 1
fi

cp "${EXECUTABLE}" "${BUILD_DIR}/FireflyMenuBar"
chmod +x "${BUILD_DIR}/FireflyMenuBar"

echo "Built: ${BUILD_DIR}/FireflyMenuBar"
echo "Run it with: ${BUILD_DIR}/FireflyMenuBar"
echo "(Ensure libsapient_soul.dylib and libFireflySiriBridge.dylib are in target/release or a known rpath.)"
