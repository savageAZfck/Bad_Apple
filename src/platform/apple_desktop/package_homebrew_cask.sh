#!/usr/bin/env bash
set -euo pipefail

# Prepare a Homebrew tap for the current release zip.
# The canonical Cask lives in the repo at homebrew-bad-apple/Casks/bad-apple.rb
# and points at GitHub Releases. This script creates a local tap in
# target/release/homebrew-bad-apple that points at the freshly built unsigned
# zip for local testing.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

VERSION="$(cd "${REPO_ROOT}" && grep '^version' Cargo.toml | head -n1 | sed -e 's/.*= *"//' -e 's/".*//')"
ZIP_NAME="Bad_Apple-${VERSION}-unsigned.zip"
ZIP_PATH="${REPO_ROOT}/target/release/${ZIP_NAME}"

if [[ ! -f "${ZIP_PATH}" ]]; then
    # Fall back to the dev-only full source zip for local testing.
    ZIP_NAME="Bad_Apple-${VERSION}-full-unsigned.zip"
    ZIP_PATH="${REPO_ROOT}/target/release/${ZIP_NAME}"
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "error: release zip not found at ${ZIP_PATH}; run package_minimal_release.sh or package_full_release.sh first" >&2
    exit 1
fi

SHA256="$(shasum -a 256 "${ZIP_PATH}" | cut -d' ' -f1)"

LOCAL_TAP="${REPO_ROOT}/target/release/homebrew-bad-apple"
rm -rf "${LOCAL_TAP}"
mkdir -p "${LOCAL_TAP}/Casks"

cp "${REPO_ROOT}/homebrew-bad-apple/README.md" "${LOCAL_TAP}/README.md"

sed -e 's#REPLACE_SHA256#'"${SHA256}"'#' \
    -e 's|url "https://github.com/savageAZfck/bad-apple-releases/releases/download/v#{version}/Bad_Apple-#{version}-unsigned.zip"|url "file://'"${ZIP_PATH}"'"|' \
    "${REPO_ROOT}/homebrew-bad-apple/Casks/bad-apple.rb" > "${LOCAL_TAP}/Casks/bad-apple.rb"

# Make the local tap look like a real git repo so brew tap is happy.
(cd "${LOCAL_TAP}" && git init --quiet && git add . && git commit -m "Bad Apple ${VERSION}" --quiet) || true

echo "Local Homebrew tap: ${LOCAL_TAP}"
echo ""
echo "To test:"
echo "  brew tap local/bad-apple ${LOCAL_TAP}"
echo "  brew install --cask bad-apple"
echo ""
echo "To use the GitHub release tap instead:"
echo "  brew tap savageAZfck/bad-apple https://github.com/savageAZfck/homebrew-bad-apple"
echo "  brew install --cask bad-apple"
