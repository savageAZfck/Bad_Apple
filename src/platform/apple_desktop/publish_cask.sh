#!/usr/bin/env bash
set -euo pipefail

# Publish the Bad Apple Homebrew cask.
#
# Syncs the vendored cask (homebrew-bad-apple/Casks/bad-apple.rb) to the
# version in Cargo.toml and the sha256 in target/release/checksums.txt,
# then pushes the result to the public tap repo
# (savageAZfck/homebrew-bad-apple). Run after package_minimal_release.sh
# and after the release zip has been uploaded to bad-apple-releases.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CASK_SRC="${REPO_ROOT}/homebrew-bad-apple/Casks/bad-apple.rb"
CHECKSUMS="${REPO_ROOT}/target/release/checksums.txt"
TAP_REPO="https://github.com/savageAZfck/homebrew-bad-apple.git"
RELEASES_REPO="savageAZfck/bad-apple-releases"

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

VERSION="$(awk -F'"' '/^\[package\]/{p=1} p && /^version = /{print $2; exit}' "${REPO_ROOT}/Cargo.toml")"
[[ -n "${VERSION}" ]] || fail "could not read version from Cargo.toml"
ZIP_NAME="Bad_Apple-${VERSION}-unsigned.zip"

[[ -f "${CHECKSUMS}" ]] || fail "no checksums.txt; run package_minimal_release.sh first"
SHA="$(awk -v f="${ZIP_NAME}" '$2 == f {print $1}' "${CHECKSUMS}")"
[[ -n "${SHA}" ]] || fail "no sha256 for ${ZIP_NAME} in ${CHECKSUMS}"

# Refuse to publish a cask that points at a release asset that does not
# exist yet — the zip must be uploaded to bad-apple-releases first.
if ! gh release view "v${VERSION}" --repo "${RELEASES_REPO}" --json assets \
        --jq '.assets[].name' 2>/dev/null | grep -qx "${ZIP_NAME}"; then
    fail "v${VERSION} asset ${ZIP_NAME} not found on ${RELEASES_REPO}; upload the release first"
fi

# Update the vendored cask in place.
sed -i '' \
    -e "s/^  version \"[^\"]*\"/  version \"${VERSION}\"/" \
    -e "s/^  sha256 \"[^\"]*\"/  sha256 \"${SHA}\"/" \
    "${CASK_SRC}"
grep -q "version \"${VERSION}\"" "${CASK_SRC}" || fail "cask version update did not take"
grep -q "sha256 \"${SHA}\"" "${CASK_SRC}" || fail "cask sha256 update did not take"

# Sync the cask to the public tap repo.
TAP_DIR="$(mktemp -d)/homebrew-bad-apple"
trap 'rm -rf "$(dirname "${TAP_DIR}")"' EXIT
git clone --quiet "${TAP_REPO}" "${TAP_DIR}"
cp "${CASK_SRC}" "${TAP_DIR}/Casks/bad-apple.rb"

cd "${TAP_DIR}"
if git diff --quiet && git diff --cached --quiet; then
    echo "tap already at ${VERSION}; nothing to push"
else
    git add Casks/bad-apple.rb
    git commit -m "bad-apple ${VERSION}"
    git push origin HEAD
    echo "pushed cask ${VERSION} (${SHA:0:12}…) to ${TAP_REPO}"
fi

echo "vendored cask updated: version ${VERSION}, sha256 ${SHA}"
echo "remember to commit homebrew-bad-apple/Casks/bad-apple.rb in the main repo"
