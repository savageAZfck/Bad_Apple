#!/usr/bin/env bash
set -euo pipefail

# Auto-update Bad Apple.app from a GitHub release, no Apple Developer ID required.
# The updater fetches the latest release, downloads the unsigned .zip, replaces
# the app in /Applications, strips the quarantine flag, and restarts the menu bar.
#
# Environment:
#   BADAPPLE_GH_REPO  - owner/repo on GitHub (default: savageAZfck/bad-apple-releases)
#   BADAPPLE_TAG      - specific tag to install, or "latest" (default: latest)
#
# The matching release asset must be named one of:
#   Bad_Apple-<version>-unsigned.zip
#   Bad_Apple-<tag>-unsigned.zip
#   Bad_Apple-<version>.zip

REPO="${BADAPPLE_GH_REPO:-savageAZfck/bad-apple-releases}"
TAG="${BADAPPLE_TAG:-latest}"
APP="/Applications/Bad Apple.app"

# Release-signing public key (cosign, ECDSA-P256). Pinned here — never
# fetched — so a compromised release repo cannot substitute its own key.
# cosign.pub in the source repo and on each release must match this.
COSIGN_PUBKEY='-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEYY34ZvQukW2U/Wb5pK+hl565k9H7
sA0iPnKZpqck92Rpwj/ALZ6YtJrWH7LNB2W0YKU+fHi4h/XIeyk1L7P1sg==
-----END PUBLIC KEY-----'

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || fail "curl is required"
command -v unzip >/dev/null || fail "unzip is required"
command -v shasum >/dev/null || fail "shasum is required; updates cannot proceed without SHA-256 verification"

installed_version() {
    if [[ ! -d "${APP}" ]]; then
        echo "not installed"
        return
    fi
    defaults read "${APP}/Contents/Info" CFBundleShortVersionString 2>/dev/null || echo "unknown"
}

resolve_latest() {
    local url="https://api.github.com/repos/${REPO}/releases/latest"
    local json
    json=$(curl -fsSL "${url}") || fail "could not fetch release info from ${url}"
    # Extract tag_name using only shell tools (no Python).
    echo "${json}" | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/'
}

verify_asset() {
    local out="$1" checksums_file="$2" name expected_sha actual_sha
    name="$(basename "${out}")"
    expected_sha="$(awk -v name="${name}" 'NF == 2 { file = $2; sub(/^\*/, "", file); if (file == name) print tolower($1) }' "${checksums_file}")"
    [[ "${expected_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "checksums.txt must contain exactly one valid SHA-256 entry for ${name}; this older or incomplete release cannot be updated safely"
    actual_sha="$(shasum -a 256 "${out}" | awk '{print $1}')"
    [[ "${actual_sha}" == "${expected_sha}" ]] || fail "checksum mismatch for ${name}: expected ${expected_sha}, got ${actual_sha}"
    echo "Checksum verified for ${name}." >&2
}

verify_signature() {
    local out="$1" tag="$2" name bundle pubkey_file
    name="$(basename "${out}")"
    bundle="${work}/${name}.sigstore.json"
    if ! curl -fsSL "https://github.com/${REPO}/releases/download/${tag}/${name}.sigstore.json" -o "${bundle}"; then
        echo "warning: no signature bundle for ${name}; falling back to checksum-only verification" >&2
        return 0
    fi
    if ! command -v cosign >/dev/null; then
        echo "warning: cosign not installed; checksum verified but signature not checked (brew install cosign for full verification)" >&2
        return 0
    fi
    pubkey_file="${work}/cosign.pub.pinned"
    printf '%s\n' "${COSIGN_PUBKEY}" > "${pubkey_file}"
    cosign verify-blob --key "${pubkey_file}" --bundle "${bundle}" "${out}" >/dev/null 2>&1 \
        || fail "signature verification FAILED for ${name}; refusing to install an artifact not signed by the pinned release key"
    echo "Signature verified for ${name} (pinned release key)." >&2
}

download_asset() {
    local tag="$1" version="${1#v}" name url out
    local tried=()
    local checksums_file="${work}/checksums.txt"
    curl -fsSL "https://github.com/${REPO}/releases/download/${tag}/checksums.txt" -o "${checksums_file}" || fail "checksums.txt is unavailable for ${tag}; refusing an unverified update. Older releases require a separately verified manual installation"
    for name in "Bad_Apple-${version}-unsigned.zip" "Bad_Apple-${tag}-unsigned.zip" "Bad_Apple-${version}.zip" "Bad_Apple-${tag}.zip"; do
        url="https://github.com/${REPO}/releases/download/${tag}/${name}"
        out="${work}/${name}"
        tried+=("${url}")
        if curl -fsSL "${url}" -o "${out}"; then
            verify_asset "${out}" "${checksums_file}"
            verify_signature "${out}" "${tag}"
            echo "${out}"
            return
        fi
    done
    fail "could not download a release asset. Tried:\n$(printf '  %s\n' "${tried[@]}")"
}

[[ "$(id -u)" -eq 0 ]] || fail "update must run as root so it can replace /Applications/Bad Apple.app"
[[ "${REPO}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || fail "invalid GitHub repository"

current=$(installed_version)
echo "Installed version: ${current}"

if [[ "${TAG}" == "latest" ]]; then
    TAG=$(resolve_latest)
    [[ -n "${TAG}" ]] || fail "could not determine latest release tag"
fi

new_version="${TAG#v}"
[[ "${new_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]] || fail "unsupported release tag: ${TAG}; expected v<version>"
echo "Latest release: ${TAG} (version ${new_version})"

if [[ "${current}" == "${new_version}" ]]; then
    runtime_program="$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' /Library/LaunchDaemons/com.badapple.mlx.plist 2>/dev/null || true)"
    case "${runtime_program}" in
        "/Library/Application Support/Bad Apple/runtimes/${new_version}."*/target/release/badapple-engine)
            if [[ -x "${runtime_program}" ]]; then
                echo "Bad Apple app and runtime are already up to date (${current})."
                exit 0
            fi
            ;;
    esac
    echo "App version matches, but its matching runtime is missing; repairing the complete platform."
fi

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "Downloading update..."
zip=$(download_asset "${TAG}")

echo "Extracting update..."
stage="${work}/stage"
mkdir "${stage}"
unzip -q "${zip}" -d "${stage}"
pkg_root="${stage}/Bad_Apple-${new_version}-unsigned"
[[ -x "${pkg_root}/install.sh" && -d "${pkg_root}/bad_apple" ]] || fail "release ${TAG} does not contain the complete source-free platform installer in Bad_Apple-${new_version}-unsigned. App-only/legacy releases cannot be auto-updated safely; no installed files were changed"
[[ -x "${pkg_root}/bad_apple/target/release/badapple-engine" ]] || fail "release ${TAG} is missing the native engine; no installed files were changed"
package_version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "${pkg_root}/Bad Apple.app/Contents/Info.plist")"
[[ "${package_version}" == "${new_version}" ]] || fail "package app version ${package_version} does not match release ${new_version}"

echo "Installing complete app and native platform..."
"${pkg_root}/install.sh"
echo "Updated Bad Apple to ${new_version}."
