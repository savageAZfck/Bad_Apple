#!/usr/bin/env bash
set -euo pipefail

# Sign Bad Apple release artifacts with cosign (key-pair, Sigstore bundle
# output — same .sigstore.json format sovereign_ledger's CI produces).
#
# The private key is generated once by --init and stored in the macOS
# Keychain (service com.badapple.release-sign), never left on disk. The
# public key is written to cosign.pub at the repo root and must be
# published: it is shipped as a release asset and pinned inside
# update_bad_apple.sh, so verifiers never have to fetch it from an
# untrusted channel.
#
# Usage:
#   sign_release.sh --init                  # generate keypair -> Keychain, write cosign.pub
#   sign_release.sh [files...]            # sign files (default: target/release zips + checksums.txt)
#   sign_release.sh --tag v0.2.2          # sign + upload bundles to that GitHub release
#   sign_release.sh --retro               # sign every existing release's assets
#
# Environment:
#   BADAPPLE_RELEASES_REPO - release repo (default: savageAZfck/bad-apple-releases)
#   BADAPPLE_NO_KEYCHAIN   - set to 1 to read the key from ./cosign.key instead

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RELEASES_REPO="${BADAPPLE_RELEASES_REPO:-savageAZfck/bad-apple-releases}"
KC_SERVICE="com.badapple.release-sign"
DIST_DIR="${REPO_ROOT}/target/release"
PUBKEY="${REPO_ROOT}/cosign.pub"

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v cosign >/dev/null || fail "cosign is required (brew install cosign)"
command -v shasum >/dev/null || fail "shasum is required"

# Generic-password items whose data contains newlines come back from
# `security -w` hex-encoded, so the PEM is stored base64 (single printable
# line) and decoded on read.
kc_get() {
    security find-generic-password -s "${KC_SERVICE}" -a "$1" -w 2>/dev/null
}

kc_get_b64() {
    kc_get "$1" | base64 --decode
}

kc_put() {
    # -A: any app may read without prompting (non-interactive release runs);
    # -U: update in place if the item already exists.
    security add-generic-password -s "${KC_SERVICE}" -a "$1" -w "$2" -A -U >/dev/null
}

init_key() {
    [[ -f "${PUBKEY}" ]] && fail "cosign.pub already exists; refusing to rotate the release key silently (delete it deliberately first)"
    local work password
    work=$(mktemp -d)
    password=$(openssl rand -hex 32)
    (cd "${work}" && COSIGN_PASSWORD="${password}" cosign generate-key-pair >/dev/null)
    kc_put cosign-key "$(base64 < "${work}/cosign.key")" || { rm -rf "${work}"; fail "could not store private key in Keychain"; }
    kc_put cosign-pass "${password}" || { rm -rf "${work}"; fail "could not store key password in Keychain"; }
    mv "${work}/cosign.pub" "${PUBKEY}"
    rm -rf "${work}"
    echo "release key generated: private key in Keychain (${KC_SERVICE}), public key at cosign.pub"
    echo "publish cosign.pub with every release; it is the trust root"
}

# Materialize the private key into a 0600 temp file for cosign; caller cleans up.
KEY_FILE=""
with_key() {
    local pem password
    if [[ "${BADAPPLE_NO_KEYCHAIN:-0}" == "1" ]]; then
        pem=$(cat "${REPO_ROOT}/cosign.key")
        password="${COSIGN_PASSWORD:?COSIGN_PASSWORD required with BADAPPLE_NO_KEYCHAIN}"
    else
        pem=$(kc_get_b64 cosign-key) || fail "no release key in Keychain; run --init"
        password=$(kc_get cosign-pass) || fail "release key password missing from Keychain; run --init"
    fi
    KEY_FILE=$(mktemp)
    chmod 600 "${KEY_FILE}"
    printf '%s' "${pem}" > "${KEY_FILE}"
    COSIGN_PASSWORD="${password}"
    export COSIGN_PASSWORD
}

sign_file() {
    local f="$1"
    [[ -f "${f}" ]] || fail "artifact not found: ${f}"
    cosign sign-blob --yes --key "${KEY_FILE}" \
        --bundle "${f}.sigstore.json" "${f}" >/dev/null
    echo "signed: $(basename "${f}") -> $(basename "${f}").sigstore.json"
}

upload_bundles() {
    local tag="$1"; shift
    local assets=()
    local f
    for f in "$@"; do
        [[ -f "${f}.sigstore.json" ]] && assets+=("${f}.sigstore.json")
    done
    assets+=("${PUBKEY}")
    gh release upload "${tag}" "${assets[@]}" --repo "${RELEASES_REPO}" --clobber
    echo "uploaded ${#assets[@]} assets to ${RELEASES_REPO} ${tag}"
}

retro_sign() {
    command -v gh >/dev/null || fail "gh is required for --retro"
    local tags tag work f name
    tags=$(gh release list --repo "${RELEASES_REPO}" --limit 100 --json tagName --jq '.[].tagName')
    for tag in ${tags}; do
        work=$(mktemp -d)
        gh release download "${tag}" --repo "${RELEASES_REPO}" --dir "${work}" --clobber 2>/dev/null || true
        for f in "${work}"/*.zip "${work}"/*.tar.gz "${work}"/checksums.txt; do
            [[ -f "${f}" ]] || continue
            name=$(basename "${f}")
            # Skip assets that are already signatures or the pubkey itself.
            case "${name}" in *.sigstore.json|cosign.pub) continue ;; esac
            # If a checksums.txt exists for this release, pin the artifact to
            # it before signing so we attest the published bytes, not whatever
            # the download produced.
            if [[ "${name}" != "checksums.txt" && -f "${work}/checksums.txt" ]]; then
                local expected actual
                expected=$(awk -v n="${name}" 'NF==2 { file=$2; sub(/^\*/,"",file); if (file==n) print tolower($1) }' "${work}/checksums.txt")
                if [[ "${expected}" =~ ^[0-9a-f]{64}$ ]]; then
                    actual=$(shasum -a 256 "${f}" | awk '{print $1}')
                    [[ "${actual}" == "${expected}" ]] || { echo "SKIP ${tag}/${name}: checksum mismatch"; continue; }
                fi
            fi
            sign_file "${f}"
        done
        upload_bundles "${tag}" "${work}"/* 2>/dev/null || echo "nothing to upload for ${tag}"
        rm -rf "${work}"
    done
}

main() {
    case "${1:-}" in
        --init) init_key; return ;;
        --retro) with_key; trap 'rm -f "${KEY_FILE}"' EXIT; retro_sign; return ;;
    esac

    local tag="" files=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tag) tag="$2"; shift 2 ;;
            *) files+=("$1"); shift ;;
        esac
    done

    [[ -f "${PUBKEY}" ]] || fail "no cosign.pub; run --init first"
    if [[ ${#files[@]} -eq 0 ]]; then
        for f in "${DIST_DIR}"/*.zip "${DIST_DIR}"/checksums.txt; do
            [[ -f "${f}" ]] && files+=("${f}")
        done
    fi
    [[ ${#files[@]} -gt 0 ]] || fail "no artifacts to sign (run a package_*.sh first, or pass files)"

    with_key
    trap 'rm -f "${KEY_FILE}"' EXIT
    for f in "${files[@]}"; do sign_file "${f}"; done
    [[ -n "${tag}" ]] && upload_bundles "${tag}" "${files[@]}"
}

main "$@"
