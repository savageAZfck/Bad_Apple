#!/usr/bin/env bash
# Download + install the 4B ANE brain artifact.
#
# The sovereign Qwen3-4B ANE shards are a release asset — nobody else can
# produce them (mil_convert/mil_infer emission). This script fetches the
# zip, drops the shards into ~/.bad_apple/ane/, verifies the manifest, and
# writes the ~/.bad_apple/ane_brain sentinel so the engine admits the
# brain on the next query.
#
# Skipped entirely when downloads are disallowed (BADAPPLE_AIRGAP=1 or
# the caller passes --airgap).

set -euo pipefail

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '  %s\n' "$*"; }

URL="${BADAPPLE_ANE_ARTIFACT_URL:-https://github.com/savageAZfck/bad-apple-releases/releases/download/v0.9.6/Bad_Apple-ANE-4B.zip}"
HOME_DIR="${HOME:-${CONSOLE_HOME:-$HOME}}"
STATE_DIR="${HOME_DIR}/.bad_apple"
ANE_DIR="${STATE_DIR}/ane"
SENTINEL="${STATE_DIR}/ane_brain"
ZIP="${TMPDIR:-/tmp}/Bad_Apple-ANE-4B.zip"

case "${1:-}" in
    --airgap)
        info "air-gap install — skipping ANE brain download."
        exit 0
        ;;
esac

if [[ -f "${ANE_DIR}/conversion_manifest.json" ]]; then
    info "ANE brain artifact already installed."
    exit 0
fi

info "fetching 4B Neural Engine brain (~3.8 GB, 3 parts)..."
# GitHub caps release assets at 2 GB — the artifact ships as three parts.
for suffix in aa ab ac ad; do
    curl -fL --retry 3 -o "${ZIP}.${suffix}.tmp" "${URL}.${suffix}" || fail "ANE brain part .${suffix} failed: ${URL}.${suffix}"
done
cat "${ZIP}.aa.tmp" "${ZIP}.ab.tmp" "${ZIP}.ac.tmp" "${ZIP}.ad.tmp" > "${ZIP}"
rm -f "${ZIP}".{aa,ab,ac,ad}.tmp

info "installing into ${ANE_DIR}..."
mkdir -p "${ANE_DIR}"
unzip -q -o "${ZIP}" -d "${ANE_DIR}"
if [[ -d "${ANE_DIR}/qwen3b_v2" ]]; then
    mv "${ANE_DIR}/qwen3b_v2/"* "${ANE_DIR}/"
    rmdir "${ANE_DIR}/qwen3b_v2"
fi
rm -f "${ZIP}"

[[ -f "${ANE_DIR}/conversion_manifest.json" ]] || fail "manifest missing after extract"
[[ -f "${ANE_DIR}/tokenizer.json" ]] || fail "tokenizer missing after extract"
[[ -f "${ANE_DIR}/embedding.f16" ]] || fail "embedding missing after extract"

touch "${SENTINEL}"
chmod 600 "${SENTINEL}" 2>/dev/null || true
info "Neural Engine brain ready — the engine admits it on the next query."
