#!/usr/bin/env bash
set -euo pipefail

# Auto-update Bad Apple.app from a GitHub release, no Apple Developer ID required.
# The updater fetches the latest release, downloads the unsigned .zip, replaces
# the app in /Applications, strips the quarantine flag, and restarts the menu bar.
#
# Environment:
#   BADAPPLE_GH_REPO  - owner/repo on GitHub (default: savag3/bad_apple)
#   BADAPPLE_TAG      - specific tag to install, or "latest" (default: latest)
#
# The matching release asset must be named one of:
#   Bad_Apple-<version>-unsigned.zip
#   Bad_Apple-<tag>-unsigned.zip

REPO="${BADAPPLE_GH_REPO:-savag3/bad_apple}"
TAG="${BADAPPLE_TAG:-latest}"
APP="/Applications/Bad Apple.app"
BACKUP_DIR="${HOME}/.bad_apple/backups"

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || fail "curl is required"
command -v unzip >/dev/null || fail "unzip is required"

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

download_asset() {
    local tag="$1"
    local version
    version="${tag#v}"

    local tmpdir
    tmpdir=$(mktemp -d)
    local tried=()

    for name in "Bad_Apple-${version}-unsigned.zip" "Bad_Apple-${tag}-unsigned.zip" "Bad_Apple-${version}.zip" "Bad_Apple-${tag}.zip"; do
        local url="https://github.com/${REPO}/releases/download/${tag}/${name}"
        local out="${tmpdir}/${name}"
        tried+=("${url}")
        if curl -fsSL "${url}" -o "${out}"; then
            # Verify SHA-256 checksum if checksums.txt is published alongside
            # the release.  If checksums.txt is absent, warn but continue so
            # older releases without checksums remain installable.
            local checksums_url="https://github.com/${REPO}/releases/download/${tag}/checksums.txt"
            local checksums_file="${tmpdir}/checksums.txt"
            if curl -fsSL "${checksums_url}" -o "${checksums_file}" 2>/dev/null; then
                local basename expected_sha actual_sha
                basename="$(basename "${out}")"
                expected_sha="$(grep -E "^[0-9a-fA-F]{64}  ${basename}$" "${checksums_file}" | awk '{print $1}')"
                if [[ -n "${expected_sha}" ]]; then
                    actual_sha="$(shasum -a 256 "${out}" | awk '{print $1}')"
                    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
                        rm -rf "${tmpdir}"
                        fail "checksum mismatch for ${basename}: expected ${expected_sha}, got ${actual_sha}"
                    fi
                    echo "Checksum verified for ${basename}." >&2
                else
                    echo "Warning: ${basename} not found in checksums.txt; skipping verification." >&2
                fi
            else
                echo "Warning: checksums.txt not found for tag ${tag}; skipping checksum verification." >&2
            fi
            echo "${out}"
            return
        fi
    done

    rm -rf "${tmpdir}"
    fail "could not download a release asset. Tried:\n$(printf '  %s\n' "${tried[@]}")"
}

[[ "$(id -u)" -eq 0 ]] || fail "update must run as root so it can replace /Applications/Bad Apple.app"

CONSOLE_USER="$(stat -f %Su /dev/console 2>/dev/null || echo "${SUDO_USER:-${USER:-root}}")"
CONSOLE_UID="$(id -u "${CONSOLE_USER}" 2>/dev/null || echo 0)"
CONSOLE_HOME="$(dscl . -read "/Users/${CONSOLE_USER}" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
CONSOLE_HOME="${CONSOLE_HOME:-/Users/${CONSOLE_USER}}"
PLIST="${CONSOLE_HOME}/Library/LaunchAgents/com.badapple.menubar.plist"

current=$(installed_version)
echo "Installed version: ${current}"

if [[ "${TAG}" == "latest" ]]; then
    TAG=$(resolve_latest)
    [[ -n "${TAG}" ]] || fail "could not determine latest release tag"
fi

new_version="${TAG#v}"
echo "Latest release: ${TAG} (version ${new_version})"

if [[ "${current}" == "${new_version}" ]]; then
    echo "Bad Apple is already up to date (${current})."
    exit 0
fi

echo "Downloading update..."
zip=$(download_asset "${TAG}")

echo "Stopping Bad Apple..."
if [[ -f "${PLIST}" && "${CONSOLE_UID}" -ne 0 ]]; then
    launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" launchctl unload "${PLIST}" 2>/dev/null || true
fi
osascript -e 'tell application "Bad Apple" to quit' 2>/dev/null || true
sleep 1

install -d "${BACKUP_DIR}"
if [[ -d "${APP}" ]]; then
    backup="${BACKUP_DIR}/Bad Apple.app-$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Backing up current app to ${backup}..."
    cp -a "${APP}" "${backup}"
fi

echo "Extracting update..."
stage=$(mktemp -d)
trap 'rm -rf "${stage}" "${zip}"' EXIT
unzip -q "${zip}" -d "${stage}"

updated_app=$(find "${stage}" -maxdepth 2 -type d -name 'Bad Apple.app' | head -n1)
[[ -d "${updated_app}" ]] || fail "Bad Apple.app not found inside the downloaded zip"

rm -rf "${APP}"
cp -a "${updated_app}" "${APP}"

echo "Removing quarantine flag..."
LOCAL_STRIP="$(cd "$(dirname "$0")" && pwd)/strip_quarantine.sh"
if [[ -x "${LOCAL_STRIP}" ]]; then
    "${LOCAL_STRIP}" 2>/dev/null || xattr -dr com.apple.quarantine "${APP}" 2>/dev/null || true
else
    xattr -dr com.apple.quarantine "${APP}" 2>/dev/null || true
fi
if ! codesign --verify --deep --strict "${APP}" >/dev/null 2>&1; then
    codesign --force --deep --sign - "${APP}"
fi

# Try to update the full platform if a full release zip is available.
full_zip=""
for name in "Bad_Apple-${new_version}-full-unsigned.zip" "Bad_Apple-${TAG}-full-unsigned.zip"; do
    url="https://github.com/${REPO}/releases/download/${TAG}/${name}"
    tmp="${stage}/${name}"
    if curl -fsSL "${url}" -o "${tmp}"; then
        full_zip="${tmp}"
        break
    fi
done

if [[ -n "${full_zip}" ]]; then
    echo "Full platform update found; installing..."
    pkg_dir="${HOME}/.bad_apple/bad_apple-${new_version}"
    rm -rf "${pkg_dir}"
    install -d "${pkg_dir}"
    unzip -q "${full_zip}" -d "${pkg_dir}"
    # Find the package root inside the zip.
    pkg_root=$(find "${pkg_dir}" -maxdepth 1 -type d -name 'Bad_Apple-*-full' | head -n1)
    if [[ -x "${pkg_root}/install.sh" ]]; then
        "${pkg_root}/install.sh"
    else
        echo "Warning: full release package does not contain install.sh; app updated only."
    fi
else
    echo "No full platform update found; app updated only."
fi

echo "Removing quarantine flag..."
LOCAL_STRIP="$(cd "$(dirname "$0")" && pwd)/strip_quarantine.sh"
if [[ -x "${LOCAL_STRIP}" ]]; then
    "${LOCAL_STRIP}" 2>/dev/null || xattr -dr com.apple.quarantine "${APP}" 2>/dev/null || true
else
    xattr -dr com.apple.quarantine "${APP}" 2>/dev/null || true
fi

echo "Restarting Bad Apple..."
if [[ -f "${PLIST}" && "${CONSOLE_UID}" -ne 0 ]]; then
    launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" launchctl load -w "${PLIST}" 2>/dev/null || true
elif [[ "${CONSOLE_UID}" -ne 0 ]]; then
    launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" open -a "Bad Apple" 2>/dev/null || \
        sudo -u "${CONSOLE_USER}" open -a "Bad Apple" 2>/dev/null || true
else
    open -a "Bad Apple" 2>/dev/null || true
fi

echo "Updated Bad Apple to ${new_version}."
