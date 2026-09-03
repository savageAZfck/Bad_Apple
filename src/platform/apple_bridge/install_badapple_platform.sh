#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${BADAPPLE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

CONSOLE_USER="${CONSOLE_USER:-$(stat -f %Su /dev/console)}"
CONSOLE_UID="$(id -u "${CONSOLE_USER}")"
CONSOLE_HOME="$(dscl . -read "/Users/${CONSOLE_USER}" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
CONSOLE_HOME="${CONSOLE_HOME:-/Users/${CONSOLE_USER}}"
CONSOLE_GROUP="$(id -gn "${CONSOLE_USER}")"

MODE="--dry-run"
UNSIGNED=0
mode_set=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--install)
      MODE="$1"
      mode_set=1
      shift
      ;;
    --unsigned-install)
      UNSIGNED=1
      shift
      ;;
    *)
      fail "usage: $0 [--dry-run|--install] [--unsigned-install]"
      ;;
  esac
done
if [[ "${UNSIGNED}" -eq 1 && "${mode_set}" -eq 0 ]]; then
  MODE="--install"
fi
BACKUP_ROOT="/var/lib/bad_apple/install_backups"
RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${BACKUP_ROOT}/${RELEASE_ID}"

[[ "${MODE}" == "--dry-run" || "${MODE}" == "--install" ]] || fail "usage: $0 [--dry-run|--install] [--unsigned-install]"
[[ "$(uname -s)" == "Darwin" ]] || fail "Bad Apple platform installation requires macOS"
[[ -x "${REPO_ROOT}/target/release/badapple" ]] || fail "release CLI is missing"
[[ -x "${REPO_ROOT}/target/release/gatekeeper" ]] || fail "release gatekeeper is missing"
[[ -x "${REPO_ROOT}/target/release/badapple-identity" ]] || fail "Secure Enclave helper is missing"
[[ -x "${REPO_ROOT}/target/release/badapple-identity-agent" ]] || fail "Secure Enclave identity agent is missing"
[[ -x "${REPO_ROOT}/target/release/badapple-supervisor" ]] || fail "release supervisor is missing"
[[ -x "${REPO_ROOT}/target/release/badapple-engine" ]] || fail "native engine daemon is missing"
[[ -f "${REPO_ROOT}/target/release/libBadAppleMLX.dylib" ]] || fail "MLX runtime dylib is missing"
[[ -f "${REPO_ROOT}/target/release/mlx.metallib" ]] || fail "MLX metallib is missing"
[[ -x "/Applications/Bad Apple.app/Contents/MacOS/BadApple" ]] || fail "menu bar app is not installed"

render_plist() {
    local src="$1" dst="$2"
    sed -e "s|__BADAPPLE_ROOT__|${REPO_ROOT}|g" \
        -e "s|__CONSOLE_USER__|${CONSOLE_USER}|g" \
        -e "s|__CONSOLE_HOME__|${CONSOLE_HOME}|g" \
        -e "s|__CONSOLE_GROUP__|${CONSOLE_GROUP}|g" \
        "${src}" > "${dst}"
}

for plist in com.badapple.gatekeeper.plist com.badapple.mlx.plist com.badapple.supervisor.plist; do
    rendered="/tmp/${plist}.rendered.$$"
    render_plist "${REPO_ROOT}/src/platform/apple_bridge/${plist}" "${rendered}"
    plutil -lint "${rendered}" >/dev/null
    echo "verified plist: ${plist}"
    rm -f "${rendered}"
done

if [[ "${UNSIGNED}" -eq 0 ]]; then
    codesign --verify --strict "${REPO_ROOT}/target/release/badapple-identity"
    codesign --verify --deep --strict "/Applications/Bad Apple.app"
fi
echo "Native-only installation (no Python venv required)."

if [[ "${MODE}" == "--dry-run" ]]; then
    echo "Dry run passed. Re-run with --install to stage, promote, and health-check services."
    exit 0
fi

[[ "$(id -u)" -eq 0 ]] || fail "--install must run as root"

if [[ "${UNSIGNED}" -eq 1 ]]; then
    if [[ -d "/Applications/Bad Apple.app" ]]; then
        echo "Removing Gatekeeper quarantine from unsigned Bad Apple.app..."
        xattr -dr com.apple.quarantine "/Applications/Bad Apple.app" 2>/dev/null || true
    fi
fi

install -d -o "${CONSOLE_USER}" -g "${CONSOLE_GROUP}" -m 770 /var/lib/bad_apple /var/run/badapple
chown -R "${CONSOLE_USER}":"${CONSOLE_GROUP}" /var/lib/bad_apple
[[ -f /var/lib/bad_apple/slicks.key ]] && chmod 600 /var/lib/bad_apple/slicks.key
install -d -o root -g wheel -m 750 "${BACKUP_DIR}"
touch /var/log/bad_apple_mlx_server.log
chown "${CONSOLE_USER}":"${CONSOLE_GROUP}" /var/log/bad_apple_mlx_server.log
chmod 644 /var/log/bad_apple_mlx_server.log
USER_DATA="${CONSOLE_HOME}/.bad_apple"
install -d -o "${CONSOLE_USER}" -g "${CONSOLE_GROUP}" -m 755 "${USER_DATA}"
chown -R "${CONSOLE_USER}":"${CONSOLE_GROUP}" "${USER_DATA}"
HF_CACHE="${CONSOLE_HOME}/.cache/huggingface"
if [[ -d "${HF_CACHE}" ]]; then
    chown -R "${CONSOLE_USER}":"${CONSOLE_GROUP}" "${HF_CACHE}"
fi

for plist in com.badapple.gatekeeper.plist com.badapple.mlx.plist com.badapple.supervisor.plist; do
    target="/Library/LaunchDaemons/${plist}"
    [[ ! -f "${target}" ]] || cp -p "${target}" "${BACKUP_DIR}/${plist}"
    incoming="${target}.incoming.$$"
    render_plist "${REPO_ROOT}/src/platform/apple_bridge/${plist}" "${incoming}"
    plutil -lint "${incoming}" >/dev/null
    install -o root -g wheel -m 644 "${incoming}" "${target}"
    rm -f "${incoming}"
done

rollback() {
    for plist in com.badapple.gatekeeper.plist com.badapple.mlx.plist com.badapple.supervisor.plist; do
        backup="${BACKUP_DIR}/${plist}"
        target="/Library/LaunchDaemons/${plist}"
        [[ ! -f "${backup}" ]] || cp -p "${backup}" "${target}"
    done
    unload_job system/com.badapple.supervisor
    unload_job system/com.badapple.mlx
    unload_job system/com.badapple.gatekeeper
    sleep 1
    launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist || true
    launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist || true
    fail "readiness failed; previous launchd configuration restored from ${BACKUP_DIR}"
}
trap rollback ERR

# Fully unload any previously loaded jobs before re-loading. `bootout` is
# sometimes not enough when launchd has the job in an orphaned/enabled state;
# disable + remove ensures the next `load -w` succeeds.
unload_job() {
    local label="$1"
    launchctl disable "${label}" 2>/dev/null || true
    launchctl remove "${label}" 2>/dev/null || true
}

unload_job system/com.badapple.supervisor
unload_job system/com.badapple.mlx
unload_job system/com.badapple.gatekeeper
sleep 2

launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist
launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist
launchctl load -w /Library/LaunchDaemons/com.badapple.supervisor.plist

ready=0
for _ in {1..90}; do
    if [[ -S /var/run/badapple/substrate.sock && -S /var/run/badapple/substrate_mlx.sock ]]; then
        if "${REPO_ROOT}/target/release/badapple" -n 16 "Reply only: ready" >/dev/null 2>&1; then
            ready=1
            break
        fi
    fi
    sleep 5
done
[[ "${ready}" -eq 1 ]] || rollback
trap - ERR

launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" "${REPO_ROOT}/src/platform/apple_bridge/install_identity_agent.sh"

# Optional native TTS agent.
if [[ -x "${REPO_ROOT}/target/release/badapple-tts" ]]; then
    launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" "${REPO_ROOT}/src/platform/apple_desktop/install_tts_agent.sh"
fi

launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" "${REPO_ROOT}/src/platform/apple_desktop/install_menu_bar_agent.sh"

echo "Bad Apple platform ${RELEASE_ID} installed and verified."
echo "Rollback snapshot: ${BACKUP_DIR}"
