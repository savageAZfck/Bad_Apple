#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

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
CONSOLE_USER="$(stat -f %Su /dev/console)"
CONSOLE_UID="$(id -u "${CONSOLE_USER}")"
BACKUP_ROOT="/var/lib/bad_apple/install_backups"
RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${BACKUP_ROOT}/${RELEASE_ID}"

[[ "${MODE}" == "--dry-run" || "${MODE}" == "--install" ]] || fail "usage: $0 [--dry-run|--install] [--unsigned-install]"
[[ "$(uname -s)" == "Darwin" ]] || fail "Bad Apple platform installation requires macOS"
[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || fail "persistent Python environment is missing"
[[ -x "${REPO_ROOT}/target/release/badapple" ]] || fail "release CLI is missing"
[[ -x "${REPO_ROOT}/target/release/gatekeeper" ]] || fail "release gatekeeper is missing"
[[ -x "${REPO_ROOT}/target/release/badapple-identity" ]] || fail "Secure Enclave helper is missing"
[[ -x "/Applications/Bad Apple.app/Contents/MacOS/BadApple" ]] || fail "menu bar app is not installed"

for plist in com.badapple.gatekeeper.plist com.badapple.mlx.plist com.badapple.supervisor.plist; do
    plutil -lint "${REPO_ROOT}/src/platform/apple_bridge/${plist}" >/dev/null
    echo "verified plist: ${plist}"
done

if [[ "${UNSIGNED}" -eq 0 ]]; then
    codesign --verify --strict "${REPO_ROOT}/target/release/badapple-identity"
    codesign --verify --deep --strict "/Applications/Bad Apple.app"
fi
"${REPO_ROOT}/.venv/bin/python" -m pip check
"${REPO_ROOT}/.venv/bin/python" -m unittest tests.test_badapple_runtime

if [[ "${MODE}" == "--dry-run" ]]; then
    echo "Dry run passed. Re-run with --install to stage, promote, and health-check services."
    exit 0
fi

[[ "$(id -u)" -eq 0 ]] || fail "--install must run as root"
install -d -o "${CONSOLE_USER}" -g staff -m 770 /var/lib/bad_apple /var/run/badapple
chown -R "${CONSOLE_USER}":staff /var/lib/bad_apple
chmod 600 /var/lib/bad_apple/slicks.key
install -d -o root -g wheel -m 750 "${BACKUP_DIR}"
touch /var/log/bad_apple_mlx_server.log
chown "${CONSOLE_USER}":staff /var/log/bad_apple_mlx_server.log
chmod 644 /var/log/bad_apple_mlx_server.log
USER_DATA="$(eval echo ~"${CONSOLE_USER}")/.bad_apple"
install -d -o "${CONSOLE_USER}" -g staff -m 755 "${USER_DATA}"
chown -R "${CONSOLE_USER}":staff "${USER_DATA}"
HF_CACHE="$(eval echo ~"${CONSOLE_USER}")/.cache/huggingface"
if [[ -d "${HF_CACHE}" ]]; then
    chown -R "${CONSOLE_USER}":staff "${HF_CACHE}"
fi

for plist in com.badapple.gatekeeper.plist com.badapple.mlx.plist com.badapple.supervisor.plist; do
    target="/Library/LaunchDaemons/${plist}"
    [[ ! -f "${target}" ]] || cp -p "${target}" "${BACKUP_DIR}/${plist}"
    incoming="${target}.incoming.$$"
    install -o root -g wheel -m 644 "${REPO_ROOT}/src/platform/apple_bridge/${plist}" "${incoming}"
    plutil -lint "${incoming}" >/dev/null
    mv -f "${incoming}" "${target}"
done

rollback() {
    for plist in com.badapple.gatekeeper.plist com.badapple.mlx.plist com.badapple.supervisor.plist; do
        backup="${BACKUP_DIR}/${plist}"
        target="/Library/LaunchDaemons/${plist}"
        [[ ! -f "${backup}" ]] || cp -p "${backup}" "${target}"
    done
    launchctl bootout system/com.badapple.supervisor 2>/dev/null || true
    launchctl bootout system/com.badapple.mlx 2>/dev/null || true
    launchctl bootout system/com.badapple.gatekeeper 2>/dev/null || true
    launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist || true
    launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist || true
    fail "readiness failed; previous launchd configuration restored from ${BACKUP_DIR}"
}
trap rollback ERR

launchctl bootout system/com.badapple.supervisor 2>/dev/null || true
launchctl bootout system/com.badapple.mlx 2>/dev/null || true
launchctl bootout system/com.badapple.gatekeeper 2>/dev/null || true
sleep 2
launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist
launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist
launchctl load -w /Library/LaunchDaemons/com.badapple.supervisor.plist

ready=0
for _ in $(seq 1 90); do
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

launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" "${REPO_ROOT}/src/platform/apple_desktop/install_tts_agent.sh"
launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" "${REPO_ROOT}/src/platform/apple_desktop/install_menu_bar_agent.sh"

echo "Bad Apple platform ${RELEASE_ID} installed and verified."
echo "Rollback snapshot: ${BACKUP_DIR}"
