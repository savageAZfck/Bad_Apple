#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${BADAPPLE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

fail() {
    printf 'error: %s\n' "$*" >&2
    printf '       For troubleshooting, run: %s/target/release/badapple --doctor\n' "${REPO_ROOT}" >&2
    printf '       See also: SUPPORT.md in the Bad Apple source folder\n' >&2
    exit 1
}

CONSOLE_USER="${CONSOLE_USER:-${SUDO_USER:-$(stat -f %Su /dev/console)}}"
CONSOLE_UID="$(id -u "${CONSOLE_USER}")"
[[ "${CONSOLE_UID}" -ne 0 ]] || fail "installation requires a logged-in non-root CONSOLE_USER"
CONSOLE_HOME="$(dscl . -read "/Users/${CONSOLE_USER}" NFSHomeDirectory | sed 's/^NFSHomeDirectory: //')"
[[ -d "${CONSOLE_HOME}" ]] || fail "cannot resolve the console user's home"
run_user() { launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" env HOME="${CONSOLE_HOME}" USER="${CONSOLE_USER}" "$@"; }
CONSOLE_GROUP="$(id -gn "${CONSOLE_USER}")"

# Pick default KV-cache, prefill, and draft-token settings based on total unified
# memory. These are written into the LaunchDaemon plists so first-run works out
# of the box on 8 GB Macs without manual tuning.
RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
if [[ "$RAM_GB" -ge 16 ]]; then
    BADAPPLE_MAX_KV_SIZE=2048
    BADAPPLE_PREFILL_STEP_SIZE=2048
    BADAPPLE_NUM_DRAFT_TOKENS=2
elif [[ "$RAM_GB" -ge 8 ]]; then
    BADAPPLE_MAX_KV_SIZE=2048
    BADAPPLE_PREFILL_STEP_SIZE=2048
    BADAPPLE_NUM_DRAFT_TOKENS=2
else
    BADAPPLE_MAX_KV_SIZE=1024
    BADAPPLE_PREFILL_STEP_SIZE=1024
    BADAPPLE_NUM_DRAFT_TOKENS=0
fi

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
        -e "s|__BADAPPLE_MAX_KV_SIZE__|${BADAPPLE_MAX_KV_SIZE}|g" \
        -e "s|__BADAPPLE_PREFILL_STEP_SIZE__|${BADAPPLE_PREFILL_STEP_SIZE}|g" \
        -e "s|__BADAPPLE_NUM_DRAFT_TOKENS__|${BADAPPLE_NUM_DRAFT_TOKENS}|g" \
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
install -d -o root -g wheel -m 750 "${BACKUP_ROOT}"
BACKUP_DIR="$(mktemp -d "${BACKUP_ROOT}/${RELEASE_ID}.XXXXXX")"
chmod 750 "${BACKUP_DIR}"
launchctl print "gui/${CONSOLE_UID}" >/dev/null || fail "no GUI session for ${CONSOLE_USER}"
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

DAEMONS=(com.badapple.gatekeeper com.badapple.mlx com.badapple.supervisor)
AGENTS=(com.badapple.identity_agent com.badapple.tts com.badapple.menubar)
AGENT_DIR="${CONSOLE_HOME}/Library/LaunchAgents"
for label in "${DAEMONS[@]}"; do
    target="/Library/LaunchDaemons/${label}.plist"
    [[ ! -f "${target}" ]] || cp -p "${target}" "${BACKUP_DIR}/${label}.plist"
    if launchctl print "system/${label}" >/dev/null 2>&1; then
        touch "${BACKUP_DIR}/${label}.loaded"
    fi
done
for label in "${AGENTS[@]}"; do
    target="${AGENT_DIR}/${label}.plist"
    [[ ! -f "${target}" ]] || cp -p "${target}" "${BACKUP_DIR}/${label}.plist"
    if run_user launchctl list "${label}" >/dev/null 2>&1; then
        touch "${BACKUP_DIR}/${label}.loaded"
    fi
done

rollback() {
    local status=$?
    trap - EXIT
    [[ "${status}" -ne 0 ]] || return 0
    set +e
    unload_job system/com.badapple.supervisor
    unload_job system/com.badapple.mlx
    unload_job system/com.badapple.gatekeeper
    for label in "${AGENTS[@]}"; do
        run_user launchctl unload "${AGENT_DIR}/${label}.plist" 2>/dev/null
        launchctl bootout "gui/${CONSOLE_UID}/${label}" 2>/dev/null
        target="${AGENT_DIR}/${label}.plist"
        if [[ -f "${BACKUP_DIR}/${label}.plist" ]]; then
            cp -p "${BACKUP_DIR}/${label}.plist" "${target}"
        else
            rm -f "${target}"
        fi
        if [[ -f "${BACKUP_DIR}/${label}.loaded" ]]; then
            run_user launchctl load -w "${target}"
        fi
    done
    for label in "${DAEMONS[@]}"; do
        target="/Library/LaunchDaemons/${label}.plist"
        if [[ -f "${BACKUP_DIR}/${label}.plist" ]]; then
            cp -p "${BACKUP_DIR}/${label}.plist" "${target}"
        else
            rm -f "${target}"
        fi
        if [[ -f "${BACKUP_DIR}/${label}.loaded" ]]; then
            load_job "${label}"
        fi
    done
    echo "error: platform installation failed; rollback attempted from ${BACKUP_DIR}. Inspect launchctl/logs if any restore command failed." >&2
    exit "${status}"
}

# Fully unload any previously loaded jobs before re-loading. `bootout` is
# sometimes not enough when launchd has the job in an orphaned/enabled state;
# unload plus remove with a bare label avoids leaving the service disabled.
unload_job() {
    local service="$1" label="${1#system/}"
    launchctl unload "/Library/LaunchDaemons/${label}.plist" 2>/dev/null || true
    launchctl bootout "${service}" 2>/dev/null || true
    launchctl remove "${label}" 2>/dev/null || true
    for _ in {1..30}; do
        launchctl print "${service}" >/dev/null 2>&1 || return 0
        sleep 1
    done
    echo "error: could not unload ${service}" >&2
    return 1
}
load_job() {
    local label="$1" target="/Library/LaunchDaemons/$1.plist"
    launchctl enable "system/${label}"
    if ! launchctl load -w "${target}"; then
        launchctl print "system/${label}" >/dev/null 2>&1 || launchctl bootstrap system "${target}"
    fi
    launchctl print "system/${label}" >/dev/null
}

trap rollback EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
unload_job system/com.badapple.supervisor
unload_job system/com.badapple.mlx
unload_job system/com.badapple.gatekeeper
sleep 2

for label in "${DAEMONS[@]}"; do
    target="/Library/LaunchDaemons/${label}.plist"
    incoming="${BACKUP_DIR}/${label}.incoming"
    render_plist "${REPO_ROOT}/src/platform/apple_bridge/${label}.plist" "${incoming}"
    plutil -lint "${incoming}" >/dev/null
    install -o root -g wheel -m 644 "${incoming}" "${target}"
done

run_user "${REPO_ROOT}/src/platform/apple_bridge/install_identity_agent.sh"
for label in "${DAEMONS[@]}"; do
    load_job "${label}"
done

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
[[ "${ready}" -eq 1 ]] || fail "native platform readiness check failed"

# Optional native TTS agent.
if [[ -x "${REPO_ROOT}/target/release/badapple-tts" ]]; then
    run_user "${REPO_ROOT}/src/platform/apple_desktop/install_tts_agent.sh"
fi

run_user "${REPO_ROOT}/src/platform/apple_desktop/install_menu_bar_agent.sh"
trap - EXIT INT TERM

echo "Bad Apple platform ${RELEASE_ID} installed and verified."
echo "Rollback snapshot: ${BACKUP_DIR}"
