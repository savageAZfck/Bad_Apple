#!/usr/bin/env bash
set -euo pipefail

# Install the Bad Apple daemon, authenticated CLI, bridge, and launchd job.
# Must be run as root (e.g., sudo ./install_daemon.sh).

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLIST_SOURCE="${REPO_ROOT}/src/platform/apple_bridge/com.badapple.substrate.plist"
PLIST_TARGET="/Library/LaunchDaemons/com.badapple.substrate.plist"
INSTALL_DIR="/usr/local/libexec/badapple"
CLI_TARGET="/usr/local/bin/badapple"
AUTOMATION_TARGET="/usr/local/bin/badapple-automation"
DATA_DIR="/var/lib/bad_apple"
SOCKET_DIR="/var/run/badapple"
KEY_FILE="${DATA_DIR}/slicks.key"
LOG_FILE="/var/log/bad_apple_daemon.log"
DAEMON_SOURCE="${REPO_ROOT}/target/release/badappled"
CLI_SOURCE="${REPO_ROOT}/target/release/badapple"
AUTOMATION_SOURCE="${REPO_ROOT}/target/release/badapple_automation"
BRIDGE_SOURCE="${REPO_ROOT}/target/release/libBadAppleBridge.dylib"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: This script must be run as root to install a system LaunchDaemon." >&2
    echo "       Re-run with: sudo $0" >&2
    exit 1
fi

for artifact in "${DAEMON_SOURCE}" "${CLI_SOURCE}" "${AUTOMATION_SOURCE}" "${BRIDGE_SOURCE}"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "ERROR: Required release artifact not found: ${artifact}" >&2
        echo "       Build with cargo build --release and src/platform/apple_bridge/build_apple_bridge.sh" >&2
        exit 1
    fi
done

DEFAULT_MODEL="${REPO_ROOT}/tests/ane_brain_perf/artifacts/qwen3b_ane_shards/conversion_manifest.json"
DEFAULT_TOKENIZER="${REPO_ROOT}/tests/ane_brain_perf/artifacts/qwen3b_ane_shards/tokenizer.json"
MODEL_PATH="${BADAPPLE_ANE_MODEL:-${DEFAULT_MODEL}}"
TOKENIZER_PATH="${BADAPPLE_ANE_TOKENIZER:-${DEFAULT_TOKENIZER}}"

case "${MODEL_PATH}" in
    /*) ;;
    *) echo "ERROR: ANE model path must be absolute: ${MODEL_PATH}" >&2; exit 1 ;;
esac
case "${TOKENIZER_PATH}" in
    /*) ;;
    *) echo "ERROR: Tokenizer path must be absolute: ${TOKENIZER_PATH}" >&2; exit 1 ;;
esac

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: ANE model artifact not found: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${TOKENIZER_PATH}" ]]; then
    echo "ERROR: Tokenizer artifact not found: ${TOKENIZER_PATH}" >&2
    exit 1
fi

install -d -o root -g wheel -m 755 "${INSTALL_DIR}"
install -d -o root -g wheel -m 755 "$(dirname "${CLI_TARGET}")"
install -d -o root -g staff -m 750 "${DATA_DIR}"
install -d -o root -g staff -m 770 "${SOCKET_DIR}"
chmod 770 "${SOCKET_DIR}"
chown root:staff "${SOCKET_DIR}"
for subdir in skills tools; do
    install -d -o root -g staff -m 700 "${DATA_DIR}/${subdir}"
done
install -d -o root -g staff -m 750 "${DATA_DIR}/strategy_db"
install -d -o root -g staff -m 770 "${DATA_DIR}/wild_workspace"
install -d -o root -g staff -m 770 "${DATA_DIR}/curriculum"

install -o root -g wheel -m 755 "${DAEMON_SOURCE}" "${INSTALL_DIR}/badappled"
install -o root -g wheel -m 755 "${CLI_SOURCE}" "${CLI_TARGET}"
install -o root -g wheel -m 755 "${AUTOMATION_SOURCE}" "${AUTOMATION_TARGET}"
install -o root -g wheel -m 755 "${BRIDGE_SOURCE}" "${INSTALL_DIR}/libBadAppleBridge.dylib"

if [[ ! -f "${KEY_FILE}" ]]; then
    install -d -o root -g staff -m 750 "$(dirname "${KEY_FILE}")"
    openssl rand -hex 32 > "${KEY_FILE}"
fi
chown root:staff "${KEY_FILE}"
chmod 640 "${KEY_FILE}"

touch "${LOG_FILE}"
chown root:staff "${LOG_FILE}"
chmod 640 "${LOG_FILE}"

cp "${PLIST_SOURCE}" "${PLIST_TARGET}"
# Resolve placeholder tokens so the plist points at the right checkout.
sed -i '' -e "s|__REPO_ROOT__|${REPO_ROOT}|g" -e "s|__HOME__|${HOME}|g" "${PLIST_TARGET}"
# Update the ANE model and tokenizer paths without Python.
plutil -replace EnvironmentVariables.BADAPPLE_ANE_MODEL -string "${MODEL_PATH}" "${PLIST_TARGET}"
plutil -replace EnvironmentVariables.BADAPPLE_ANE_TOKENIZER -string "${TOKENIZER_PATH}" "${PLIST_TARGET}"
chown root:wheel "${PLIST_TARGET}"
chmod 644 "${PLIST_TARGET}"
plutil -lint "${PLIST_TARGET}"

launchctl unload "${PLIST_TARGET}" 2>/dev/null || true
launchctl load -w "${PLIST_TARGET}"

echo "Bad Apple daemon installed and loaded."
echo "Label:     com.badapple.substrate"
echo "Daemon:    ${INSTALL_DIR}/badappled"
echo "CLI:       ${CLI_TARGET}"
echo "Automation:${AUTOMATION_TARGET}"
echo "Model:     ${MODEL_PATH}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Socket:    ${SOCKET_DIR}/substrate.sock"
echo "Data:      ${DATA_DIR}"
echo "Log:       ${LOG_FILE}"
echo ""
echo "Check status with: sudo launchctl print system/com.badapple.substrate"
echo "Run a query with:  badapple \"your question\""
echo "Unload with:       sudo launchctl bootout system ${PLIST_TARGET}"
