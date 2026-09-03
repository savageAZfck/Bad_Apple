#!/usr/bin/env bash
set -euo pipefail

# Install the Bad Apple identity agent as a user LaunchAgent so it runs in the
# Aqua session and can use the Secure Enclave on behalf of the MLX daemon.

SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
# Resolve the script's real path so this works regardless of $0 or cwd.
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PLIST_SRC="${REPO_ROOT}/src/platform/apple_bridge/com.badapple.identity_agent.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.badapple.identity_agent.plist"

if [[ ! -f "${PLIST_SRC}" ]]; then
    echo "Identity agent plist not found: ${PLIST_SRC}" >&2
    exit 1
fi

mkdir -p "${HOME}/Library/LaunchAgents"

CONSOLE_USER="${USER}"
CONSOLE_HOME="${HOME}"
CONSOLE_GROUP="$(id -gn)"

# Render the plist by replacing placeholder tokens. Using '|' as the sed
# delimiter avoids escaping path characters.
sed -e "s|__BADAPPLE_ROOT__|${REPO_ROOT}|g" \
    -e "s|__CONSOLE_USER__|${CONSOLE_USER}|g" \
    -e "s|__CONSOLE_HOME__|${CONSOLE_HOME}|g" \
    -e "s|__CONSOLE_GROUP__|${CONSOLE_GROUP}|g" \
    "${PLIST_SRC}" > "${PLIST_DST}"

plutil -lint "${PLIST_DST}"

launchctl unload "${PLIST_DST}" 2>/dev/null || true
launchctl load -w "${PLIST_DST}"

echo "Bad Apple identity agent is installed as a LaunchAgent."
echo "LaunchAgent: ${PLIST_DST}"
