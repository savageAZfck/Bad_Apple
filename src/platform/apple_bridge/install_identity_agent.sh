#!/usr/bin/env bash
set -euo pipefail

# Install the Bad Apple identity agent as a user LaunchAgent so it runs in the
# Aqua session and can use the Secure Enclave on behalf of the MLX daemon.

SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
# Resolve the script's real path so this works regardless of $0 or cwd.
SCRIPT_PATH="$(/usr/bin/python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${SCRIPT_SOURCE}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/../../.." && pwd)"
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

/usr/bin/python3 - "${PLIST_SRC}" "${PLIST_DST}" "${REPO_ROOT}" "${CONSOLE_USER}" "${CONSOLE_HOME}" "${CONSOLE_GROUP}" <<'PY'
import sys
src, dst, root, user, home, group = sys.argv[1:7]
text = open(src).read()
text = text.replace("__BADAPPLE_ROOT__", root)
text = text.replace("__CONSOLE_USER__", user)
text = text.replace("__CONSOLE_HOME__", home)
text = text.replace("__CONSOLE_GROUP__", group)
open(dst, "w").write(text)
PY

plutil -lint "${PLIST_DST}"

launchctl unload "${PLIST_DST}" 2>/dev/null || true
launchctl load -w "${PLIST_DST}"

echo "Bad Apple identity agent is installed as a LaunchAgent."
echo "LaunchAgent: ${PLIST_DST}"
