#!/usr/bin/env bash
set -euo pipefail

# Friendly one-click installer for the Bad Apple platform.
# This is a thin wrapper around the bridge installer that uses AppleScript
# so the user only has to enter their Mac password.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

escape_applescript() {
    # Escape double quotes and backslashes for embedding in an AppleScript string.
    sed 's/\\/\\\\/g; s/"/\\"/g'
}

display_message() {
    local title="$1"
    local body="$2"
    local icon="${3:-note}"
    local escaped_title
    local escaped_body
    escaped_title=$(printf '%s' "$title" | escape_applescript)
    escaped_body=$(printf '%s' "$body" | escape_applescript)
    osascript -e "display dialog \"$escaped_body\" with title \"$escaped_title\" buttons {\"OK\"} default button \"OK\" with icon $icon" &>/dev/null || true
}

find_repo_root() {
    # 1. Explicit override.
    if [[ -n "${BADAPPLE_ROOT:-}" && -f "${BADAPPLE_ROOT}/src/platform/apple_bridge/install_badapple_platform.sh" ]]; then
        printf '%s' "$BADAPPLE_ROOT"
        return 0
    fi

    # 2. Normal source layout: this script is at src/platform/apple_desktop.
    local candidate
    candidate="$(cd "${SCRIPT_DIR}/../../.." && pwd 2>/dev/null)" || candidate=""
    if [[ -f "${candidate}/src/platform/apple_bridge/install_badapple_platform.sh" ]]; then
        printf '%s' "$candidate"
        return 0
    fi

    # 3. Packaged or installed layout: installer is next to the Bad Apple.app bundle.
    candidate="$(cd "${SCRIPT_DIR}/.." && pwd 2>/dev/null)" || candidate=""
    if [[ -f "${candidate}/src/platform/apple_bridge/install_badapple_platform.sh" ]]; then
        printf '%s' "$candidate"
        return 0
    fi

    # 4. Installed version directories under ~/.bad_apple.
    local home_dir
    home_dir="${HOME:-$(eval echo ~"$(whoami)")}"
    if [[ -d "${home_dir}/.bad_apple" ]]; then
        for version_dir in "${home_dir}/.bad_apple"/bad_apple-*; do
            if [[ -d "$version_dir/bad_apple" && -f "$version_dir/bad_apple/src/platform/apple_bridge/install_badapple_platform.sh" ]]; then
                printf '%s' "$version_dir/bad_apple"
                return 0
            fi
        done
    fi

    return 1
}

REPO_ROOT="${BADAPPLE_ROOT:-}"
if [[ -z "$REPO_ROOT" ]]; then
    if ! REPO_ROOT=$(find_repo_root); then
        display_message "Bad Apple source folder not found" \
"This installer needs the Bad Apple source folder.\n\nSet BADAPPLE_ROOT to the folder you downloaded, or run this installer from inside the Bad Apple source folder." "stop"
        exit 1
    fi
fi

if [[ ! -d "/Applications/Bad Apple.app" ]]; then
    display_message "Move Bad Apple to Applications" \
"Please drag Bad Apple.app into your Applications folder first, then run this installer again." "stop"
    exit 1
fi

INSTALLER="${REPO_ROOT}/src/platform/apple_bridge/install_badapple_platform.sh"
if [[ ! -f "$INSTALLER" ]]; then
    display_message "Bad Apple source folder is missing files" \
"The installer could not find the platform setup script.\n\nMake sure BADAPPLE_ROOT points to the full Bad Apple source folder." "stop"
    exit 1
fi

REPO_ESCAPED=$(printf '%s' "$REPO_ROOT" | escape_applescript)
INSTALLER_ESCAPED=$(printf '%s' "$INSTALLER" | escape_applescript)

# Let the user know what is about to happen.
osascript -e "display dialog \"This will install a small background helper so Bad Apple can run on your Mac. Your password is only used to allow this one setup step.\" with title \"Installing Bad Apple...\" buttons {\"Continue\"} default button \"Continue\" with icon note" &>/dev/null || true

# Run the bridge installer with administrator privileges.
set +e
OUTPUT=$(osascript <<EOF 2>&1
set repo to quoted form of POSIX path of "${REPO_ESCAPED}"
set installer to quoted form of POSIX path of "${INSTALLER_ESCAPED}"
do shell script "cd " & repo & " && " & installer & " --install --unsigned-install" with administrator privileges
EOF
)
EXIT_CODE=$?
set -e

if [[ $EXIT_CODE -eq 0 ]]; then
    display_message "Bad Apple is ready" \
"Installation finished. Bad Apple is now set up and the menu bar icon will appear in a few moments." "note"
else
    display_message "Installation could not finish" \
"Something went wrong while installing Bad Apple.\n\n${OUTPUT}\n\nNext step: make sure Bad Apple.app is in /Applications and that this installer is run from the Bad Apple source folder. If the problem continues, open the Bad Apple menu bar and choose Status to see the latest issue." "stop"
    exit "$EXIT_CODE"
fi
