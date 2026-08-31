#!/usr/bin/env bash
set -euo pipefail

# Bad Apple — 5-Minute Guided Video Demo Tour
#
# Records a 5-minute screen capture showing Bad Apple's full user-facing
# experience: splash screen, menu bar, web dashboard, CLI, personas, and
# tool use. No trade secrets are shown.
#
# Output: ~/Desktop/badapple_demo_tour.mov

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$HOME/Desktop/badapple_demo_tour.mov"
DURATION=300
CLI="${REPO_ROOT}/target/release/badapple"
export BADAPPLE_SOCKET_PATH="/var/run/badapple/substrate.sock"
export BADAPPLE_SLICKS_KEY_PATH="/var/lib/bad_apple/slicks.key"

if [[ ! -x "${CLI}" ]]; then
    echo "ERROR: badapple CLI not found at ${CLI}" >&2
    exit 1
fi

echo "=== Bad Apple 5-Minute Guided Demo Tour ==="
echo "Output: ${OUTPUT}"
echo "Duration: ${DURATION} seconds (5 minutes)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — write AppleScript to temp files to avoid quoting hell
# ─────────────────────────────────────────────────────────────────────────────
TMPDIR_DEMO="$(mktemp -d /tmp/badapple_demo_XXXXXX)"

# Show a title card in Terminal
show_title_card() {
    local title="$1"
    local subtitle="$2"
    cat > "${TMPDIR_DEMO}/title.scpt" <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "clear && echo '' && echo '' && echo '' && echo '' && echo '' && echo '    ======================================' && echo '' && echo '      ${title}' && echo '' && echo '      ${subtitle}' && echo '' && echo '    ======================================' && echo '' && echo '' && sleep 4 && clear"
end tell
APPLESCRIPT
    osascript "${TMPDIR_DEMO}/title.scpt" 2>/dev/null
}

# Run a CLI query in Terminal
run_query() {
    local label="$1"
    local prompt="$2"
    local tokens="${3:-120}"
    cat > "${TMPDIR_DEMO}/query.scpt" <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "echo '  > ${label}' && echo '  --------------------------------------' && '${CLI}' --max-tokens ${tokens} '${prompt}' 2>/dev/null && echo '' && echo ''"
end tell
APPLESCRIPT
    osascript "${TMPDIR_DEMO}/query.scpt" 2>/dev/null
}

# Run a raw command in Terminal
run_in_terminal() {
    local cmd="$1"
    cat > "${TMPDIR_DEMO}/raw.scpt" <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "${cmd}"
end tell
APPLESCRIPT
    osascript "${TMPDIR_DEMO}/raw.scpt" 2>/dev/null
}

# Open the menu bar dropdown
open_menubar() {
    osascript -e 'tell application "System Events" to tell process "BadApple" to click menu bar item 1 of menu bar 1' 2>/dev/null
}

# Close the menu bar dropdown
close_menubar() {
    osascript -e 'tell application "System Events" to key code 53' 2>/dev/null
}

# Open a URL
open_url() {
    open "${1}" 2>/dev/null
}

# Cleanup
cleanup() {
    rm -rf "${TMPDIR_DEMO}" 2>/dev/null
}
trap cleanup EXIT

# ═════════════════════════════════════════════════════════════════════════════
# START RECORDING — 5 minutes
# ═════════════════════════════════════════════════════════════════════════════
echo "Starting screen recording (${DURATION}s)..."
screencapture -V${DURATION} "${OUTPUT}" &
RECORD_PID=$!
sleep 3

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Splash Screen (0:03 - 0:30) — 27 seconds
# ═════════════════════════════════════════════════════════════════════════════
echo "[1/8] Splash screen..."

# Reset splash preference and restart to trigger the splash.
# Use the consistent bundle name "Bad Apple" for both quit and open.
defaults delete com.badapple.app BadAppleSplashBootTime 2>/dev/null || true
osascript -e 'tell application "Bad Apple" to quit' 2>/dev/null || true
sleep 3
# Verify the process is gone before relaunching.
pgrep -x "BadApple" >/dev/null 2>&1 && sleep 2
open -a "Bad Apple" 2>/dev/null || open "${REPO_ROOT}/target/release/Bad Apple.app" 2>/dev/null

# Splash shows on launch — let it sit for the viewer
sleep 22

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Menu Bar (0:30 - 1:15) — 45 seconds
# ═════════════════════════════════════════════════════════════════════════════
echo "[2/8] Menu bar overview..."
show_title_card "Bad Apple Menu Bar" "System-level AI - always one click away"
sleep 5

# Open the menu bar dropdown so the viewer can see the full menu
open_menubar
sleep 15

# Close it
close_menubar
sleep 3

# Open it again and navigate to Tools submenu
osascript -e 'tell application "System Events" to tell process "BadApple" to click menu bar item 1 of menu bar 1' 2>/dev/null
sleep 1
osascript -e 'tell application "System Events" to tell process "BadApple" to click menu item "Tools" of menu 1 of menu bar item 1 of menu bar 1' 2>/dev/null
sleep 10

close_menubar
sleep 3

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Web Dashboard — Chat (1:15 - 2:15) — 60 seconds
# ═════════════════════════════════════════════════════════════════════════════
echo "[3/8] Web dashboard — chat..."
show_title_card "Web Dashboard" "Local-only UI at 127.0.0.1:8787"
sleep 5

# Open the dashboard main page
open_url "http://127.0.0.1:8787"
sleep 8

# Open the chat page
open_url "http://127.0.0.1:8787/chat"
sleep 8

# Run queries in Terminal for visible streaming output
run_query "Who are you?" "Who are you and what makes you different from Siri or ChatGPT?" 150
sleep 20

run_query "Write a haiku" "Write a haiku about on-device AI and privacy" 60
sleep 12

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Dashboard — Models & Control (2:15 - 3:00) — 45 seconds
# ═════════════════════════════════════════════════════════════════════════════
echo "[4/8] Dashboard — models & control..."
open_url "http://127.0.0.1:8787/models"
sleep 12

open_url "http://127.0.0.1:8787/control"
sleep 10

open_url "http://127.0.0.1:8787/ambient"
sleep 10

# Go back to the main dashboard
open_url "http://127.0.0.1:8787/dashboard"
sleep 8

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: CLI Power User (3:00 - 4:00) — 60 seconds
# ═════════════════════════════════════════════════════════════════════════════
echo "[5/8] CLI demo..."
show_title_card "Rust CLI" "Fast - scripted - pipeable - from the terminal"
sleep 5

run_query "Quick math" "What is 17 times 23?" 32
sleep 6

run_query "Explain sockets" "Explain what a Unix domain socket is in one sentence" 80
sleep 10

# Show model list
run_in_terminal "echo '  > Model Registry' && echo '  --------------------------------------' && '${CLI}' model list 2>/dev/null | head -30 && echo '' && echo ''"
sleep 10

# Show doctor
run_in_terminal "echo '  > System Doctor' && echo '  --------------------------------------' && '${CLI}' --doctor 2>/dev/null | head -25 && echo '' && echo ''"
sleep 12

# Show benchmark
run_in_terminal "echo '  > Benchmark' && echo '  --------------------------------------' && '${CLI}' --benchmark 2>/dev/null | tail -15 && echo '' && echo ''"
sleep 15

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Personas (4:00 - 4:30) — 30 seconds
# ═════════════════════════════════════════════════════════════════════════════
echo "[6/8] Personas..."
show_title_card "Persona System" "Bad Apple has personality - 5 built-in personas"
sleep 5

# Default persona
run_query "Default: Cloud AI?" "What do you think of cloud AI?" 80
sleep 10

# Roast persona
run_in_terminal "echo '  > Switching to Drill (roast) persona...' && echo '' && '${CLI}' --roast --max-tokens 80 'What do you think of cloud AI?' 2>/dev/null && echo '' && echo ''"
sleep 15

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Tool Use (4:30 - 4:50) — 20 seconds
# ═════════════════════════════════════════════════════════════════════════════
echo "[7/8] Tool use..."
show_title_card "Tool Use" "Read files - run commands - interact with macOS"
sleep 5

run_query "List desktop files" "List the files on my Desktop and briefly describe what you see" 100
sleep 15

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: Closing (4:50 - 5:00) — 10 seconds
# ═════════════════════════════════════════════════════════════════════════════
echo "[8/8] Closing..."
show_title_card "Bad Apple" "100% on-device - 0% cloud - 0% telemetry"
sleep 10

# ═════════════════════════════════════════════════════════════════════════════
# WAIT FOR RECORDING TO FINISH
# ═════════════════════════════════════════════════════════════════════════════
echo "Waiting for recording to finalize..."
wait $RECORD_PID 2>/dev/null
RECORD_STATUS=$?
if [[ ${RECORD_STATUS} -ne 0 ]]; then
    echo "WARNING: screencapture exited with status ${RECORD_STATUS}" >&2
fi

# Verify output
if [[ -f "${OUTPUT}" ]]; then
    SIZE=$(stat -f%z "${OUTPUT}" 2>/dev/null || echo "0")
    echo ""
    echo "=== Demo tour recorded successfully ==="
    echo "File: ${OUTPUT}"
    echo "Size: $((SIZE / 1024 / 1024)) MB"
    echo ""
    echo "Ready to upload to LinkedIn."
else
    echo ""
    echo "ERROR: Output file not found at ${OUTPUT}" >&2
    echo ""
    echo "Troubleshooting:"
    echo "  1. System Settings > Privacy & Security > Screen Recording"
    echo "     Add Terminal to the allowed list and restart Terminal."
    echo "  2. Make sure Bad Apple menu bar app is running."
    echo "  3. Make sure the daemon is running: ls /var/run/badapple/substrate.sock"
fi
