#!/usr/bin/env bash
set -euo pipefail

# Bad Apple self-demo recorder.
# Records a ~45 second video of Bad Apple working on autopilot.
# Saves to ~/Desktop/badapple_demo.mov
#
# No trade secrets are shown: no audit ledger, no SLICKS keys, no policy.yaml,
# no cert suite, no internal sockets. Just the user-facing chat experience.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$HOME/Desktop/badapple_demo.mov"
DURATION=45
CLI="${REPO_ROOT}/target/release/badapple"

if [[ ! -x "${CLI}" ]]; then
    echo "error: badapple CLI not found at ${CLI}" >&2
    exit 1
fi

echo "=== Bad Apple Demo Recorder ==="
echo "Output: ${OUTPUT}"
echo "Duration: ${DURATION} seconds"
echo ""

# Step 1: Open a Terminal window with large text and run the demo.
echo "[1/3] Opening Terminal with demo script..."

# Create the inner demo script that runs inside the visible Terminal
DEMO_SCRIPT="$(mktemp /tmp/badapple_demo_XXXXXX)"
trap 'rm -f "${DEMO_SCRIPT}"' EXIT
cat > "${DEMO_SCRIPT}" <<'INNER_EOF'
#!/usr/bin/env bash
# This runs inside the visible Terminal window.
# CLI path is passed as $1 by the outer script.

CLI="${1:?missing CLI path}"
export BADAPPLE_SOCKET_PATH="/var/run/badapple/substrate.sock"
export BADAPPLE_SLICKS_KEY_PATH="/var/lib/bad_apple/slicks.key"

clear
echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║          Bad Apple — On-Device AI OS Layer           ║"
echo "  ║          Air-gapped · Private · Local                ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""
sleep 2

echo "  ▸ Query 1: Who are you?"
echo "  ─────────────────────────────────────────────────────"
sleep 1
"${CLI}" --max-tokens 120 "Who are you and what makes you different from other AI assistants?" 2>/dev/null
echo ""
sleep 2

echo "  ▸ Query 2: Write a haiku about on-device AI"
echo "  ─────────────────────────────────────────────────────"
sleep 1
"${CLI}" --max-tokens 80 "Write a haiku about on-device AI and privacy" 2>/dev/null
echo ""
sleep 2

echo "  ▸ Query 3: What can you see on my desktop?"
echo "  ─────────────────────────────────────────────────────"
sleep 1
"${CLI}" --max-tokens 100 "List the files on my Desktop and briefly describe what you see" 2>/dev/null
echo ""
sleep 2

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   Bad Apple — 100% on-device · 0% cloud · 0% telemetry  ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""
sleep 3
INNER_EOF
chmod +x "${DEMO_SCRIPT}"

CLI_PATH="${REPO_ROOT}/target/release/badapple"

# Open Terminal and run the demo script with the CLI path as $1
osascript -e "
tell application \"Terminal\"
    activate
    -- Create a new window with large text
    do script \"clear && printf '\\e[8;40;120t' && osascript -e 'tell application \\\"Terminal\\\" to set font size of front window to 16' && '${DEMO_SCRIPT}' '${CLI_PATH}'\"
end tell
" 2>/dev/null

# Give Terminal time to open and start
sleep 3

# Step 2: Start screen recording.
echo "[2/3] Starting screen recording (${DURATION}s)..."
screencapture -V${DURATION} "${OUTPUT}" &
RECORD_PID=$!

# Step 3: Wait for recording to finish.
echo "[3/3] Recording... will stop automatically after ${DURATION}s"
wait $RECORD_PID 2>/dev/null || true

# Clean up the temp script
rm -f "${DEMO_SCRIPT}" 2>/dev/null

# Verify output
if [[ -f "${OUTPUT}" ]]; then
    SIZE=$(stat -f%z "${OUTPUT}" 2>/dev/null || echo "?")
    echo ""
    echo "=== Demo recorded successfully ==="
    echo "File: ${OUTPUT}"
    echo "Size: $((SIZE / 1024 / 1024)) MB"
    echo ""
    echo "Ready to upload to LinkedIn."
else
    echo "ERROR: Output file not found at ${OUTPUT}" >&2
    echo ""
    echo "NOTE: screencapture -V requires Screen Recording permission."
    echo "      System Settings > Privacy & Security > Screen Recording"
    echo "      Add Terminal (or the script's parent process) to the allowed list."
    exit 1
fi
