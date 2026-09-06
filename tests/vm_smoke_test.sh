#!/usr/bin/env bash
#
# Bad Apple clean-machine / VM smoke test
#
# Run this on a fresh macOS VM or a fresh test Mac to verify the full-unsigned
# release installs, starts, passes the air-gap cert suite, and can answer a
# simple query.
#
# Usage:
#   sudo -E ./tests/vm_smoke_test.sh [path/to/Bad_Apple-<version>-full-unsigned.zip]
#
# Environment:
#   BADAPPLE_ZIP            - path to the release zip (overrides positional arg)
#   MODEL_CACHE_SRC         - copy a pre-seeded HuggingFace cache from this
#                             directory into the test user's ~/.cache/huggingface
#   BADAPPLE_ALLOW_DOWNLOADS - if non-empty, the daemon will be allowed to
#                             download model weights on first run (uses network)
#   REPORT_PATH             - where to write the JSON report
#                             (default: bad_apple_vm_smoke_report_<timestamp>.json)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Configuration ---
BADAPPLE_ZIP="${BADAPPLE_ZIP:-${1:-}}"
MODEL_CACHE_SRC="${MODEL_CACHE_SRC:-}"
BADAPPLE_ALLOW_DOWNLOADS="${BADAPPLE_ALLOW_DOWNLOADS:-}"
REPORT_PATH="${REPORT_PATH:-${REPO_ROOT}/bad_apple_vm_smoke_report_$(date -u +%Y%m%dT%H%M%SZ).json}"

START_TIME_MS=$(python3 -c 'import time; print(int(time.time()*1000))' 2>/dev/null || date +%s000)

# --- Colors and helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0
NOTES=()
JSTEPS=()

ok()   { echo -e "${GREEN}✓${NC} $1"; PASS=$((PASS+1)); add_step "$1" "pass" ""; }
ko()   { echo -e "${RED}✗${NC} $1"; FAIL=$((FAIL+1)); add_step "$1" "fail" "${2:-}"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; WARN=$((WARN+1)); add_step "$1" "warn" ""; }
note() { NOTES+=("$1"); }

add_step() {
  local name="$1" status="$2" detail="${3:-}"
  local name_b64 status_b64 detail_b64
  name_b64=$(printf '%s' "$name" | /usr/bin/base64 | tr -d '\n')
  status_b64=$(printf '%s' "$status" | /usr/bin/base64 | tr -d '\n')
  detail_b64=$(printf '%s' "$detail" | /usr/bin/base64 | tr -d '\n')
  JSTEPS+=("{\"name_b64\":\"$name_b64\",\"status_b64\":\"$status_b64\",\"detail_b64\":\"$detail_b64\"}")
}

write_report() {
  local status="pass"
  [[ $FAIL -eq 0 ]] || status="fail"
  END_TIME_MS=$(python3 -c 'import time; print(int(time.time()*1000))' 2>/dev/null || date +%s000)
  local steps="[$(printf '%s,' "${JSTEPS[@]}" | sed 's/,$//')]"
  local notes_b64
  notes_b64=$(printf '%s\n' "${NOTES[@]}" | /usr/bin/base64 | tr -d '\n')
  cat > "$REPORT_PATH" <<EOF
{
  "status": "$status",
  "start_ms": $START_TIME_MS,
  "end_ms": $END_TIME_MS,
  "pass": $PASS,
  "fail": $FAIL,
  "warn": $WARN,
  "steps": $steps,
  "notes_b64": "$notes_b64"
}
EOF
  echo ""
  echo "Report: $REPORT_PATH"
  if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}Smoke test passed.${NC}"
  else
    echo -e "${RED}Smoke test failed.${NC}"
  fi
}

trap write_report EXIT

# --- Pre-flight checks ---
echo "=== Bad Apple Clean-Machine VM Smoke Test ==="
echo ""

if [[ "$(uname -s)" != "Darwin" ]]; then
  ko "macOS required" "this script is only for macOS"
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  warn "not Apple Silicon; Bad Apple is optimized for Apple Silicon"
fi

PRODUCT_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
if [[ -z "$PRODUCT_VERSION" ]]; then
  ko "could not read macOS version"
  exit 1
fi
ok "macOS version: $PRODUCT_VERSION"

FREE_GB=$(df -g / 2>/dev/null | tail -1 | awk '{print $4}')
if [[ -n "$FREE_GB" && "$FREE_GB" -lt 50 ]]; then
  ko "insufficient disk space" "need at least 50 GB free, found $FREE_GB GB"
  exit 1
fi
ok "disk space: ${FREE_GB:-unknown} GB free"

TOTAL_MEM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
if [[ "$TOTAL_MEM_GB" -lt 6 ]]; then
  ko "insufficient RAM" "Bad Apple needs ~6 GB of available unified memory; found $TOTAL_MEM_GB GB total"
  exit 1
fi
if [[ "$TOTAL_MEM_GB" -lt 8 ]]; then
  warn "only $TOTAL_MEM_GB GB RAM: Bad Apple itself uses ~6 GB; macOS may be tight"
else
  if [[ "$TOTAL_MEM_GB" -lt 16 ]]; then
    ok "RAM: ${TOTAL_MEM_GB} GB (works; 16 GB is more comfortable)"
  else
    ok "RAM: ${TOTAL_MEM_GB} GB"
  fi
fi

if [[ -z "$BADAPPLE_ZIP" ]]; then
  # Try to find the latest full-unsigned zip in the release dir
  BADAPPLE_ZIP=$(find "$REPO_ROOT/target/release" -maxdepth 1 -name 'Bad_Apple-*-full-unsigned.zip' 2>/dev/null | sort | tail -1)
fi

if [[ -z "$BADAPPLE_ZIP" || ! -f "$BADAPPLE_ZIP" ]]; then
  ko "release zip not found" "set BADAPPLE_ZIP or pass the path to a Bad_Apple-<version>-full-unsigned.zip"
  exit 1
fi
ok "release zip: $BADAPPLE_ZIP"

if [[ $EUID -ne 0 ]]; then
  ko "not running as root" "re-run with: sudo -E ./tests/vm_smoke_test.sh ..."
  exit 1
fi
ok "running as root"

CONSOLE_USER="${SUDO_USER:-$(logname 2>/dev/null || id -un)}"
CONSOLE_UID=$(id -u "$CONSOLE_USER")
CONSOLE_HOME=$(eval echo "~$CONSOLE_USER")
ok "console user: $CONSOLE_USER (uid $CONSOLE_UID, home $CONSOLE_HOME)"

# --- Extract release ---
EXTRACT_DIR=$(mktemp -d)
trap 'rm -rf "$EXTRACT_DIR"' EXIT

note "extract dir: $EXTRACT_DIR"

if unzip -q "$BADAPPLE_ZIP" -d "$EXTRACT_DIR"; then
  ok "extracted release zip"
else
  ko "failed to extract release zip"
  exit 1
fi

PKG_DIR=$(find "$EXTRACT_DIR" -maxdepth 1 -type d -name 'Bad_Apple-*-full' | head -n1)
if [[ -z "$PKG_DIR" || ! -d "$PKG_DIR" ]]; then
  ko "release package directory not found inside zip"
  exit 1
fi
ok "package directory: $PKG_DIR"

REPO_IN_PKG="${PKG_DIR}/bad_apple"
if [[ ! -d "$REPO_IN_PKG" ]]; then
  ko "bad_apple source directory missing from package"
  exit 1
fi
BADAPPLE_CLI="${REPO_IN_PKG}/target/release/badapple"
if [[ ! -x "$BADAPPLE_CLI" ]]; then
  ko "badapple CLI missing in package"
  exit 1
fi
ok "found CLI: $BADAPPLE_CLI"

# --- Optional model cache seed ---
if [[ -n "$MODEL_CACHE_SRC" ]]; then
  if [[ -d "$MODEL_CACHE_SRC" ]]; then
    mkdir -p "${CONSOLE_HOME}/.cache/huggingface"
    cp -R "${MODEL_CACHE_SRC}/." "${CONSOLE_HOME}/.cache/huggingface/"
    chown -R "${CONSOLE_USER}:$(id -gn "$CONSOLE_USER")" "${CONSOLE_HOME}/.cache/huggingface"
    ok "seeded HuggingFace cache from $MODEL_CACHE_SRC"
  else
    warn "MODEL_CACHE_SRC not found: $MODEL_CACHE_SRC"
  fi
fi

# --- Allow downloads if requested ---
PLIST_TARGET="/Library/LaunchDaemons/com.badapple.mlx.plist"
if [[ -n "$BADAPPLE_ALLOW_DOWNLOADS" ]]; then
  note "BADAPPLE_ALLOW_DOWNLOADS is set; the daemon may download model weights"
fi

# --- Run installer ---
INSTALLER="${PKG_DIR}/install.sh"
if [[ ! -x "$INSTALLER" ]]; then
  chmod +x "$INSTALLER"
fi

if "$INSTALLER" >"${EXTRACT_DIR}/install.log" 2>&1; then
  ok "package installer completed"
else
  ko "package installer failed" "see ${EXTRACT_DIR}/install.log"
  tail -n 50 "${EXTRACT_DIR}/install.log" || true
  exit 1
fi

# Wait for install to fully apply
sleep 2

# If downloads are allowed, patch the rendered plist before the daemon tries to load
if [[ -n "$BADAPPLE_ALLOW_DOWNLOADS" && -f "$PLIST_TARGET" ]]; then
  if plutil -replace EnvironmentVariables.HF_HUB_OFFLINE -string 0 "$PLIST_TARGET" 2>/dev/null; then
    ok "patched HF_HUB_OFFLINE to 0 for model download"
    launchctl unload "$PLIST_TARGET" 2>/dev/null || true
    sleep 1
    launchctl load -w "$PLIST_TARGET" 2>/dev/null || true
  else
    warn "could not patch HF_HUB_OFFLINE; model download may fail"
  fi
fi

# --- Wait for daemon readiness ---
echo ""
echo "Waiting for Bad Apple daemons..."
ready=0
for i in {1..120}; do
  if [[ -S /var/run/badapple/substrate.sock && -S /var/run/badapple/substrate_mlx.sock ]]; then
    if sudo -u "$CONSOLE_USER" -E "$BADAPPLE_CLI" -n 4 "Reply only: ready" >/dev/null 2>&1; then
      ready=1
      break
    fi
  fi
  sleep 3
  echo -n "."
done
echo ""

if [[ $ready -eq 1 ]]; then
  ok "daemon is ready"
else
  ko "daemon did not become ready"
  note "mlx log tail:"
  tail -n 50 /var/log/bad_apple_mlx_server.log 2>/dev/null | while IFS= read -r line; do note "$line"; done || true
  exit 1
fi

# If we allowed downloads, wait for the model to cache, then restore offline mode
if [[ -n "$BADAPPLE_ALLOW_DOWNLOADS" && -f "$PLIST_TARGET" ]]; then
  echo ""
  echo "Waiting for model weights to cache (this may take several minutes)..."
  model_cached=0
  for i in {1..180}; do
    if tail -n 20 /var/log/bad_apple_mlx_server.log 2>/dev/null | grep -qi "model loaded\|loaded.*model"; then
      model_cached=1
      break
    fi
    sleep 5
    echo -n "."
  done
  echo ""

  if [[ $model_cached -eq 1 ]]; then
    ok "model weights cached"
    plutil -replace EnvironmentVariables.HF_HUB_OFFLINE -string 1 "$PLIST_TARGET" 2>/dev/null || true
    launchctl unload "$PLIST_TARGET" 2>/dev/null || true
    sleep 2
    launchctl load -w "$PLIST_TARGET" 2>/dev/null || true
    ok "restored HF_HUB_OFFLINE=1 for air-gap cert"

    # wait for daemon to come back
    for i in {1..120}; do
      if [[ -S /var/run/badapple/substrate.sock && -S /var/run/badapple/substrate_mlx.sock ]]; then
        if sudo -u "$CONSOLE_USER" -E "$BADAPPLE_CLI" -n 4 "Reply only: ready" >/dev/null 2>&1; then
          break
        fi
      fi
      sleep 3
    done
  else
    warn "model did not finish caching before timeout"
  fi
fi

# --- Cert suite ---
echo ""
echo "Running air-gap certification..."
if CERT_OUT=$(sudo -u "$CONSOLE_USER" -E "$BADAPPLE_CLI" cert 2>/dev/null); then
  if echo "$CERT_OUT" | grep -q '"failures": 0'; then
    ok "air-gap cert passed"
  else
    ko "air-gap cert reported failures" "$CERT_OUT"
  fi
else
  ko "air-gap cert command failed" "$CERT_OUT"
fi

# --- Doctor ---
echo ""
echo "Running doctor..."
if DOCTOR_OUT=$(sudo -u "$CONSOLE_USER" -E "$BADAPPLE_CLI" --doctor 2>/dev/null); then
  if echo "$DOCTOR_OUT" | grep -q "com.badapple.mlx: running"; then
    ok "doctor reports mlx daemon running"
  else
    warn "doctor did not report a running mlx daemon"
  fi
else
  ko "doctor command failed" "$DOCTOR_OUT"
fi

# --- Simple query ---
echo ""
echo "Running a simple query..."
if QUERY_OUT=$(sudo -u "$CONSOLE_USER" -E "$BADAPPLE_CLI" -n 24 "What is 2+2?" 2>/dev/null); then
  if echo "$QUERY_OUT" | grep -qi "4\|four"; then
    ok "query returned a sane answer"
  else
    warn "query returned an unexpected answer"
    note "query output: $QUERY_OUT"
  fi
else
  ko "query command failed" "$QUERY_OUT"
fi

# --- Menu bar agent (optional) ---
echo ""
echo "Checking menu bar agent..."
MENUBAR_PLIST="${CONSOLE_HOME}/Library/LaunchAgents/com.badapple.menubar.plist"
if [[ -f "$MENUBAR_PLIST" ]]; then
  if launchctl print "gui/${CONSOLE_UID}/com.badapple.menubar" >/dev/null 2>&1; then
    ok "menu bar LaunchAgent loaded"
  else
    warn "menu bar LaunchAgent not loaded (may require a GUI session)"
  fi
else
  warn "menu bar LaunchAgent plist not installed"
fi

# --- Crash report collection ---
if [[ $FAIL -gt 0 ]]; then
  echo ""
  echo "Collecting crash report..."
  if CRASH_OUT=$(sudo -u "$CONSOLE_USER" -E "$BADAPPLE_CLI" --crash-report 2>/dev/null); then
    note "crash report: $CRASH_OUT"
  else
    note "could not collect crash report"
  fi
fi

# Final report is written by the EXIT trap.
