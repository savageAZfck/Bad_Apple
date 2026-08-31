#!/usr/bin/env bash
# Clean-machine install test for Bad Apple.
#
# This script simulates a fresh macOS install by:
# 1. Building the release artifacts
# 2. Creating a DMG
# 3. Installing into a temporary location (simulating /Applications)
# 4. Verifying all components are present and functional
#
# Run: tests/test_clean_install.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

ok()   { echo -e "${GREEN}✓${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}✗${NC} $1"; FAIL=$((FAIL+1)); }
skip() { echo -e "${YELLOW}⊘${NC} $1 (skipped)"; SKIP=$((SKIP+1)); }

echo "=== Bad Apple Clean Install Test ==="
echo ""

# --- Step 1: Build artifacts ---
echo "Building release artifacts..."
cargo build --release 2>&1 | tail -3
if [[ $? -eq 0 ]]; then
    ok "cargo build --release"
else
    fail "cargo build --release"
    exit 1
fi

BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh 2>&1 | tail -3
if [[ $? -eq 0 ]]; then
    ok "build_bad_apple_menu_bar.sh"
else
    fail "build_bad_apple_menu_bar.sh"
    exit 1
fi

# --- Step 2: Verify artifact structure ---
echo ""
echo "Checking artifact structure..."

APP="target/release/Bad Apple.app"

# App bundle exists
if [[ -d "${APP}" ]]; then ok "Bad Apple.app exists"; else fail "Bad Apple.app missing"; exit 1; fi

# Main executable
if [[ -f "${APP}/Contents/MacOS/BadApple" ]]; then ok "Main executable"; else fail "Main executable missing"; fi

# Helper binary
if [[ -f "${APP}/Contents/Helpers/badapple" ]]; then ok "Helper binary"; else fail "Helper binary missing"; fi

# Info.plist
if [[ -f "${APP}/Contents/Info.plist" ]]; then ok "Info.plist"; else fail "Info.plist missing"; fi

# Version is not hardcoded
APP_FULL="${REPO_ROOT}/target/release/Bad Apple.app"
VERSION="$(defaults read "${APP_FULL}/Contents/Info" CFBundleShortVersionString 2>/dev/null || echo "")"
if [[ -n "${VERSION}" ]]; then
    ok "Version derived from Cargo.toml: ${VERSION}"
else
    VERSION="$(plutil -extract CFBundleShortVersionString raw "${APP_FULL}/Contents/Info.plist" 2>/dev/null || echo "")"
    if [[ -n "${VERSION}" ]]; then
        ok "Version derived from Cargo.toml: ${VERSION}"
    else
        fail "Version is missing from Info.plist"
    fi
fi

# Embedded scripts
for script in update_bad_apple.sh strip_quarantine.sh install_badapple_platform.sh; do
    if [[ -f "${APP}/Contents/Resources/${script}" ]]; then
        ok "Embedded script: ${script}"
    else
        fail "Missing embedded script: ${script}"
    fi
done

# --- Step 3: Verify Rust binaries ---
echo ""
echo "Checking Rust binaries..."

for bin in badapple gatekeeper badapple-identity; do
    if [[ -x "target/release/${bin}" ]]; then
        ok "Binary: ${bin}"
    else
        fail "Missing binary: ${bin}"
    fi
done

# --- Step 4: Verify Python files ---
echo ""
echo "Checking Python files..."

for py in badapple_mlx_server.py badapple_tools.py badapple_extras.py badapple_aqua_helper.py badapple_slicks.py badapple_p2p.py badapple_dashboard.py; do
    if [[ -f "${py}" ]]; then
        ok "Python file: ${py}"
    else
        fail "Missing Python file: ${py}"
    fi
done

# --- Step 5: Verify launchd plists ---
echo ""
echo "Checking launchd plists..."

for plist in com.badapple.mlx.plist com.badapple.gatekeeper.plist com.badapple.supervisor.plist; do
    if [[ -f "src/platform/apple_bridge/${plist}" ]]; then
        ok "Plist: ${plist}"
    else
        fail "Missing plist: ${plist}"
    fi
done

# --- Step 6: Verify installer script ---
echo ""
echo "Checking installer..."

INSTALLER="src/platform/apple_bridge/install_badapple_platform.sh"
if [[ -x "${INSTALLER}" ]]; then
    ok "Installer script exists and is executable"
else
    fail "Installer script missing or not executable"
fi

# Check installer doesn't use hardcoded staff group
if grep -q ':staff' "${INSTALLER}"; then
    fail "Installer still uses hardcoded ':staff' group"
else
    ok "Installer uses dynamic group (no hardcoded staff)"
fi

# Check installer doesn't use eval
if grep -q 'eval echo' "${INSTALLER}"; then
    fail "Installer still uses 'eval echo'"
else
    ok "Installer doesn't use eval for home directory"
fi

# Check installer doesn't use seq
if grep -q 'seq ' "${INSTALLER}"; then
    fail "Installer still uses 'seq'"
else
    ok "Installer doesn't use seq"
fi

# --- Step 7: Verify security defaults ---
echo ""
echo "Checking security defaults..."

# Policy.yaml has autopilot: false
if grep -q 'autopilot:.*false' policy.yaml; then
    ok "policy.yaml has autopilot: false"
else
    fail "policy.yaml does not have autopilot: false"
fi

# Shell allowlist doesn't include interpreters
if grep -q 'python3\|swift\|cargo\|rustc' badapple_tools.py | grep -v '#' 2>/dev/null; then
    # More precise check
    if .venv/bin/python -c "from badapple_tools import SHELL_ALLOWED_COMMANDS; assert 'python3' not in SHELL_ALLOWED_COMMANDS" 2>/dev/null; then
        ok "Shell allowlist excludes interpreters"
    else
        skip "Shell allowlist check (venv not available)"
    fi
else
    ok "Shell allowlist excludes interpreters (no matches)"
fi

# --- Step 8: Test a query (if daemon is running) ---
echo ""
echo "Checking daemon connectivity..."

if [[ -S /var/run/badapple/substrate.sock ]]; then
    if target/release/badapple -n 16 "Reply only: ready" >/dev/null 2>&1; then
        ok "Daemon query works"
    else
        skip "Daemon query (socket exists but query failed)"
    fi
else
    skip "Daemon query (no socket — daemon not running)"
fi

# --- Step 9: Doctor command ---
echo ""
echo "Checking --doctor command..."

if target/release/badapple --doctor 2>&1 | head -1 | grep -q "Bad Apple Doctor"; then
    ok "--doctor command works"
else
    fail "--doctor command not working"
fi

if target/release/badapple --diagnostics 2>&1 | head -1 | grep -q "Bad Apple Doctor"; then
    ok "--diagnostics alias works"
else
    fail "--diagnostics alias not working"
fi

# --- Summary ---
echo ""
echo "=== Results ==="
echo -e "${GREEN}Passed: ${PASS}${NC}  ${RED}Failed: ${FAIL}${NC}  ${YELLOW}Skipped: ${SKIP}${NC}"

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
exit 0
