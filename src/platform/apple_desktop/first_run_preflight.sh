#!/usr/bin/env bash
set -euo pipefail

# Bad Apple first-run preflight.
#
# This script checks the machine before installation and prints actionable
# guidance. It does not modify the system. Run as root from install.sh, or
# standalone with `bash first_run_preflight.sh`.

__p() {
  local code="\e[${1}m"
  shift
  printf "${code}%s\e[0m\n" "$*"
}
info()    { __p "34"  "ℹ  $*"; }
success() { __p "32"  "✓  $*"; }
warn()    { __p "33"  "⚠  $*"; }
fail()    { __p "31"  "✗  $*"; }

MIN_MACOS=26.0
MIN_RAM_GB=6
REC_RAM_GB=8
COMFORT_RAM_GB=16
MIN_FREE_GB=40

error_count=0
warning_count=0

info "Checking Bad Apple first-run requirements..."
echo ""

# macOS version
macos_version=$(sw_vers -productVersion 2>/dev/null || echo "0")
major=$(echo "$macos_version" | cut -d. -f1)
minor=$(echo "$macos_version" | cut -d. -f2)
macos_float="${major}.${minor}"
if awk "BEGIN { exit !($macos_float >= $MIN_MACOS) }"; then
  success "macOS $macos_version"
else
  fail "macOS $macos_version is too old. Bad Apple requires macOS $MIN_MACOS or later."
  ((error_count++)) || true
fi

# Apple Silicon
arch=$(uname -m)
if [[ "$arch" == "arm64" ]]; then
  success "Apple Silicon ($arch)"
else
  fail "Bad Apple runs on Apple Silicon (arm64) only. This machine is $arch."
  ((error_count++)) || true
fi

# Memory
ram_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
ram_gb=$((ram_bytes / 1024 / 1024 / 1024))
if [[ "$ram_gb" -lt "$MIN_RAM_GB" ]]; then
  fail "Only $ram_gb GB RAM. Bad Apple needs at least ~$MIN_RAM_GB GB total (the 7B model uses about 4 GB at peak)."
  ((error_count++)) || true
elif [[ "$ram_gb" -lt "$REC_RAM_GB" ]]; then
  warn "$ram_gb GB RAM. This is below the recommended $REC_RAM_GB GB. macOS may be tight."
  ((warning_count++)) || true
elif [[ "$ram_gb" -lt "$COMFORT_RAM_GB" ]]; then
  success "$ram_gb GB RAM (works; 16 GB is more comfortable)"
else
  success "$ram_gb GB RAM"
fi

# Free disk space
free_gb=$(($(df -Pk / | awk 'NR==2 {print $4}') / 1024 / 1024))
if [[ "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  warn "Only $free_gb GB free. Bad Apple needs at least $MIN_FREE_GB GB for the OS, model, and swap."
  ((warning_count++)) || true
else
  success "$free_gb GB free disk space"
fi

# Xcode / Command Line Tools
if xcode-select -p &>/dev/null && [[ -d "$(xcode-select -p)" ]]; then
  success "Xcode / Command Line Tools installed at $(xcode-select -p)"
else
  fail "Xcode Command Line Tools not found. Run: xcode-select --install"
  ((error_count++)) || true
fi

# Admin / sudo
if [[ "$(id -u)" -eq 0 ]]; then
  success "Running as root"
else
  warn "Not running as root. Run this installer with sudo."
  ((warning_count++)) || true
fi

# Model cache
model_cached=false
for cache_dir in "$HOME/.cache/badapple" "$HOME/.cache/huggingface"; do
  if [[ -d "$cache_dir" ]] && find "$cache_dir" -type f -name "*.safetensors" 2>/dev/null | grep -q .; then
    model_cached=true
    break
  fi
done

if $model_cached; then
  success "Model weights cache found"
else
  warn "No model weights cache found. The first launch will download the 7B model (~4-6 GB)."
  warn "  - For air-gap: seed the cache before installing (see tests/vm_smoke_test.sh)."
  warn "  - Otherwise: set BADAPPLE_ALLOW_DOWNLOADS=1 before the first query."
  ((warning_count++)) || true
fi

# 8 GB memory guidance
if [[ "$ram_gb" -ge "$REC_RAM_GB" && "$ram_gb" -lt "$COMFORT_RAM_GB" ]]; then
  echo ""
  info "8 GB RAM tuning recommendation:"
  echo "  - Lower the KV cache to 2048 after install:"
  echo "      badapple 'set max kv size to 2048'"
  echo "  - Or edit BADAPPLE_MAX_KV_SIZE in com.badapple.mlx.plist and reload."
fi

echo ""
if [[ "$error_count" -gt 0 ]]; then
  fail "Preflight failed with $error_count error(s) and $warning_count warning(s)."
  exit 1
fi

if [[ "$warning_count" -gt 0 ]]; then
  warn "Preflight passed with $warning_count warning(s)."
else
  success "Preflight passed. This Mac looks ready for Bad Apple."
fi

exit 0
