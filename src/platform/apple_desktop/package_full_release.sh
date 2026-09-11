#!/usr/bin/env bash
set -euo pipefail

# Package a one-command consumer release of the Bad Apple OS.
# The output is a zip the user extracts and runs with `sudo ./install.sh`.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VERSION="$(cd "${REPO_ROOT}" && grep '^version' Cargo.toml | head -n1 | sed -e 's/.*= *"//' -e 's/".*//')"
PKG_DIR="${REPO_ROOT}/target/release/Bad_Apple-${VERSION}-full"
ZIP_PATH="${REPO_ROOT}/target/release/Bad_Apple-${VERSION}-full-unsigned.zip"

echo "Building release binaries..."
cd "${REPO_ROOT}"
cargo build --release

echo "Building Bad Apple.app..."
BADAPPLE_NO_SIGN=1 "${REPO_ROOT}/src/platform/apple_desktop/build_bad_apple_menu_bar.sh"

echo "Building Apple bridge (badapple-identity)..."
bash "${REPO_ROOT}/src/platform/apple_bridge/build_apple_bridge.sh"

rm -rf "${PKG_DIR}"
mkdir -p "${PKG_DIR}/bad_apple"

echo "Copying Bad Apple.app..."
cp -R "${REPO_ROOT}/target/release/Bad Apple.app" "${PKG_DIR}/Bad Apple.app"

echo "Copying platform files..."
rsync -a \
  --exclude='.git' --exclude='.DS_Store' --exclude='.github' \
  --exclude='.cargo' --exclude='target' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.py' \
  --exclude='data' --exclude='strategy_db' --exclude='voices' \
  --exclude='tests/ane_brain_perf' --exclude='qwen1.7b_*' --exclude='curriculum' \
  --exclude='state.*' --exclude='state-backup*' --exclude='scavenger_paths.json' --exclude='tokenizer.json' \
  --exclude='.badapple_dev_cert.*' --exclude='sapient_agi_soul*' --exclude='test_*.wasm' \
  --exclude='test_cage' --exclude='wild_workspace' --exclude='*.defense' --exclude='*.network' --exclude='*.weights' \
  --exclude='src/platform/apple_desktop/.build' --exclude='src/platform/apple_desktop/.swiftpm' \
  --exclude='MLXInference/.build' --exclude='MLXInference/.swiftpm' \
  --exclude='.build' --exclude='.swiftpm' \
  --exclude='requirements.in' \
  "${REPO_ROOT}/" "${PKG_DIR}/bad_apple/"

# Copy only the release binaries we need for the platform.
install -d "${PKG_DIR}/bad_apple/target/release"
for bin in badapple gatekeeper badapple-identity badapple-identity-agent badapple-supervisor badapple-engine; do
    if [[ -x "${REPO_ROOT}/target/release/${bin}" ]]; then
        install -m 755 "${REPO_ROOT}/target/release/${bin}" "${PKG_DIR}/bad_apple/target/release/${bin}"
    fi
done
for lib in libBadAppleBridge.dylib libbad_apple.dylib; do
    if [[ -f "${REPO_ROOT}/target/release/${lib}" ]]; then
        install -m 755 "${REPO_ROOT}/target/release/${lib}" "${PKG_DIR}/bad_apple/target/release/${lib}"
    fi
done

# The app bundle already contains updater/strip scripts, but keep a copy at the top level.
install -m 755 "${REPO_ROOT}/src/platform/apple_desktop/strip_quarantine.sh" "${PKG_DIR}/strip_quarantine.sh"
install -m 755 "${REPO_ROOT}/src/platform/apple_desktop/first_run_preflight.sh" "${PKG_DIR}/first_run_preflight.sh"

cat > "${PKG_DIR}/README.txt" <<'EOF'
Welcome to Bad Apple!

This is a self-contained, unsigned release. It does not require an Apple
Developer ID and never phones home to Apple for notarization.

Quick install:

    sudo ./install.sh

The installer runs a friendly preflight that checks your Mac (macOS version,
Apple Silicon, RAM, free disk, Xcode tools, and model cache). If anything is
off, it tells you exactly what to do.

What the installer does:
- Copies Bad Apple.app into /Applications.
- Strips the Gatekeeper quarantine flag.
- Installs the native system daemons (no Python venv required).
- Tunes the KV cache and memory governor to your Mac's RAM.
- Starts the menu bar.

Requirements:
- macOS 26.0 or later
- Apple Silicon (M1 or newer)
- 8 GB of unified memory minimum, 16 GB recommended. The 7B model uses about
  4 GB at peak.
- 40 GB of free disk space for the OS, model cache, and swap.

The first launch will download model weights (~4-6 GB) if they are not cached.
For an air-gap install, seed the model cache before running the installer.

For the manual path, copy Bad Apple.app to /Applications, then run the
platform installer from a checkout as documented in INSTALL.md.

To update later, use Bad Apple > Check for Updates in the menu bar.
EOF

cat > "${PKG_DIR}/install.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Please run this installer as root: sudo ./install.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run a friendly preflight that checks RAM, disk, macOS version, and model cache.
"${SCRIPT_DIR}/first_run_preflight.sh"
REPO_ROOT="${SCRIPT_DIR}/bad_apple"
APP="${SCRIPT_DIR}/Bad Apple.app"

if [[ ! -d "${APP}" ]]; then
    echo "Bad Apple.app not found in this package" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "bad_apple platform directory not found in this package" >&2
    exit 1
fi

echo "Copying Bad Apple.app to /Applications..."
rm -rf "/Applications/Bad Apple.app"
cp -R "${APP}" "/Applications/Bad Apple.app"

echo "Stripping quarantine flag..."
"${APP}/Contents/Resources/strip_quarantine.sh" 2>/dev/null || xattr -dr com.apple.quarantine "/Applications/Bad Apple.app" 2>/dev/null || true

echo "Installing platform from ${REPO_ROOT}..."
cd "${REPO_ROOT}"
BADAPPLE_ROOT="${REPO_ROOT}" src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install

echo "Installing menu bar LaunchAgent..."
CONSOLE_USER="${SUDO_USER:-$(logname 2>/dev/null || id -un)}"
CONSOLE_HOME=$(eval echo "~$CONSOLE_USER")
# The LaunchAgent must be installed in the logged-in user's domain.
sudo -u "$CONSOLE_USER" -E HOME="$CONSOLE_HOME" src/platform/apple_desktop/install_menu_bar_agent.sh

echo "Bad Apple is installed. Launch it from /Applications."
EOF
chmod +x "${PKG_DIR}/install.sh"

rm -f "${ZIP_PATH}"
( cd "${REPO_ROOT}/target/release" && zip -r -y -9 "${ZIP_PATH}" "Bad_Apple-${VERSION}-full" )

echo "Packaged: ${ZIP_PATH}"
