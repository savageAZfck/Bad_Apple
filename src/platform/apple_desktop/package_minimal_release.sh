#!/usr/bin/env bash
set -euo pipefail

# Package a source-free consumer release of the Bad Apple OS.
# The output zip contains the app, the native platform binaries, and just
# enough install scripts/plists to run `sudo ./install.sh`. It does not
# include the full Rust/Swift source, so it is safe to publish in a public
# release repo while keeping the dev repo private.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VERSION="$(cd "${REPO_ROOT}" && grep '^version' Cargo.toml | head -n1 | sed -e 's/.*= *"//' -e 's/".*//')"
PKG_DIR="${REPO_ROOT}/target/release/Bad_Apple-${VERSION}-unsigned"
ZIP_PATH="${REPO_ROOT}/target/release/Bad_Apple-${VERSION}-unsigned.zip"

# Binaries required at runtime.
BINS=(
    badapple
    badapple-fetch
    badapple-identity
    badapple-identity-agent
    badapple-supervisor
    badapple-engine
    badapple-tts
    gatekeeper
)

# Libraries required at runtime.
LIBS=(
    libBadAppleBridge.dylib
    libBadAppleMLX.dylib
    libbad_apple.dylib
)

# LaunchDaemon plists and install scripts.
BRIDGE_FILES=(
    src/platform/apple_bridge/com.badapple.gatekeeper.plist
    src/platform/apple_bridge/com.badapple.mlx.plist
    src/platform/apple_bridge/com.badapple.supervisor.plist
    src/platform/apple_bridge/com.badapple.identity_agent.plist
    src/platform/apple_bridge/install_badapple_platform.sh
    src/platform/apple_bridge/install_identity_agent.sh
)

# LaunchAgent plists and install scripts.
DESKTOP_FILES=(
    src/platform/apple_desktop/com.badapple.menubar.plist
    src/platform/apple_desktop/com.badapple.tts.plist
    src/platform/apple_desktop/install_menu_bar_agent.sh
    src/platform/apple_desktop/install_tts_agent.sh
    src/platform/apple_desktop/strip_quarantine.sh
)

# Runtime hot-reloadable config.
CONFIG_FILES=(
    prompt.txt
    personas.json
)

echo "Building release binaries..."
cd "${REPO_ROOT}"
cargo build --release

echo "Building Bad Apple.app..."
BADAPPLE_NO_SIGN=1 "${REPO_ROOT}/src/platform/apple_desktop/build_bad_apple_menu_bar.sh"

echo "Building Apple bridge..."
bash "${REPO_ROOT}/src/platform/apple_bridge/build_apple_bridge.sh"

rm -rf "${PKG_DIR}"
mkdir -p "${PKG_DIR}/bad_apple" "${PKG_DIR}/bad_apple/target/release"

echo "Copying Bad Apple.app..."
cp -R "${REPO_ROOT}/target/release/Bad Apple.app" "${PKG_DIR}/Bad Apple.app"

echo "Copying native binaries..."
for bin in "${BINS[@]}"; do
    if [[ -x "${REPO_ROOT}/target/release/${bin}" ]]; then
        install -m 755 "${REPO_ROOT}/target/release/${bin}" "${PKG_DIR}/bad_apple/target/release/${bin}"
    fi
done

for lib in "${LIBS[@]}"; do
    if [[ -f "${REPO_ROOT}/target/release/${lib}" ]]; then
        install -m 755 "${REPO_ROOT}/target/release/${lib}" "${PKG_DIR}/bad_apple/target/release/${lib}"
    fi
done

# The engine also needs the pre-compiled Metal shader library.
if [[ -f "${REPO_ROOT}/target/release/mlx.metallib" ]]; then
    install -m 644 "${REPO_ROOT}/target/release/mlx.metallib" "${PKG_DIR}/bad_apple/target/release/mlx.metallib"
fi

copy_file() {
    local src="$1" dst="$2"
    mkdir -p "$(dirname "${dst}")"
    install -m 644 "${src}" "${dst}"
}

echo "Copying install scripts and plists..."
for rel in "${BRIDGE_FILES[@]}" "${DESKTOP_FILES[@]}"; do
    copy_file "${REPO_ROOT}/${rel}" "${PKG_DIR}/bad_apple/${rel}"
    case "${rel}" in
        *.sh) chmod +x "${PKG_DIR}/bad_apple/${rel}" ;;
    esac
done

for rel in "${CONFIG_FILES[@]}"; do
    copy_file "${REPO_ROOT}/${rel}" "${PKG_DIR}/bad_apple/${rel}"
done

# Top-level helpers for the manual path.
install -m 755 "${REPO_ROOT}/src/platform/apple_desktop/first_run_preflight.sh" "${PKG_DIR}/first_run_preflight.sh"
install -m 755 "${REPO_ROOT}/src/platform/apple_desktop/strip_quarantine.sh" "${PKG_DIR}/strip_quarantine.sh"

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
- 8 GB of unified memory minimum, 16 GB recommended

The first launch will download model weights (~4-6 GB) if they are not cached.
For an air-gap install, seed the model cache before running the installer.

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

# Link the native CLI and fetch helper into a standard PATH directory so
# `badapple --doctor` and `badapple-fetch` work from a fresh Terminal.
echo "Linking CLI into /usr/local/bin..."
install -d /usr/local/bin
ln -sf "${REPO_ROOT}/target/release/badapple" /usr/local/bin/badapple
ln -sf "${REPO_ROOT}/target/release/badapple-fetch" /usr/local/bin/badapple-fetch

echo "Bad Apple is installed. Launch it from /Applications."
EOF
chmod +x "${PKG_DIR}/install.sh"

rm -f "${ZIP_PATH}"
( cd "${REPO_ROOT}/target/release" && zip -r -y -9 "${ZIP_PATH}" "Bad_Apple-${VERSION}-unsigned" )

echo "Packaged: ${ZIP_PATH}"
