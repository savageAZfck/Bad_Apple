#!/usr/bin/env bash
set -euo pipefail

# Package a source-free consumer release of the Bad Apple OS.
# The output zip contains the app, the native platform binaries, and just
# enough install scripts/plists to run `sudo ./install.sh`. It does not
# include the full Rust/Swift source, so it is safe to publish in a public
# release repo while keeping the dev repo private.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VERSION="$(cd "${REPO_ROOT}" && grep '^version' Cargo.toml | head -n1 | sed -e 's/.*= *"//' -e 's/".*//')"
BUILD_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}/release"
PKG_DIR="${BUILD_DIR}/Bad_Apple-${VERSION}-unsigned"
ZIP_PATH="${BUILD_DIR}/Bad_Apple-${VERSION}-unsigned.zip"
COMPAT_DIR="${BUILD_DIR}/Bad_Apple-${VERSION}-full"
COMPAT_ZIP="${BUILD_DIR}/Bad_Apple-${VERSION}-full-unsigned.zip"
CHECKSUMS="${BUILD_DIR}/checksums.txt"

# Binaries required at runtime.
BINS=(
    badapple
    badapple-fetch
    badapple-identity
    badapple-identity-agent
    badapple-supervisor
    badapple-engine
    badapple-tts
    badapple-dashboard
    badapple-sovereign
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
    src/platform/apple_desktop/com.badapple.dashboard.plist
    src/platform/apple_desktop/com.badapple.checkpoint.plist
    src/platform/apple_desktop/install_menu_bar_agent.sh
    src/platform/apple_desktop/install_tts_agent.sh
    src/platform/apple_desktop/install_dashboard_agent.sh
    src/platform/apple_desktop/install_checkpoint_agent.sh
    src/platform/apple_desktop/strip_quarantine.sh
)

# Runtime hot-reloadable config.
CONFIG_FILES=(
    prompt.txt
    personas.json
)

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]] || fail "invalid package version: ${VERSION}"
for output in "${PKG_DIR}" "${ZIP_PATH}" "${COMPAT_DIR}" "${COMPAT_ZIP}" "${CHECKSUMS}"; do
    [[ ! -e "${output}" && ! -L "${output}" ]] || fail "refusing to replace existing release artifact: ${output}"
done

if [[ "${BADAPPLE_SKIP_BUILD:-0}" != 1 ]]; then
    echo "Building release binaries..."
    cd "${REPO_ROOT}"
    cargo build --release

    echo "Building Bad Apple.app..."
    BADAPPLE_NO_SIGN=1 "${REPO_ROOT}/src/platform/apple_desktop/build_bad_apple_menu_bar.sh"

    echo "Building Apple bridge..."
    bash "${REPO_ROOT}/src/platform/apple_bridge/build_apple_bridge.sh"
fi

for bin in "${BINS[@]}"; do
    [[ -x "${BUILD_DIR}/${bin}" ]] || fail "required executable missing: ${bin}"
done
for rel in "${LIBS[@]}" mlx.metallib; do
    [[ -s "${BUILD_DIR}/${rel}" ]] || fail "required runtime library missing: ${rel}"
done
for rel in "${BRIDGE_FILES[@]}" "${DESKTOP_FILES[@]}" "${CONFIG_FILES[@]}" src/platform/apple_desktop/first_run_preflight.sh; do
    [[ -s "${REPO_ROOT}/${rel}" ]] || fail "required package file missing: ${rel}"
done
APP_SOURCE="${BUILD_DIR}/Bad Apple.app"
[[ -x "${APP_SOURCE}/Contents/MacOS/BadApple" ]] || fail "Bad Apple.app executable missing"
APP_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "${APP_SOURCE}/Contents/Info.plist")"
[[ "${APP_VERSION}" == "${VERSION}" ]] || fail "app version ${APP_VERSION} does not match package version ${VERSION}"
for rel in Contents/Frameworks/libBadAppleBridge.dylib Contents/Frameworks/libbad_apple.dylib Contents/Libraries/libBadAppleMLX.dylib Contents/Libraries/mlx.metallib Contents/Resources/update_bad_apple.sh Contents/Resources/strip_quarantine.sh Contents/Resources/web/index.html; do
    [[ -s "${APP_SOURCE}/${rel}" ]] || fail "required app resource missing: ${rel}"
done
for helper in badapple badapple-fetch BadAppleScreenCapture BadAppleAmbient BadAppleUI badapple-dashboard; do
    [[ -x "${APP_SOURCE}/Contents/Helpers/${helper}" ]] || fail "required app helper missing: ${helper}"
done
cmp -s "${REPO_ROOT}/src/platform/apple_desktop/update_bad_apple.sh" "${APP_SOURCE}/Contents/Resources/update_bad_apple.sh" || fail "bundled updater is stale; rebuild the app"

mkdir "${PKG_DIR}"
mkdir -p "${PKG_DIR}/bad_apple/target/release"

echo "Copying Bad Apple.app..."
cp -R "${BUILD_DIR}/Bad Apple.app" "${PKG_DIR}/Bad Apple.app"

echo "Copying native binaries..."
for bin in "${BINS[@]}"; do
    install -m 755 "${BUILD_DIR}/${bin}" "${PKG_DIR}/bad_apple/target/release/${bin}"
done

for lib in "${LIBS[@]}"; do
    install -m 755 "${BUILD_DIR}/${lib}" "${PKG_DIR}/bad_apple/target/release/${lib}"
done

# The engine also needs the pre-compiled Metal shader library.
install -m 644 "${BUILD_DIR}/mlx.metallib" "${PKG_DIR}/bad_apple/target/release/mlx.metallib"

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

# Bundle the web dashboard assets so the packaged runtime can serve /control.
if [[ -d "${REPO_ROOT}/web" ]]; then
    mkdir -p "${PKG_DIR}/bad_apple/web"
    cp -R "${REPO_ROOT}/web/." "${PKG_DIR}/bad_apple/web/"
    find "${PKG_DIR}/bad_apple/web" -type f -exec chmod 644 {} + 2>/dev/null || true
    find "${PKG_DIR}/bad_apple/web" -type d -exec chmod 755 {} + 2>/dev/null || true
fi

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
- Starts the local web Control Center at http://127.0.0.1:8787/control.

Requirements:
- macOS 26.0 or later
- Apple Silicon (M1 or newer)
- 8 GB of unified memory minimum, 16 GB recommended. The 7B model uses about
  4 GB at peak.
- 40 GB of free disk space for the OS, model cache, and swap. The 7B model
  weights are about 4-6 GB on disk once cached.

Model downloads (~4-6 GB for the default model) require explicit permission
through the menu bar or BADAPPLE_ALLOW_DOWNLOADS=1. Cached weights are reused.
For an air-gap install, obtain and transfer the model cache before disconnecting.

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

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
CONSOLE_USER="${CONSOLE_USER:-${SUDO_USER:-$(stat -f %Su /dev/console)}}"
CONSOLE_UID="$(id -u "${CONSOLE_USER}")"
[[ "${CONSOLE_UID}" -ne 0 ]] || fail "install requires a logged-in non-root user; set CONSOLE_USER explicitly"
launchctl print "gui/${CONSOLE_UID}" >/dev/null || fail "no GUI session for ${CONSOLE_USER}"
CONSOLE_HOME="$(dscl . -read "/Users/${CONSOLE_USER}" NFSHomeDirectory | sed 's/^NFSHomeDirectory: //')"
[[ -d "${CONSOLE_HOME}" ]] || fail "cannot resolve home for ${CONSOLE_USER}"
export CONSOLE_USER
run_user() { launchctl asuser "${CONSOLE_UID}" sudo -u "${CONSOLE_USER}" env HOME="${CONSOLE_HOME}" USER="${CONSOLE_USER}" "$@"; }
PLIST="${CONSOLE_HOME}/Library/LaunchAgents/com.badapple.menubar.plist"
stop_menu() {
    run_user launchctl unload "${PLIST}" 2>/dev/null || true
    launchctl bootout "gui/${CONSOLE_UID}/com.badapple.menubar" 2>/dev/null || true
    if run_user launchctl list com.badapple.menubar >/dev/null 2>&1; then
        fail "could not stop menu LaunchAgent; app was not replaced"
    fi
    pkill -u "${CONSOLE_UID}" -x BadApple 2>/dev/null || true
    for _ in {1..30}; do
        pgrep -u "${CONSOLE_UID}" -x BadApple >/dev/null || return 0
        sleep 1
    done
    fail "Bad Apple did not stop; app was not replaced"
}
case "${1:-}" in
    --stop-menu) stop_menu; exit 0 ;;
    "") ;;
    *) fail "unknown installer option: $1" ;;
esac

# Run a friendly preflight that checks RAM, disk, macOS version, and model cache.
HOME="${CONSOLE_HOME}" "${SCRIPT_DIR}/first_run_preflight.sh"

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "${APP}/Contents/Info.plist")"
[[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]] || fail "invalid package version"
RUNTIME_BASE="/Library/Application Support/Bad Apple"
install -d -o root -g wheel -m 755 "${RUNTIME_BASE}" "${RUNTIME_BASE}/runtimes" "${RUNTIME_BASE}/backups"
RUNTIME="$(mktemp -d "${RUNTIME_BASE}/runtimes/${VERSION}.XXXXXX")"
chmod 755 "${RUNTIME}"
cp -R "${REPO_ROOT}/." "${RUNTIME}/"
chown -R root:wheel "${RUNTIME}"
xattr -dr com.apple.quarantine "${RUNTIME}" 2>/dev/null || true
REPO_ROOT="${RUNTIME}"
BACKUP="$(mktemp -d "${RUNTIME_BASE}/backups/${VERSION}.XXXXXX")"
APP_TARGET="/Applications/Bad Apple.app"
APP_REPLACED=0
APP_BACKED_UP=0
MENU_WAS_LOADED=0
run_user launchctl list com.badapple.menubar >/dev/null 2>&1 && MENU_WAS_LOADED=1
cp -RH "${APP}" "${BACKUP}/incoming.app"
for cli in badapple badapple-fetch; do
    if [[ -e "/usr/local/bin/${cli}" && ! -L "/usr/local/bin/${cli}" ]]; then
        fail "refusing to overwrite non-symlink /usr/local/bin/${cli}"
    fi
    if [[ -L "/usr/local/bin/${cli}" ]]; then
        cp -P "/usr/local/bin/${cli}" "${BACKUP}/${cli}"
    fi
done

rollback_app() {
    local status=$?
    trap - EXIT
    if [[ "${status}" -ne 0 ]]; then
        set +e
        run_user launchctl unload "${PLIST}" 2>/dev/null
        launchctl bootout "gui/${CONSOLE_UID}/com.badapple.menubar" 2>/dev/null
        pkill -u "${CONSOLE_UID}" -x BadApple 2>/dev/null
        if [[ "${APP_REPLACED}" -eq 1 && -d "${APP_TARGET}" ]]; then
            mv "${APP_TARGET}" "${BACKUP}/failed.app"
        fi
        if [[ "${APP_BACKED_UP}" -eq 1 ]]; then
            mv "${BACKUP}/Bad Apple.app" "${APP_TARGET}"
        fi
        for cli in badapple badapple-fetch; do
            if [[ -L "${BACKUP}/${cli}" ]]; then
                cp -Pf "${BACKUP}/${cli}" "/usr/local/bin/${cli}"
            elif [[ "$(readlink "/usr/local/bin/${cli}")" == "${REPO_ROOT}/target/release/${cli}" ]]; then
                rm -f "/usr/local/bin/${cli}"
            fi
        done
        if [[ "${MENU_WAS_LOADED}" -eq 1 && -f "${PLIST}" ]]; then
            run_user launchctl load -w "${PLIST}"
        fi
        echo "Installation failed; previous app/CLI restored where present. Retained diagnostics: ${BACKUP}; runtime: ${RUNTIME}" >&2
    fi
    exit "${status}"
}
trap rollback_app EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
stop_menu

echo "Copying Bad Apple.app to /Applications..."
if [[ -e "${APP_TARGET}" ]]; then
    mv "${APP_TARGET}" "${BACKUP}/Bad Apple.app"
    APP_BACKED_UP=1
fi
APP_REPLACED=1
mv "${BACKUP}/incoming.app" "${APP_TARGET}"

echo "Stripping quarantine flag..."
xattr -dr com.apple.quarantine "${APP_TARGET}" 2>/dev/null || true

# Link the native CLI and fetch helper into a standard PATH directory so
# `badapple --doctor` and `badapple-fetch` work from a fresh Terminal.
echo "Linking CLI into /usr/local/bin..."
install -d /usr/local/bin
ln -sf "${REPO_ROOT}/target/release/badapple" /usr/local/bin/badapple
ln -sf "${REPO_ROOT}/target/release/badapple-fetch" /usr/local/bin/badapple-fetch

echo "Installing platform from ${REPO_ROOT}..."
cd "${REPO_ROOT}"
BADAPPLE_ROOT="${REPO_ROOT}" src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install
trap - EXIT

echo "Runtime installed at ${RUNTIME}; app backup retained at ${BACKUP}."
echo "Bad Apple is installed. Launch it from /Applications."
EOF
chmod +x "${PKG_DIR}/install.sh"

if [[ -n "$(find "${PKG_DIR}" -type f \( -name '*.swift' -o -name '*.rs' \) -print -quit)" ]]; then
    fail "source files found in public release staging directory"
fi
if [[ -n "$(find "${PKG_DIR}/bad_apple/src" -type f ! -name '*.sh' ! -name '*.plist' -print -quit)" ]]; then
    fail "unexpected file in public runtime src directory"
fi
[[ ! -e "${ZIP_PATH}" && ! -L "${ZIP_PATH}" ]] || fail "refusing to replace existing ZIP: ${ZIP_PATH}"
( cd "${BUILD_DIR}" && zip -r -y -9 "${ZIP_PATH}" "Bad_Apple-${VERSION}-unsigned" )

cp -R "${PKG_DIR}" "${COMPAT_DIR}"
( cd "${BUILD_DIR}" && zip -r -y -9 "${COMPAT_ZIP}" "Bad_Apple-${VERSION}-full" )
( cd "${BUILD_DIR}" && shasum -a 256 "$(basename "${ZIP_PATH}")" "$(basename "${COMPAT_ZIP}")" ) > "${CHECKSUMS}"

echo "Packaged: ${ZIP_PATH}"
echo "Legacy updater compatibility package: ${COMPAT_ZIP}"
echo "Checksums: ${CHECKSUMS}"
echo ""
echo "To publish:"
echo "  1. gh release create v${VERSION} --repo savageAZfck/bad-apple-releases ${ZIP_PATH} ${COMPAT_ZIP} ${CHECKSUMS}"
echo "  2. src/platform/apple_desktop/publish_cask.sh   # syncs version+sha and pushes the tap"
