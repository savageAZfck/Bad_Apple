# Install Bad Apple

Bad Apple runs entirely on your Mac. **No Apple Developer ID is required.** The unsigned build ships without Apple notarization, so you strip the Gatekeeper quarantine flag locally instead of paying Apple to scan your binary.

## What you need

- macOS 26.0 or later (macOS 26 beta/Sonoma+ for `FoundationModels`)
- Apple Silicon Mac (M1 or newer)
- Xcode 16+ or Command Line Tools with `swiftc`
- Rust via [rustup](https://rustup.rs)
- Python 3.12 and a persistent virtual environment
- Model weights cached locally (the first download can be several GB)

## One-command build (from source)

```bash
cargo build --release
```

This produces:

- `target/release/badapple` — CLI client
- `target/release/gatekeeper` — front proxy
- `target/release/badapple-identity` — Secure Enclave helper

## Build the unsigned menu-bar app

```bash
src/platform/apple_desktop/build_bad_apple_menu_bar.sh
```

If you do not have an Apple Developer ID, the script warns about missing code signing and produces an unsigned `target/release/Bad Apple.app`. Move it to `/Applications`:

```bash
cp -R "target/release/Bad Apple.app" /Applications/
```

### Open an unsigned app for the first time

Because the app is not signed or notarized, Gatekeeper flags it by default. Do **not** disable Gatekeeper globally. You have two options:

**Option A — strip the quarantine flag (fastest):**

```bash
sudo src/platform/apple_desktop/strip_quarantine.sh
```

Then launch Bad Apple normally. The `--unsigned-install` platform installer does this automatically.

**Option B — right-click Open (manual):**

1. Open **System Settings → Privacy & Security**.
2. Scroll down to **Security**.
3. Next to "Bad Apple" click **Open Anyway**.
4. Launch `/Applications/Bad Apple.app` once with Control-click → **Open**.

After that first approval, launchd and the menu bar agent can start it normally.

## Install the platform daemons

The daemons need root to write to `/var/lib/bad_apple` and `/Library/LaunchDaemons`:

```bash
cargo build --release
osascript -e 'do shell script "cd /Users/savag3/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install" with administrator privileges'
```

If you are running without an Apple Developer ID, use the unsigned mode after copying the unsigned `Bad Apple.app` to `/Applications`:

```bash
osascript -e 'do shell script "cd /Users/savag3/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --unsigned-install" with administrator privileges'
```

The installer creates a rollback snapshot under `/var/lib/bad_apple/install_backups/`. If the health check fails, it restores the previous launchd configuration.

## Install the persistent menu bar agent

```bash
src/platform/apple_desktop/install_menu_bar_agent.sh
```

This makes Bad Apple start at login and restart after a crash.

## First launch

After the platform installer finishes, wait about 45 seconds for the 9B MLX model and embedding model to load. Check the log:

```bash
tail -n 20 /var/log/bad_apple_mlx_server.log
```

Then run a quick test:

```bash
target/release/badapple "What is 2+2?"
```

Open the dashboard at `http://127.0.0.1:8787` in your browser.

## Permissions you will be asked for

Bad Apple needs a few macOS permissions:

- **Accessibility** — to drive UI actions via System Events.
- **Microphone** — for voice input.
- **Speech Recognition** — for local transcription.
- **Screen Recording** — for screen capture / vision tools.

Approve each when prompted. They are handled entirely on-device.

## Need help?

See [SUPPORT.md](SUPPORT.md) and the [AGENTS.md](AGENTS.md) cheat sheet.
