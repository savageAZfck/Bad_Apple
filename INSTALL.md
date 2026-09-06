# Install Bad Apple

Bad Apple runs entirely on your Mac. **No Apple Developer ID is required.** The unsigned build ships without Apple notarization, so you strip the Gatekeeper quarantine flag locally instead of paying Apple to scan your binary.

## What you need

- macOS 26.0 or later (macOS 26 beta/Sonoma+ for `FoundationModels`)
- Apple Silicon Mac (M1 or newer)
- Xcode 16+ or Command Line Tools with `swiftc`
- Rust via [rustup](https://rustup.rs)
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
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install" with administrator privileges'
```

If you are running without an Apple Developer ID, use the unsigned mode after copying the unsigned `Bad Apple.app` to `/Applications`:

```bash
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --unsigned-install" with administrator privileges'
```

The installer creates a rollback snapshot under `/var/lib/bad_apple/install_backups/`. If the health check fails, it restores the previous launchd configuration.

## Install the persistent menu bar agent

```bash
src/platform/apple_desktop/install_menu_bar_agent.sh
```

This makes Bad Apple start at login and restart after a crash.

## First launch

After the platform installer finishes, wait about 45 seconds for the 7B MLX model and embedding model to load. Check the log:

```bash
tail -n 20 /var/log/bad_apple_mlx_server.log
```

Then run a quick test:

```bash
target/release/badapple "What is 2+2?"
```

Open the dashboard at `http://127.0.0.1:8787` in your browser.

## One-command consumer install (prebuilt release)

Download `Bad_Apple-<version>-full-unsigned.zip` and run the bundled installer as root:

```bash
unzip Bad_Apple-<version>-full-unsigned.zip
cd Bad_Apple-<version>-full
sudo ./install.sh
```

This:

1. Copies `Bad Apple.app` into `/Applications`.
2. Strips the Gatekeeper quarantine flag.
3. Installs the `badapple`, `gatekeeper`, `badapple-identity`, and `badapple-tts` native binaries.
4. Renders the platform LaunchDaemon plists for your user and install path.
5. Loads the gatekeeper, MLX, and supervisor daemons and waits for readiness.
6. Installs the menu-bar login agent.

## Using larger models

Bad Apple can load any MLX-compatible model. The default is a 7B Qwen 2.5 Coder 4-bit; the 9B Qwen 3.5 is switchable. For 32B models, reduce the KV-cache budget to avoid unified-memory pressure:

```bash
# Ask Bad Apple to shrink the KV cache before switching
badapple "set max kv size to 1024"
badapple "use model mlx-community/Qwen3.5-32B-MLX-4bit"
```

Or change `BADAPPLE_MAX_KV_SIZE` in `src/platform/apple_bridge/com.badapple.mlx.plist` and reload the daemon.

For 70B MoE models such as DeepSeek V3, 64GB+ of unified memory is recommended. Run `badapple "recommend models"` for a curated list.

## Updating

The menu bar app has a **Check for Updates** item that downloads the latest unsigned release from GitHub and replaces `/Applications/Bad Apple.app`.

You can also update manually from a new release zip:

```bash
unzip Bad_Apple-<version>-unsigned.zip
rm -rf /Applications/Bad\ Apple.app
cp -R "Bad Apple.app" /Applications/
sudo "Bad Apple.app/Contents/Resources/strip_quarantine.sh"
sudo "Bad Apple.app/Contents/Resources/update_bad_apple.sh"
```

Or, from an existing install, run the bundled updater directly:

```bash
sudo /Applications/Bad\ Apple.app/Contents/Resources/update_bad_apple.sh
```

Note: the app updater only replaces the menu-bar app. To also update the platform daemons, re-run the platform installer from the matching release.

## Permissions you will be asked for

Bad Apple needs a few macOS permissions:

- **Accessibility** — to drive UI actions via System Events.
- **Microphone** — for voice input.
- **Speech Recognition** — for local transcription.
- **Screen Recording** — for screen capture / vision tools.

Approve each when prompted. They are handled entirely on-device.

## Need help?

See [SUPPORT.md](SUPPORT.md) and the [AGENTS.md](AGENTS.md) cheat sheet.
