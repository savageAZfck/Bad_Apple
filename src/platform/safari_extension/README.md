# Bad Apple Safari Companion

A Safari web extension scaffold for Bad Apple. It lets the assistant see the
current web page and answer questions about it, all routed through the local
SLICKS Unix socket.

## Layout

- `BadAppleCompanion/Resources/` — Safari web extension bundle (manifest, content
  script, popup, background worker).
- `NativeHost/` — `com.badapple.companion.json` manifest and `badapple_companion_host.py`
  native messaging bridge.

## Install (macOS)

1. Open `src/platform/apple_bridge/BadAppleBridge.xcodeproj` or create a new
   Safari Extension App target that embeds `BadAppleCompanion`.
2. Copy `NativeHost/com.badapple.companion.json` to
   `~/Library/Application Support/com.apple.Safari/NativeMessagingHosts/`.
3. Symlink `NativeHost/badapple_companion_host.py` into the same directory.
4. Enable the extension in Safari → Settings → Extensions.

## Security

The content script only runs on user interaction. The native host talks only to
`/var/run/badapple/substrate.sock`. No data leaves the device.
