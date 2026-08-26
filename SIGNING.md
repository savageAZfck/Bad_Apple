# Code Signing and Notarization

Bad Apple can be released as an unsigned zip (the default) or as a signed,
notarized macOS application. This document explains both paths.

## Unsigned release (default)

No Apple Developer ID is required. The unsigned zip includes a quarantine
stripper and a platform installer that removes the Gatekeeper quarantine flag.

```bash
src/platform/apple_desktop/package_full_release.sh
```

This produces:

```
target/release/Bad_Apple-<version>-full-unsigned.zip
```

Users extract it and run:

```bash
sudo ./install.sh
```

## Signed and notarized release

To ship a consumer build that does not trigger Gatekeeper on modern macOS, you
need an Apple Developer ID Application certificate and notarization credentials.

### Prerequisites

1. Apple Developer ID Application certificate installed in a keychain.
2. An app-specific password from [appleid.apple.com](https://appleid.apple.com).
3. Your 10-character Apple Team ID.

### Build

```bash
export CODESIGN_ID="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="you@example.com"
export APPLE_TEAM_ID="TEAMID"
export APPLE_APP_PASSWORD="abcd-1234-abcd-1234"

src/platform/apple_desktop/package_signed_release.sh
```

This produces:

```
target/release/Bad_Apple-<version>-full-signed.zip
```

If notarization credentials are omitted, the script signs the bundle but prints
a warning and skips notarization.

### Local self-signed testing

For local development and TCC persistence, use the dev certificate path:

```bash
src/platform/apple_desktop/create_dev_signing_cert.sh
src/platform/apple_desktop/sign_bad_apple.sh
```

This creates a keychain-local certificate. It is not suitable for distribution,
but it lets Accessibility/Automation permissions survive rebuilds.
