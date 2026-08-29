# Identity helper — Secure Enclave daemon constraints

`target/release/badapple-identity` is built from
`src/platform/apple_bridge/BadAppleIdentity.swift` (build command in
`src/platform/apple_bridge/build_apple_bridge.sh:67-77`). It is a one-shot
Swift helper that links `CryptoKit`, `Foundation`, `LocalAuthentication`, and
`Security`.

## What it does

- `ensure` / `public-key` / `sign` load or create a
  `SecureEnclave.P256.Signing.PrivateKey` from
  `~/Library/Application Support/BadApple/identity.sekey`
  (path overridable via `BADAPPLE_IDENTITY_BLOB`).
- The key is created with the default SE initializer and written to disk with
  `Data.WritingOptions.completeFileProtection` (`BadAppleIdentity.swift:44`).
- `biometric-gate` uses `LAContext.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics)`
  (`BadAppleIdentity.swift:53-67`).

## Why it fails from a daemon

1. Secure Enclave key creation / loading requires the per-user
   `com.apple.CoreAuthentication.agent` / `setoken` service, which only exists
   in the user's `Aqua` session. A `LaunchDaemon` (even with `UserName`) runs
   in a `Background` session, so `SecureEnclave` operations fail or return
   "unavailable".
2. `LocalAuthentication` cannot evaluate a biometric policy without a window
   server / GUI session and a TTY, so `biometric-gate` always fails in a
   daemon.
3. The key blob is written with `completeFileProtection`
   (`NSFileProtectionComplete`), which makes it unreadable when the screen is
   locked or from a non-user data-protection context.
4. The file is created `0o600`, so a `root`-running daemon (e.g. gatekeeper)
   cannot read a console user's key even if it can find the path.
5. `status` can only report `unavailable` / `missing`; the non-zero exit is
   surfaced as an `IdentityError` in `badapple_identity.py`.

## Workarounds / long-lived signing agent

- Do not call the helper from a `LaunchDaemon`. Run an agent in the user's
  `Aqua` session (a `LaunchAgent` with `LimitLoadToSessionType` `Aqua`, or a
  child of the menu-bar app) and keep the `SecureEnclave` key loaded there.
- Expose signing over a Unix domain socket the same way `badapple_aqua_helper.py`
  does for Shortcuts.
- Make `badapple_identity.py` prefer `BADAPPLE_IDENTITY_AGENT_SOCKET` and
  fall back to the one-shot binary only when the agent is absent.
- If signing while the screen is locked is required, change the file
  protection to `.completeFileProtectionUntilFirstUserAuthentication`
  (or keychain `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`) instead of
  `.completeFileProtection`.
- If per-signature user approval is desired, add a `SecAccessControl` with
  `.userPresence` / `.biometryCurrentSet` — but this cannot be unattended.
