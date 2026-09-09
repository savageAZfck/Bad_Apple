# homebrew-bad-apple

A Homebrew tap for Bad Apple, a sovereign, local AI operating-system layer for macOS.

## Install

```bash
brew tap savageAZfck/bad-apple https://github.com/savageAZfck/homebrew-bad-apple
brew install --cask bad-apple
```

This installs `Bad Apple.app` into `/Applications`, removes its Gatekeeper
quarantine flag, and runs the platform installer to set up the system
daemons. The 7B model is downloaded on first use.

## Notes

- The cask is unsigned; it uses `xattr -dr com.apple.quarantine` and the
  `--unsigned-install` platform installer flag.
- A signed release can be produced with
  `src/platform/apple_desktop/package_signed_release.sh` in the main repo.
- The platform code is copied to `~/.bad_apple/versions/<version>/Bad_Apple-<version>-unsigned`.
- The 7B model uses about 4 GB of unified memory at peak; **8 GB** total is the
  practical minimum and **16 GB** is the comfortable recommendation. Keep at
  least **40 GB** free for the OS, model cache, and swap.

## Uninstall

```bash
brew uninstall --cask bad-apple
brew untap savageAZfck/bad-apple
```
