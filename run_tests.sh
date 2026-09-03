#!/bin/bash
# CI-style test runner for Bad Apple.
# Builds the Rust CLI and menu bar and runs available tests.
# The smoke test requires a running daemon; it is skipped if the MLX socket is absent.
set -euo pipefail
cd "$(dirname "$0")"

SMOKE_SOCKET="/var/run/badapple/substrate_mlx.sock"

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo build --release"
cargo build --release

echo "==> build menu bar app"
src/platform/apple_desktop/build_bad_apple_menu_bar.sh

echo "==> cargo test --release"
cargo test --release

echo "==> smoke test"
if [[ -S "$SMOKE_SOCKET" ]]; then
    target/release/badapple -n 8 "Reply only: ready"
else
    echo "Smoke test skipped: no daemon at $SMOKE_SOCKET"
    echo "Start the daemon and re-run to exercise the live path."
fi

echo "==> all checks passed"
