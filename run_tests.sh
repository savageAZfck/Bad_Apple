#!/bin/bash
# CI-style test runner for Bad Apple.
# Builds the Rust CLI and menu bar, syntax-checks Python, and runs available tests.
# The smoke test requires a running daemon; it is skipped if the MLX socket is absent.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
SMOKE_SOCKET="/var/run/badapple/substrate_mlx.sock"

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo build --release"
cargo build --release

echo "==> build menu bar app"
src/platform/apple_desktop/build_bad_apple_menu_bar.sh

echo "==> Python syntax check"
$PYTHON -m py_compile badapple_*.py

echo "==> runtime unit tests"
$PYTHON -m unittest -v tests.test_badapple_runtime

echo "==> smoke tests"
if [[ -S "$SMOKE_SOCKET" ]]; then
    $PYTHON tests/test_smoke.py
else
    echo "Smoke test skipped: no daemon at $SMOKE_SOCKET"
    echo "Start the daemon and re-run to exercise the live path."
fi

echo "==> all checks passed"
