#!/usr/bin/env bash
# Pre-commit hook for Bad Apple
# Prevents unwrap() in non-test Rust code and runs linters.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# 1. Check for .unwrap() in non-test Rust code
# The non-test .unwrap() counter has been removed because it relied on a legacy
# external interpreter. Re-enable this guard with a Rust-based pre-commit check
# (e.g. a clippy lint or a dedicated cargo xtask).
# TODO: Rust-based .unwrap() pre-commit lint.

# 2. Check for TODO/FIXME in production code (exclude XXXXXX in mktemp templates)
TODO_COUNT=$(grep -rn 'TODO\|FIXME\|HACK' src/ 2>/dev/null | wc -l | tr -d ' ' || true)
if [ "${TODO_COUNT}" -gt 0 ]; then
    echo "WARNING: Found ${TODO_COUNT} TODO/FIXME markers in source." >&2
fi
# TODOs are warnings, not errors — continue

# 3. Check for world-writable file permissions in code
WORLD_WRITABLE=$(grep -rn '0o666\|0o777\|chmod.*777\|chmod.*666' src/ *.sh 2>/dev/null | grep -v test | grep -v '#\|//' | wc -l | tr -d ' ' || true)
if [ "${WORLD_WRITABLE}" -gt 0 ]; then
    echo "ERROR: Found ${WORLD_WRITABLE} world-writable permission references." >&2
    exit 1
fi

echo "Pre-commit checks passed."
exit 0
