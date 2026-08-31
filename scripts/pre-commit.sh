#!/usr/bin/env bash
# Pre-commit hook for Bad Apple
# Prevents unwrap() in non-test Rust code and runs linters.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# 1. Check for unwrap() in non-test Rust code
# Use a Python script that properly excludes test modules by tracking
# #[cfg(test)] markers and mod tests blocks per-file
UNWRAP_COUNT=$(python3 -c "
import subprocess
result = subprocess.run(['grep', '-rn', '.unwrap()', 'src/'], capture_output=True, text=True)

# Build a per-file map of test module start lines
import re
test_ranges = {}  # file -> list of (start_line, end_line_or_None)
for fpath in set(l.split(':')[0] for l in result.stdout.splitlines()):
    try:
        with open(fpath) as f:
            lines = f.readlines()
    except (OSError, FileNotFoundError):
        continue
    in_test_attr = False
    test_start = None
    for i, line in enumerate(lines, 1):
        if '#[cfg(test)]' in line:
            in_test_attr = True
            continue
        if in_test_attr and 'mod tests' in line:
            test_start = i
            in_test_attr = False
            # Find the end of the test module by brace counting
            brace_count = line.count('{') - line.count('}')
            for j in range(i, len(lines)):
                brace_count += lines[j].count('{') - lines[j].count('}')
                if brace_count <= 0:
                    test_ranges.setdefault(fpath, []).append((test_start, j + 1))
                    break
            else:
                test_ranges.setdefault(fpath, []).append((test_start, len(lines) + 1))
            test_start = None
        else:
            in_test_attr = False

count = 0
for line in result.stdout.splitlines():
    parts = line.split(':')
    if len(parts) < 2:
        continue
    fpath = parts[0]
    try:
        lineno = int(parts[1])
    except ValueError:
        continue
    # Check if this line is inside a test module
    in_test_mod = False
    for start, end in test_ranges.get(fpath, []):
        if start <= lineno <= end:
            in_test_mod = True
            break
    if in_test_mod:
        continue
    # Skip lines inside raw string literals (code templates)
    # Check if this line is within a r##"..."## block by reading the file
    try:
        with open(fpath) as f:
            file_lines = f.readlines()
        in_raw = False
        for idx, fl in enumerate(file_lines, 1):
            if 'r##\"' in fl:
                in_raw = True
            if in_raw and '\"##' in fl:
                in_raw = False
                continue
            if in_raw and idx == lineno:
                count -= 1  # This unwrap is inside a raw string, not real code
                break
    except (OSError, FileNotFoundError):
        pass
    count += 1
print(count)
")

if [ "${UNWRAP_COUNT}" -gt 0 ]; then
    echo "ERROR: Found ${UNWRAP_COUNT} unwrap() calls in non-test Rust code." >&2
    echo "Use ? with .context() or unwrap_or_default() instead." >&2
    grep -rn '\.unwrap()' src/ 2>/dev/null | grep -v 'r#' | head -10 >&2
    exit 1
fi

# 2. Check for TODO/FIXME in production code (exclude XXXXXX in mktemp templates)
TODO_COUNT=$(grep -rn 'TODO\|FIXME\|HACK' src/ *.py 2>/dev/null | wc -l | tr -d ' ' || true)
if [ "${TODO_COUNT}" -gt 0 ]; then
    echo "WARNING: Found ${TODO_COUNT} TODO/FIXME markers in source." >&2
fi
# TODOs are warnings, not errors — continue

# 3. Check for dangerous Python patterns (not model.eval() or _safe_eval)
DANGEROUS_PY=$(grep -rn '\beval(\|exec(' *.py 2>/dev/null | grep -v 'test' | grep -v '#' | grep -v 'model.eval\|\.eval()\|_safe_eval\|mx.eval' | wc -l | tr -d ' ' || true)
if [ "${DANGEROUS_PY}" -gt 0 ]; then
    echo "ERROR: Found ${DANGEROUS_PY} dangerous Python patterns (eval/exec)." >&2
    grep -rn '\beval(\|exec(' *.py 2>/dev/null | grep -v 'test' | grep -v '#' | grep -v 'model.eval\|\.eval()\|_safe_eval\|mx.eval' | head -10 >&2
    exit 1
fi
DANGEROUS_PY2=$(grep -rn 'pickle\.loads\|subprocess.*shell=True' *.py 2>/dev/null | grep -v test | grep -v '#' | wc -l | tr -d ' ' || true)
if [ "${DANGEROUS_PY2}" -gt 0 ]; then
    echo "ERROR: Found ${DANGEROUS_PY2} dangerous Python patterns (pickle/shell=True)." >&2
    exit 1
fi

# 4. Check for world-writable file permissions in code
WORLD_WRITABLE=$(grep -rn '0o666\|0o777\|chmod.*777\|chmod.*666' src/ *.py *.sh 2>/dev/null | grep -v test | grep -v '#\|//' | wc -l | tr -d ' ' || true)
if [ "${WORLD_WRITABLE}" -gt 0 ]; then
    echo "ERROR: Found ${WORLD_WRITABLE} world-writable permission references." >&2
    exit 1
fi

echo "Pre-commit checks passed."
exit 0
