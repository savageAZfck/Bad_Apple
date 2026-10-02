#!/usr/bin/env bash
set -euo pipefail

# gen_sbom.sh — emit a CycloneDX 1.5 SBOM for a Bad Apple release.
#
# Components come from two sources:
#   - Cargo.lock: every resolved crate with its registry sha256 (the same
#     content hash crates.io serves), emitted as a library component with a
#     pkg:cargo purl.
#   - The staged release payloads: every binary, dylib, metallib, and the
#     .app bundle zip(s), hashed at package time so the SBOM attests the
#     bytes that actually ship.
#
# Usage: gen_sbom.sh <version> <cargo_lock> <payload_dir> [extra_files...]
#        Writes <payload_dir>/sbom.json (or the path in $3's parent when
#        payload_dir is a file list dir).

VERSION="${1:?version required}"
LOCK="${2:?Cargo.lock path required}"
OUT="${3:?output path required}"
shift 3

python3 - "$VERSION" "$LOCK" "$OUT" "$@" <<'PY'
import hashlib, json, os, re, sys

version, lock_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
payload_files = sys.argv[4:]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# Cargo.lock [[package]] blocks.
components = []
name = ver = checksum = None
with open(lock_path) as f:
    for line in f:
        line = line.strip()
        m = re.match(r'name = "(.+)"', line)
        if m and name is None:
            name = m.group(1); continue
        m = re.match(r'version = "(.+)"', line)
        if m and ver is None:
            ver = m.group(1); continue
        m = re.match(r'checksum = "([0-9a-f]{64})"', line)
        if m:
            checksum = m.group(1); continue
        if line == "[[package]]":
            name = ver = checksum = None
        if name and ver:
            comp = {
                "type": "library",
                "name": name,
                "version": ver,
                "purl": f"pkg:cargo/{name}@{ver}",
                "scope": "required",
            }
            if checksum:
                comp["hashes"] = [{"alg": "SHA-256", "content": checksum}]
            components.append(comp)
            name = ver = checksum = None

# Shipped payloads — the bytes a verifier can hash locally.
for path in payload_files:
    if not os.path.isfile(path):
        continue
    components.append({
        "type": "file",
        "name": os.path.basename(path),
        "hashes": [{"alg": "SHA-256", "content": sha256_file(path)}],
    })

bom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.5",
    "version": 1,
    "metadata": {
        "component": {
            "type": "application",
            "name": "Bad Apple",
            "version": version,
            "purl": "pkg:github/savageAZfck/Bad_Apple@" + version,
        }
    },
    "components": sorted(components, key=lambda c: (c["type"], c["name"])),
}
with open(out_path, "w") as f:
    json.dump(bom, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"sbom: {out_path} ({len(components)} components)")
PY
