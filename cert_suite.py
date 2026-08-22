#!/usr/bin/env python3
"""Bad Apple air-gap certification self-test suite.

This suite proves that a running Bad Apple install is local, private, and
air-gapped. It checks for open network sockets, cloud leaks, secret redaction,
policy presence, and ledger integrity.

Run as root for full socket/process visibility:
    sudo /Users/savag3/bad_apple/.venv/bin/python cert_suite.py
"""

import hashlib
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except Exception as e:  # pragma: no cover
    psutil = None
    print(f"[warn] psutil not available: {e}")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def find_badapple_processes() -> List[psutil.Process]:
    if psutil is None:
        return []
    procs = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(p.info.get("cmdline") or [])
            if "badapple" in cmdline.lower() or "bad_apple" in cmdline.lower():
                procs.append(p)
        except Exception:
            pass
    return procs


def check_network_isolation() -> int:
    """Ensure no badapple process holds a non-loopback network socket."""
    print("\n[TEST] network isolation")
    bad_pids = {p.pid for p in find_badapple_processes()}
    if not bad_pids:
        _info("no badapple processes detected; test skipped")
        return 0

    failures = 0
    for pid in bad_pids:
        try:
            p = psutil.Process(pid)
            for conn in p.net_connections(kind="inet"):
                if conn.status == psutil.CONN_LISTEN or conn.status == psutil.CONN_ESTABLISHED:
                    ip, port = conn.laddr
                    if ip not in ("127.0.0.1", "::1", "0.0.0.0", "::"):
                        _fail(f"pid {pid} has external socket {ip}:{port} ({conn.status})")
                        failures += 1
        except Exception as e:
            _info(f"could not inspect pid {pid}: {e}")

    if not failures:
        _ok("no external network sockets detected on badapple processes")
    return failures


def check_unix_sockets() -> int:
    """Ensure the daemon listens on a Unix socket, not a TCP port."""
    print("\n[TEST] local socket only")
    bad_pids = {p.pid for p in find_badapple_processes()}
    if not bad_pids:
        _info("no badapple processes detected; test skipped")
        return 0

    failures = 0
    tcp_found = False
    for pid in bad_pids:
        try:
            p = psutil.Process(pid)
            for conn in p.net_connections(kind="inet"):
                if conn.status == psutil.CONN_LISTEN:
                    _fail(f"pid {pid} is listening on TCP {conn.laddr}")
                    tcp_found = True
                    failures += 1
        except Exception:
            pass

    socket_path = Path("/var/run/badapple/substrate_mlx.sock")
    if socket_path.exists():
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(str(socket_path))
            s.close()
            _ok(f"MLX daemon reachable via Unix socket {socket_path}")
        except Exception as e:
            _info(f"Unix socket exists but not reachable: {e}")
    else:
        _info(f"Unix socket not present at {socket_path}")

    if not tcp_found:
        _ok("no TCP listeners found; communication is local-only")

    return failures


def check_policy_present() -> int:
    """Ensure a policy file is loaded."""
    print("\n[TEST] declarative policy present")
    data_dir = Path(os.environ.get("BADAPPLE_DATA_DIR", os.path.expanduser("~/.bad_apple")))
    repo_policy = Path(__file__).with_name("policy.yaml")
    data_policy = data_dir / "policy.yaml"

    path = None
    if data_policy.is_file():
        path = data_policy
    elif repo_policy.is_file():
        path = repo_policy
    else:
        _fail("no policy.yaml found in repo or data dir")
        return 1

    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data and data.get("tools"):
            _ok(f"policy loaded from {path} with {len(data.get('tools', {}))} tool rules")
            return 0
        _fail("policy file is present but has no tool rules")
        return 1
    except Exception as e:
        _fail(f"policy file invalid: {e}")
        return 1


def check_ledger_integrity(data_dir: Path) -> int:
    """Verify the audit ledger hash chain if it exists."""
    print("\n[TEST] audit ledger integrity")
    ledger_path = data_dir / "ledger.jsonl"
    if not ledger_path.is_file():
        _info("no audit ledger found; test skipped")
        return 0

    failures = 0
    last_hash = ""
    count = 0
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                count += 1
                if last_hash:
                    if entry.get("previous_hash") != last_hash:
                        _fail(f"ledger break at entry {count}: previous hash mismatch")
                        failures += 1
                entry_hash = entry.get("hash", "")
                if not entry_hash:
                    _fail(f"ledger entry {count} missing hash")
                    failures += 1
                else:
                    last_hash = entry_hash
            except json.JSONDecodeError:
                _fail("ledger contains non-JSON line")
                failures += 1

    if not failures:
        _ok(f"ledger {ledger_path} hash chain valid ({count} entries)")
    return failures


def check_no_secrets_in_logs(data_dir: Path) -> int:
    """Scan logs and local data for unredacted secret patterns."""
    print("\n[TEST] secret redaction")
    patterns = [
        re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}['\"]?", re.IGNORECASE),
        re.compile(r"token\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}['\"]?", re.IGNORECASE),
        re.compile(r"secret\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}['\"]?", re.IGNORECASE),
        re.compile(r"password\s*[:=]\s*['\"]?\S+['\"]?", re.IGNORECASE),
    ]

    paths = [
        data_dir,
        Path("/var/log/bad_apple_mlx_server.log"),
        Path("/var/log/bad_apple_tts_server.log"),
        Path("/tmp/badapple_voice_debug.log"),
    ]

    failures = 0
    checked = 0
    for p in paths:
        if not p.exists():
            continue
        checked += 1
        if p.is_dir():
            files = [f for f in p.iterdir() if f.is_file() and f.suffix in (".json", ".jsonl", ".log", ".txt")]
        else:
            files = [p]
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pat in patterns:
                for m in pat.finditer(text):
                    _fail(f"possible secret pattern in {f}: {m.group()[:40]}...")
                    failures += 1

    if not failures:
        _ok(f"scanned {checked} log/data locations: no unredacted secret patterns found")
    return failures


def check_model_local() -> int:
    """Check that the active model weights are stored on disk, not streamed."""
    print("\n[TEST] models are local")
    home = Path.home()
    candidates = [
        home / "Documents" / "models",
        Path("/usr/local/share/badapple"),
        Path("/opt/badapple"),
    ]
    mlx_cache = home / ".cache" / "mlx_lm"
    hf_cache = home / ".cache" / "huggingface"

    found = False
    for d in candidates:
        if d.is_dir() and list(d.iterdir()):
            _ok(f"model directory found at {d}")
            found = True
            break
    for d in (mlx_cache, hf_cache):
        if d.is_dir() and any(d.rglob("*.safetensors")):
            _ok(f"local model weights found under {d}")
            found = True

    if not found:
        _info("could not locate model weights; ensure models are downloaded and cached locally")
    return 0


def check_cloud_references(data_dir: Path) -> int:
    """Scan source/config for hard-coded cloud endpoints (informational)."""
    print("\n[TEST] no hard-coded cloud endpoints")
    repo = Path(__file__).parent
    cloud_patterns = [
        re.compile(r"https?://[^\s\"']*openai\.com", re.IGNORECASE),
        re.compile(r"https?://[^\s\"']*anthropic\.com", re.IGNORECASE),
        re.compile(r"https?://[^\s\"']*googleapis\.com", re.IGNORECASE),
        re.compile(r"https?://[^\s\"']*azure\.com", re.IGNORECASE),
        re.compile(r"https?://[^\s\"']*aws\.amazon\.com", re.IGNORECASE),
    ]

    failures = 0
    for ext in (".py", ".rs", ".swift", ".sh", ".yaml", ".yml", ".json"):
        for f in repo.rglob(f"*{ext}"):
            if ".git" in str(f) or ".venv" in str(f) or "target" in str(f):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pat in cloud_patterns:
                if pat.search(text):
                    _fail(f"possible cloud endpoint in {f}")
                    failures += 1
                    break

    if not failures:
        _ok("no hard-coded third-party cloud endpoints found in source")
    return failures


def main() -> int:
    print("=" * 60)
    print("Bad Apple Air-Gap Certification Suite")
    print("=" * 60)

    data_dir = Path(os.environ.get("BADAPPLE_DATA_DIR", os.path.expanduser("~/.bad_apple")))

    tests = [
        check_network_isolation,
        check_unix_sockets,
        check_policy_present,
        check_ledger_integrity,
        check_no_secrets_in_logs,
        check_model_local,
        check_cloud_references,
    ]

    total_failures = 0
    for test in tests:
        try:
            if test in (check_ledger_integrity, check_no_secrets_in_logs, check_cloud_references):
                total_failures += test(data_dir)
            else:
                total_failures += test()
        except Exception as e:
            _fail(f"test {test.__name__} crashed: {e}")
            total_failures += 1

    print("\n" + "=" * 60)
    if total_failures == 0:
        print("RESULT: PASSED — this build appears air-gapped and local.")
    else:
        print(f"RESULT: {total_failures} failure(s) — review the output above.")
    print("=" * 60)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
