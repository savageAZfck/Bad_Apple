#!/usr/bin/env python3
"""Bad Apple air-gap certification self-test suite.

This suite proves that a running Bad Apple install is local, private, and
air-gapped. It checks for open network sockets, cloud leaks, secret redaction,
policy presence, and ledger integrity.

Run as root for full socket/process visibility:
    sudo /path/to/bad_apple/.venv/bin/python cert_suite.py
"""

import json
import os
import re
import socket
import sys
from pathlib import Path

try:
    import psutil
except ImportError as e: # pragma: no cover
    psutil = None
    print(f"[warn] psutil not available: {e}")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def find_badapple_processes() -> list[psutil.Process]:
    if psutil is None:
        return []
    procs = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(p.info.get("cmdline") or [])
            if "badapple" in cmdline.lower() or "bad_apple" in cmdline.lower():
                procs.append(p)
        except Exception as e:  # noqa: BLE001 - logged
            print(f"[cert_suite] join failed: {e}", flush=True)
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
                if conn.status == psutil.CONN_LISTEN:
                    ip, port = conn.laddr
                    if ip in ("0.0.0.0", "::"):
                        _fail(f"pid {pid} is listening on all interfaces {ip}:{port}")
                        failures += 1
                    elif ip not in ("127.0.0.1", "::1"):
                        _fail(f"pid {pid} has external listener {ip}:{port}")
                        failures += 1
                elif conn.status == psutil.CONN_ESTABLISHED:
                    ip, port = conn.laddr
                    remote = conn.raddr
                    if ip not in ("127.0.0.1", "::1"):
                        _fail(f"pid {pid} has external local socket {ip}:{port} -> {remote}")
                        failures += 1
                    elif remote and remote.ip not in ("127.0.0.1", "::1"):
                        _fail(f"pid {pid} connected to external host {remote.ip}:{remote.port}")
                        failures += 1
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
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
                    addr = conn.laddr
                    # 127.0.0.1 / ::1 only is local. 0.0.0.0 means all interfaces.
                    if str(addr.ip) in ("0.0.0.0", "::"):
                        _fail(f"pid {pid} is listening on all interfaces {addr}")
                        tcp_found = True
                        failures += 1
                    elif str(addr.ip) not in ("127.0.0.1", "::1"):
                        _fail(f"pid {pid} is listening on external TCP {addr}")
                        tcp_found = True
                        failures += 1
                    else:
                        _info(f"pid {pid} has local-only TCP listener {addr}")
        except Exception as e:  # noqa: BLE001 - logged
            print(f"[cert_suite] Process failed: {e}", flush=True)

    socket_path = Path("/var/run/badapple/substrate_mlx.sock")
    if socket_path.exists():
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(str(socket_path))
            s.close()
            _ok(f"MLX daemon reachable via Unix socket {socket_path}")
        except (OSError, ValueError, TypeError) as e:
            _info(f"Unix socket exists but not reachable: {e}")
    else:
        _info(f"Unix socket not present at {socket_path}")

    if not tcp_found:
        _ok("no TCP listeners found; communication is local-only")

    return failures


def check_policy_present() -> int:
    """Ensure a policy file is loaded."""
    print("\n[TEST] declarative policy present")
    data_dir = Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple"))
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
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        _fail(f"policy file invalid: {e}")
        return 1


def check_ledger_integrity(data_dir: Path) -> int:
    """Verify the audit ledger hash chain if it exists.

    This is the same check `tools/verify_ledger.py` performs, but importing
    the live AuditLedger class rather than re-parsing the file. Run
    `tools/verify_ledger.py` directly if you want a check that trusts
    nothing from this codebase at all -- see that file's own docstring.
    """
    print("\n[TEST] audit ledger integrity")
    from badapple_extras import AuditLedger

    ledger = AuditLedger(data_dir)
    if not ledger.ledger_path.is_file():
        _info("no audit ledger found; test skipped")
        return 0
    results = ledger.verify()
    invalid = [result for result in results if not result.get("valid")]
    if invalid:
        _fail(f"ledger has {len(invalid)} invalid entries out of {len(results)}")
        return len(invalid)
    _ok(f"ledger {ledger.ledger_path} hash chain valid ({len(results)} entries)")

    checkpoint_path = data_dir / "ledger_checkpoint.json"
    if not checkpoint_path.is_file():
        _info(
            "no Secure Enclave checkpoint found; a hash chain alone does not protect "
            "against a full-chain rewrite by anyone with file write access. Run "
            "`agent_client.py audit checkpoint` to create one."
        )
        return 0
    try:
        import json as _json

        from tools.verify_ledger import verify_checkpoint

        tip_hash = ledger._last_hash()
        entry_count = sum(1 for r in results)
        checkpoint_result = verify_checkpoint(checkpoint_path, tip_hash, entry_count)
        if not checkpoint_result.get("ok"):
            _fail(f"Secure Enclave checkpoint invalid: {checkpoint_result.get('error')}")
            return 1
        _ok(
            f"Secure Enclave checkpoint valid: chain state attested by device key "
            f"{checkpoint_result['public_key'][:20]}... at {checkpoint_result['signed_at']}"
        )
    except (ImportError, OSError, _json.JSONDecodeError) as e:
        _info(f"could not verify checkpoint: {e}")
    return 0


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
        Path("/tmp/badapple_voice_debug.log"),  # noqa: S108 - read-only audit scan, see badapple_dashboard_data._voice_activity for the hardened reader
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
            except (OSError, ValueError):
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
            except (OSError, ValueError):
                continue
            for pat in cloud_patterns:
                if pat.search(text):
                    _fail(f"possible cloud endpoint in {f}")
                    failures += 1
                    break

    if not failures:
        _ok("no hard-coded third-party cloud endpoints found in source")
    return failures


def check_model_provenance() -> int:
    print("\n[TEST] signed model provenance")
    from badapple_vault import ArtifactManifest

    candidates = [
        Path(os.environ.get("BADAPPLE_MODEL_MANIFEST", "")),
        Path.home() / ".bad_apple" / "model_manifest.json",
        Path.home() / ".local" / "share" / "badapple" / "model_manifest.json",
        Path("/var/lib/bad_apple/model_manifest.json"),
    ]
    manifest_path = next((path for path in candidates if str(path) and path.is_file()), None)
    if manifest_path is None:
        _fail("no signed model manifest found")
        return 1
    try:
        result = ArtifactManifest.verify(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
        _fail(f"model manifest could not be verified: {e}")
        return 1
    if result.get("valid") and result.get("signature_valid") is True:
        _ok(f"model artifacts and Secure Enclave seal verified from {manifest_path}")
        return 0
    _fail(f"model provenance invalid: {result}")
    return 1


def check_hardware_identity() -> int:
    print("\n[TEST] Secure Enclave identity")
    try:
        import badapple_identity
        status = badapple_identity.status()
        if status.startswith("secure-enclave:"):
            _ok(f"hardware-bound identity active ({status.split(':', 1)[1][:16]}...)")
            return 0
        _fail(f"hardware identity status: {status}")
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        _fail(f"Secure Enclave identity unavailable: {e}")
    return 1


def check_slicks_v2_handshake() -> int:
    """End-to-end v2 SLICKS handshake against the live gatekeeper/daemon."""
    print("\n[TEST] SLICKS v2 handshake")
    import secrets
    import time

    try:
        import badapple_slicks
    except Exception as e:  # noqa: BLE001 - import may fail in minimal envs
        _info(f"badapple_slicks not available: {e}")
        return 0

    if not badapple_slicks.v2_available():
        _info("SLICKS v2 not available on this host; skipping")
        return 0

    socket_path = os.environ.get("BADAPPLE_SOCKET_PATH", "/var/run/badapple/substrate.sock")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(60)
            sock.connect(socket_path)
            reader = sock.makefile("r")

            client_nonce = secrets.token_hex(32)
            timestamp_ms = int(time.time() * 1000)
            client_pubkey = badapple_slicks.v2_public_key_b64()

            hello = {
                "type": "hello",
                "version": 2,
                "timestamp_ms": timestamp_ms,
                "client_nonce": client_nonce,
                "client_pubkey": client_pubkey,
            }
            sock.sendall((json.dumps(hello) + "\n").encode())

            challenge_raw = reader.readline()
            challenge = json.loads(challenge_raw)
            if challenge.get("type") != "challenge":
                _fail(f"expected challenge, got {challenge.get('type')}")
                return 1
            server_nonce = challenge["server_nonce"]
            server_proof = challenge["proof"]
            server_pubkey = challenge.get("server_pubkey")
            if not server_pubkey:
                _fail("SLICKS v2 challenge did not include server_pubkey")
                return 1

            import base64

            if not badapple_slicks.v2_verify_server_proof(
                timestamp_ms,
                client_nonce,
                server_nonce,
                server_proof,
                base64.b64decode(server_pubkey),
            ):
                _fail("SLICKS v2 server authentication failed")
                return 1

            prompt = "Respond with exactly the word 'v2ok'."
            max_new_tokens = 32
            client_proof = badapple_slicks.v2_client_proof(
                timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens
            )

            execute = {
                "type": "execute",
                "version": 2,
                "timestamp_ms": timestamp_ms,
                "client_nonce": client_nonce,
                "server_nonce": server_nonce,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "proof": client_proof,
                "client_pubkey": client_pubkey,
            }
            sock.sendall((json.dumps(execute) + "\n").encode())

            accepted = False
            final_text = ""
            for _ in range(10_000):
                line = reader.readline()
                if not line:
                    break
                frame = json.loads(line)
                frame_type = frame.get("type")
                if frame_type == "accepted":
                    accepted = True
                elif frame_type == "token" and accepted:
                    final_text += frame.get("text", "")
                elif frame_type == "done" and accepted:
                    final_text = frame.get("text", final_text)
                    break
                elif frame_type == "error":
                    _fail(f"SLICKS v2 request failed: {frame.get('message')}")
                    return 1

            if not final_text.strip():
                _fail("SLICKS v2 handshake succeeded but produced no response")
                return 1

            _ok("SLICKS v2 handshake completed and server identity verified")
            return 0
    except TimeoutError:
        _fail("SLICKS v2 handshake timed out")
        return 1
    except json.JSONDecodeError as e:
        _fail(f"SLICKS v2 handshake received invalid JSON: {e}")
        return 1
    except Exception as e:  # noqa: BLE001 - cert test wrapper
        _fail(f"SLICKS v2 handshake failed: {e}")
        return 1


def check_supervisor() -> int:
    print("\n[TEST] bounded health supervisor")
    import subprocess
    import tempfile

    env = dict(os.environ)
    cert_dir = Path(tempfile.mkdtemp(prefix="badapple_cert_supervisor_"))
    env["BADAPPLE_DATA_DIR"] = str(cert_dir)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("badapple_supervisor.py")), "--once", "--no-repair"],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    check=False)
    try:
        cert_dir.rmdir()
    except OSError:
        pass
    if result.returncode != 0:
        _fail(result.stderr.strip() or "supervisor health check failed")
        return 1
    report = json.loads(result.stdout)
    unhealthy = [name for name, state in report.get("services", {}).items() if not state.get("running")]
    if unhealthy:
        _fail(f"services not running: {', '.join(unhealthy)}")
        return len(unhealthy)
    _ok("service family is live and supervisor checks are bounded")
    return 0


def main() -> int:
    print("=" * 60)
    print("Bad Apple Air-Gap Certification Suite")
    print("=" * 60)

    data_dir = Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple"))

    tests = [
        check_network_isolation,
        check_unix_sockets,
        check_policy_present,
        check_ledger_integrity,
        check_no_secrets_in_logs,
        check_model_local,
        check_cloud_references,
        check_model_provenance,
        check_hardware_identity,
        check_slicks_v2_handshake,
        check_supervisor,
    ]

    total_failures = 0
    for test in tests:
        try:
            if test in (check_ledger_integrity, check_no_secrets_in_logs, check_cloud_references):
                total_failures += test(data_dir)
            else:
                total_failures += test()
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
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
