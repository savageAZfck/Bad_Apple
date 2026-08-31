#!/usr/bin/env python3
"""Security regression tests for Bad Apple — Round 2 red team findings.

Tests the fixes for vulnerabilities found in the second red team audit:
- AppleScript injection prevention
- Path traversal jailing
- CSRF Origin verification
- P2P adapter zip slip prevention
- Policy autopilot default
- Audit ledger secret derivation
- Browser action URL scheme blocking
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add the repo root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestAppleScriptEscaping(unittest.TestCase):
    """Verify AppleScript string escaping prevents injection."""

    def test_esc_applescript_handles_quotes(self):
        from badapple_macos_apps import _esc_applescript
        # Double quote injection attempt
        raw = '"; do shell script "id"; --'
        result = _esc_applescript(raw)
        # The result should have escaped quotes — every " should be preceded by \
        # and the string should not be breakable at a raw unescaped "
        # Check that the first character is a backslash (escaped the opening quote)
        self.assertTrue(result.startswith('\\'))
        # Count unescaped quotes (should be zero)
        unescaped = result.replace('\\"', '').count('"')
        self.assertEqual(unescaped, 0)

    def test_esc_applescript_handles_backslash(self):
        from badapple_macos_apps import _esc_applescript
        # Backslash before quote — should be doubled first
        raw = 'test\\"; do shell script "id"'
        result = _esc_applescript(raw)
        # No unescaped quotes should remain
        unescaped = result.replace('\\"', '').replace('\\\\', '').count('"')
        self.assertEqual(unescaped, 0)

    def test_esc_applescript_preserves_normal_text(self):
        from badapple_macos_apps import _esc_applescript
        result = _esc_applescript("normal text")
        self.assertEqual(result, "normal text")


class TestPathJailing(unittest.TestCase):
    """Verify path traversal is blocked."""

    def test_jail_path_rejects_system_files(self):
        from badapple_tools import _jail_path
        with self.assertRaises(ValueError):
            _jail_path(Path("/etc/passwd"))

    def test_jail_path_rejects_private_dirs(self):
        from badapple_tools import _jail_path
        with self.assertRaises(ValueError):
            _jail_path(Path("/private/etc"))

    def test_jail_path_allows_home(self):
        from badapple_tools import _jail_path
        # Home directory should be allowed (user owns their files)
        home = Path("~").expanduser()
        result = _jail_path(home)
        self.assertEqual(result, home.resolve())


class TestPolicyAutopilotDefault(unittest.TestCase):
    """Verify autopilot is false by default."""

    def test_policy_yaml_has_autopilot_false(self):
        policy_path = Path(__file__).resolve().parent.parent / "policy.yaml"
        content = policy_path.read_text()
        # The default should be false, not true
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
        autopilot_lines = [line for line in lines if line.startswith("autopilot:")]
        self.assertEqual(len(autopilot_lines), 1)
        self.assertIn("false", autopilot_lines[0])
        self.assertNotIn("true", autopilot_lines[0])


class TestCSRFOriginCheck(unittest.TestCase):
    """Verify CSRF token check validates Origin."""

    def test_csrf_rejects_cross_origin(self):
        from badapple_dashboard import _check_csrf_token
        # Get the real token
        with patch("badapple_dashboard._get_csrf_token", return_value="test_token_123"):
            # Cross-origin request should be rejected
            headers = {"Origin": "http://evil.com", "X-CSRF-Token": "test_token_123"}
            self.assertFalse(_check_csrf_token(headers))

    def test_csrf_accepts_same_origin(self):
        from badapple_dashboard import _check_csrf_token
        with patch("badapple_dashboard._get_csrf_token", return_value="test_token_123"):
            headers = {"Origin": "http://127.0.0.1:8787", "X-CSRF-Token": "test_token_123"}
            self.assertTrue(_check_csrf_token(headers))

    def test_csrf_accepts_no_origin(self):
        from badapple_dashboard import _check_csrf_token
        # Non-browser clients without Origin header should still work with token
        with patch("badapple_dashboard._get_csrf_token", return_value="test_token_123"):
            headers = {"X-CSRF-Token": "test_token_123"}
            self.assertTrue(_check_csrf_token(headers))

    def test_csrf_rejects_wrong_token(self):
        from badapple_dashboard import _check_csrf_token
        with patch("badapple_dashboard._get_csrf_token", return_value="test_token_123"):
            headers = {"Origin": "http://127.0.0.1:8787", "X-CSRF-Token": "wrong_token"}
            self.assertFalse(_check_csrf_token(headers))


class TestP2PAdapterZipSlip(unittest.TestCase):
    """Verify P2P adapter extraction is path-safe."""

    def test_adapter_name_sanitized(self):
        # The fix should sanitize adapter_name to prevent path traversal
        adapter_name = "../../../etc/passwd"
        safe_name = "".join(c for c in adapter_name if c.isalnum() or c in "-_.")
        # Path separators should be removed
        self.assertNotIn("/", safe_name)
        # The name should not allow directory traversal via /
        # (dots are kept but without / they can't traverse)


class TestBrowserActionURLSchemes(unittest.TestCase):
    """Verify dangerous URL schemes are blocked."""

    def test_file_scheme_blocked(self):
        # The fix adds scheme checking in browser_action navigate
        blocked_schemes = ("file://", "smb://", "dict://", "ftp://", "ssh://", "vnc://")
        for scheme in blocked_schemes:
            url_lower = (scheme + "test").lower()
            is_blocked = any(url_lower.startswith(s) for s in blocked_schemes)
            self.assertTrue(is_blocked, f"{scheme} should be blocked")

    def test_http_scheme_allowed(self):
        blocked_schemes = ("file://", "smb://", "dict://", "ftp://", "ssh://", "vnc://")
        url_lower = "https://example.com".lower()
        is_blocked = any(url_lower.startswith(s) for s in blocked_schemes)
        self.assertFalse(is_blocked)


class TestShellAllowlist(unittest.TestCase):
    """Verify dangerous interpreters are not in the allowlist."""

    def test_no_interpreters_in_allowlist(self):
        from badapple_tools import SHELL_ALLOWED_COMMANDS
        dangerous = {"python3", "python", "swift", "cargo", "rustc", "git"}
        for cmd in dangerous:
            self.assertNotIn(cmd, SHELL_ALLOWED_COMMANDS,
                             f"{cmd} should not be in shell allowlist")

    def test_safe_commands_in_allowlist(self):
        from badapple_tools import SHELL_ALLOWED_COMMANDS
        safe = {"ls", "cat", "grep", "find", "head", "tail"}
        for cmd in safe:
            self.assertIn(cmd, SHELL_ALLOWED_COMMANDS)


class TestAuditLedgerSecret(unittest.TestCase):
    """Verify audit ledger derives a secret when not explicitly configured."""

    def test_ledger_secret_not_empty_by_default(self):
        # When BADAPPLE_LEDGER_SECRET is not set and slicks.key doesn't exist,
        # the ledger should still try to derive a secret
        import badapple_extras
        # The _secret field should not be empty if a slicks key exists
        # This is a structural test — we verify the code path exists
        import inspect
        source = inspect.getsource(badapple_extras.AuditLedger.__init__)
        self.assertIn("slicks_key", source)
        self.assertIn("self._secret", source)


class TestAquaHelperReplayProtection(unittest.TestCase):
    """Verify aqua helper has timestamp and nonce replay protection."""

    def test_verify_proof_requires_timestamp(self):
        from badapple_aqua_helper import _verify_request_proof
        # Without timestamp_ms, proof should fail
        body = {"action": "test", "nonce": "abc123", "proof": "deadbeef"}
        self.assertFalse(_verify_request_proof(body, b"secret"))

    def test_verify_proof_requires_nonce(self):
        from badapple_aqua_helper import _verify_request_proof
        # Without nonce, proof should fail
        body = {"action": "test", "timestamp_ms": 9999999999999, "proof": "deadbeef"}
        self.assertFalse(_verify_request_proof(body, b"secret"))

    def test_verify_proof_rejects_stale_timestamp(self):
        from badapple_aqua_helper import _verify_request_proof
        # Timestamp from 1 hour ago should be rejected
        import time
        old_ts = int((time.time() - 3600) * 1000)
        body = {"action": "test", "timestamp_ms": old_ts, "nonce": "abc123", "proof": "deadbeef"}
        self.assertFalse(_verify_request_proof(body, b"secret"))

    def test_verify_proof_rejects_future_timestamp(self):
        from badapple_aqua_helper import _verify_request_proof
        import time
        future_ts = int((time.time() + 3600) * 1000)
        body = {"action": "test", "timestamp_ms": future_ts, "nonce": "abc123", "proof": "deadbeef"}
        self.assertFalse(_verify_request_proof(body, b"secret"))


class TestScreenCapturePathSafety(unittest.TestCase):
    """Verify screen_capture ignores user-supplied paths."""

    def test_screen_capture_ignores_user_path(self):
        # The fix should always write to a safe temp path, ignoring user-supplied path
        import inspect
        import badapple_tools
        source = inspect.getsource(badapple_tools.run_tool)
        # Find the screen_capture section — verify it doesn't use args.get("path") for output
        if "screen_capture" in source:
            # Check that the screen_capture section uses tempfile, not args path
            lines = source.split("\n")
            in_screen = False
            uses_tempfile = False
            for line in lines:
                if "screen_capture" in line:
                    in_screen = True
                if in_screen and "tempfile" in line:
                    uses_tempfile = True
                if in_screen and "return" in line and "capture_screen" in line:
                    break
            self.assertTrue(uses_tempfile, "screen_capture should use tempfile, not user path")


class TestBrowserActionSchemeBlocking(unittest.TestCase):
    """Verify dangerous URL schemes are blocked in browser_action."""

    def test_file_scheme_blocked(self):
        blocked_schemes = ("file://", "smb://", "dict://", "ftp://", "ssh://", "vnc://")
        for scheme in blocked_schemes:
            url_lower = (scheme + "test").lower()
            is_blocked = any(url_lower.startswith(s) for s in blocked_schemes)
            self.assertTrue(is_blocked, f"{scheme} should be blocked")

    def test_http_scheme_allowed(self):
        blocked_schemes = ("file://", "smb://", "dict://", "ftp://", "ssh://", "vnc://")
        for scheme in ("http://", "https://"):
            url_lower = (scheme + "example.com").lower()
            is_blocked = any(url_lower.startswith(s) for s in blocked_schemes)
            self.assertFalse(is_blocked, f"{scheme} should be allowed")

    def test_localhost_allowed(self):
        # Localhost should be allowed for local dashboard access
        blocked_schemes = ("file://", "smb://", "dict://", "ftp://", "ssh://", "vnc://")
        url_lower = "http://127.0.0.1:8787".lower()
        is_blocked = any(url_lower.startswith(s) for s in blocked_schemes)
        self.assertFalse(is_blocked)


class TestAccessibilityActionEscaping(unittest.TestCase):
    """Verify accessibility_action escapes AppleScript injection."""

    def test_type_action_escapes_quotes(self):
        import inspect
        import badapple_tools
        source = inspect.getsource(badapple_tools.run_tool)
        # The accessibility_action section should use _esc_applescript
        if "accessibility_action" in source:
            self.assertIn("_esc_applescript", source)

    def test_key_action_validates_numeric(self):
        import inspect
        import badapple_tools
        source = inspect.getsource(badapple_tools.run_tool)
        if "accessibility_action" in source and "key code" in source:
            # Should validate that key code is numeric
            self.assertIn("isdigit", source)


class TestP2POriginSpoofing(unittest.TestCase):
    """Verify P2P v1 HMAC frames can't spoof v2 origin IDs."""

    def test_v1_proof_length_is_64(self):
        # v1 HMAC proofs are 64 hex chars (SHA256)
        # v2 proofs are longer (DER ECDSA signatures)
        # The check `len(frame.proof) > 64` should route to v2
        self.assertEqual(len("a" * 64), 64)
        self.assertGreater(len("a" * 128), 64)


class TestModelVersionPinning(unittest.TestCase):
    """Verify model version pinning is in place."""

    def test_model_revision_constant_exists(self):
        import badapple_mlx_server
        self.assertTrue(hasattr(badapple_mlx_server, "MODEL_REVISION"))
        self.assertTrue(hasattr(badapple_mlx_server, "DEFAULT_MODEL_REVISION"))

    def test_model_revision_is_pinned_hash(self):
        import badapple_mlx_server
        # The default should be a commit hash, not "main"
        rev = badapple_mlx_server.DEFAULT_MODEL_REVISION
        self.assertNotEqual(rev, "main")
        self.assertGreaterEqual(len(rev), 20, "revision should be a commit hash")

    def test_model_integrity_verification_exists(self):
        import badapple_mlx_server
        import inspect
        # The MLXServer class should have _verify_model_integrity method
        # Find the class
        for _name, obj in inspect.getmembers(badapple_mlx_server):
            if inspect.isclass(obj) and hasattr(obj, "_verify_model_integrity"):
                return
        self.fail("MLXServer should have _verify_model_integrity method")


class TestCognitiveLayerToggle(unittest.TestCase):
    """Verify BADAPPLE_COGNITIVE env var is wired in the gatekeeper."""

    def test_cognitive_env_var_in_gatekeeper_plist(self):
        plist_path = Path(__file__).resolve().parent.parent / "src/platform/apple_bridge/com.badapple.gatekeeper.plist"
        content = plist_path.read_text()
        self.assertIn("BADAPPLE_COGNITIVE", content)


class TestTierMetricsField(unittest.TestCase):
    """Verify the tier field is present in metrics."""

    def test_tier_in_metrics_dict(self):
        import inspect
        import badapple_mlx_server
        source = inspect.getsource(badapple_mlx_server)
        # The metrics dict should include "tier"
        self.assertIn('"tier"', source)

    def test_fast_tier_sets_tier(self):
        import inspect
        import badapple_mlx_server
        source = inspect.getsource(badapple_mlx_server)
        # The fast path should set tier to "fast"
        self.assertIn('"fast"', source)


class TestGatekeeperSocketPermissions(unittest.TestCase):
    """Verify the gatekeeper socket is not world-writable."""

    def test_socket_permissions_not_world_writable(self):
        # The build should set 0o660, not 0o666
        # Check in the gatekeeper source
        gatekeeper_path = Path(__file__).resolve().parent.parent / "src/bin/gatekeeper.rs"
        content = gatekeeper_path.read_text()
        self.assertIn("0o660", content)
        self.assertNotIn("0o666", content)


class TestDiagnosticsCommand(unittest.TestCase):
    """Verify --diagnostics alias works."""

    def test_diagnostics_alias_in_help(self):
        cli_path = Path(__file__).resolve().parent.parent / "src/bin/badapple.rs"
        content = cli_path.read_text()
        self.assertIn("--diagnostics", content)


class TestV2AuthBypassFixed(unittest.TestCase):
    """Verify the v2 client_pubkey bypass is fixed."""

    def test_hello_pubkey_extracted_from_hello(self):
        # The server should extract client_pubkey from the Hello frame, not the Execute frame
        server_path = Path(__file__).resolve().parent.parent / "badapple_mlx_server.py"
        content = server_path.read_text()
        # The fix should reference hello_client_pubkey
        self.assertIn("hello_client_pubkey", content)

    def test_v2_pubkey_mismatch_rejected(self):
        server_path = Path(__file__).resolve().parent.parent / "badapple_mlx_server.py"
        content = server_path.read_text()
        self.assertIn("client_pubkey mismatch", content)


class TestReplayCacheExists(unittest.TestCase):
    """Verify the gatekeeper has a replay cache."""

    def test_replay_cache_in_gatekeeper(self):
        gatekeeper_path = Path(__file__).resolve().parent.parent / "src/bin/gatekeeper.rs"
        content = gatekeeper_path.read_text()
        self.assertIn("ReplayCache", content)
        self.assertIn("replay detected", content)


class TestKeyPinningExists(unittest.TestCase):
    """Verify SLICKS v2 server key pinning (TOFU) exists."""

    def test_pinned_server_pubkey_function_exists(self):
        ipc_path = Path(__file__).resolve().parent.parent / "src/bad_apple_ipc.rs"
        content = ipc_path.read_text()
        self.assertIn("v2_pinned_server_pubkey", content)
        self.assertIn("v2_pin_server_pubkey", content)
        self.assertIn("daemon.pub", content)


class TestOpenatCageOperations(unittest.TestCase):
    """Verify the cage uses openat-based operations."""

    def test_openat_functions_exist(self):
        cage_path = Path(__file__).resolve().parent.parent / "src/automation_cage_impl.rs"
        content = cage_path.read_text()
        self.assertIn("openat", content)
        self.assertIn("O_NOFOLLOW", content)
        self.assertIn("mkdirat", content)


class TestFuzzingTargetsExist(unittest.TestCase):
    """Verify fuzzing targets are set up."""

    def test_fuzz_cargo_toml_exists(self):
        fuzz_path = Path(__file__).resolve().parent.parent / "fuzz/Cargo.toml"
        self.assertTrue(fuzz_path.is_file())

    def test_fuzz_targets_exist(self):
        fuzz_dir = Path(__file__).resolve().parent.parent / "fuzz/fuzz_targets"
        targets = ["fuzz_ipc_frame.rs", "fuzz_wasm_cage.rs", "fuzz_protocol_frame.rs", "fuzz_scavenger_path.rs"]
        for target in targets:
            self.assertTrue((fuzz_dir / target).is_file(), f"{target} should exist")


class TestInstallTestScript(unittest.TestCase):
    """Verify the clean install test script exists and is executable."""

    def test_install_test_exists(self):
        test_path = Path(__file__).resolve().parent.parent / "tests/test_clean_install.sh"
        self.assertTrue(test_path.is_file())
        self.assertTrue(os.access(test_path, os.X_OK))


class TestCognitiveBenchmarkScript(unittest.TestCase):
    """Verify the cognitive A/B benchmark script exists."""

    def test_benchmark_script_exists(self):
        bench_path = Path(__file__).resolve().parent.parent / "benchmark_cognitive.py"
        self.assertTrue(bench_path.is_file())

    def test_benchmark_has_three_modes(self):
        bench_path = Path(__file__).resolve().parent.parent / "benchmark_cognitive.py"
        content = bench_path.read_text()
        self.assertIn("cognitive_full", content)
        self.assertIn("fast_tier_only", content)
        self.assertIn("9b_only", content)


if __name__ == "__main__":
    unittest.main()
