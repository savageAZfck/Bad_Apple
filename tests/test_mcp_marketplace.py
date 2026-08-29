#!/usr/bin/env python3
"""Unit tests for the MCP marketplace catalog and registry helpers."""

import json
import random
import re
import shutil
import string
import tempfile
import unittest
from pathlib import Path

import badapple_mcp_marketplace


class CatalogTests(unittest.TestCase):
    def test_load_catalog_returns_servers(self) -> None:
        """list_catalog_servers parses a valid JSON catalog."""
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "mcp_registry.json"
            catalog.write_text(json.dumps({
                "version": 1,
                "servers": [
                    {"name": "time", "description": "Get the time"},
                    {"name": "fs", "description": "Read files"},
                ],
            }))
            servers = badapple_mcp_marketplace.list_catalog_servers(catalog)
            self.assertEqual(len(servers), 2)
            self.assertEqual(servers[0]["name"], "time")

    def test_load_catalog_missing_file_returns_empty(self) -> None:
        """A missing catalog returns an empty list."""
        servers = badapple_mcp_marketplace.list_catalog_servers(Path("/no/such/file.json"))
        self.assertEqual(servers, [])

    def test_marketplace_catalog_uses_home_paths(self) -> None:
        """The built-in catalog uses the user's home, not a hardcoded dev path."""
        catalog = badapple_mcp_marketplace.marketplace_catalog()
        self.assertIn(str(Path.home()), catalog)
        self.assertIn("Documents", catalog)
        self.assertIn("filesystem", catalog)
        self.assertIn("fetch", catalog)
        self.assertIn("time", catalog)

    def test_install_unknown_server(self) -> None:
        """Installing an unknown catalog server returns an error."""
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "mcp_registry.json"
            catalog.write_text(json.dumps({"version": 1, "servers": []}))
            result = badapple_mcp_marketplace.install_catalog_server("missing", catalog)
            self.assertIn("not found", result)


class PathHelperTests(unittest.TestCase):
    def test_mcp_data_dir_is_user_writable(self) -> None:
        """_mcp_data_dir points to a directory under the user's home."""
        path = badapple_mcp_marketplace._mcp_data_dir()
        self.assertTrue(str(path).startswith(str(Path.home())))
        self.assertTrue(path.is_dir())

    def test_mcp_workspace_dir_is_documents(self) -> None:
        """_mcp_workspace_dir points at the user Documents folder."""
        path = badapple_mcp_marketplace._mcp_workspace_dir()
        self.assertEqual(path.name, "Documents")


class SecurityPropertyTests(unittest.TestCase):
    """Property and fuzz-style checks for the MCP marketplace hardening."""

    def setUp(self) -> None:
        self._original_registry = badapple_mcp_marketplace.DEFAULT_REGISTRY
        self._home_tmp = Path(tempfile.mkdtemp(dir=Path.home()))
        badapple_mcp_marketplace.DEFAULT_REGISTRY = self._home_tmp / "mcp_servers.json"

    def tearDown(self) -> None:
        badapple_mcp_marketplace.DEFAULT_REGISTRY = self._original_registry
        badapple_mcp_marketplace._MARKETPLACE.stop_all()
        shutil.rmtree(self._home_tmp, ignore_errors=True)

    def _write_catalog(self, servers: list[dict]) -> Path:
        catalog = self._home_tmp / "mcp_registry.json"
        catalog.write_text(json.dumps({"version": 1, "servers": servers}))
        return catalog

    def test_valid_catalog_and_install(self) -> None:
        """A clean catalog with safe commands can be listed and installed."""
        catalog = self._write_catalog([
            {
                "name": "echo",
                "description": "Echo server",
                "install_type": "command",
                "command": ["/bin/echo", "ok"],
                "env": {"FOO": "bar"},
            },
            {
                "name": "pythonmod",
                "description": "Python module",
                "install_type": "command",
                "command": ["python", "-m", "mcp_server_time"],
            },
        ])
        servers = badapple_mcp_marketplace.list_catalog_servers(catalog)
        self.assertEqual(len(servers), 2)

        result = badapple_mcp_marketplace.install_catalog_server("echo", catalog)
        self.assertIn("Added", result)

        result = badapple_mcp_marketplace.list_mcp_servers()
        self.assertIn("echo", result)

    def test_invalid_server_names_fuzz(self) -> None:
        """Random invalid server names are rejected by all entry points."""
        unsafe_alphabet = string.punctuation + string.whitespace
        for _ in range(200):
            length = random.randint(0, 80)
            name = "".join(random.choice(unsafe_alphabet) for _ in range(length))
            if not re.fullmatch(r"^[A-Za-z0-9_-]{1,64}$", name):
                self.assertFalse(badapple_mcp_marketplace._is_safe_name(name))
                self.assertIn("MCP server name", badapple_mcp_marketplace.add_mcp_server(name, "echo ok"))
                self.assertIn("MCP server name", badapple_mcp_marketplace.remove_mcp_server(name))
                self.assertIn("MCP server name", badapple_mcp_marketplace.invoke_mcp_tool(name, "x", {}))

        for _ in range(50):
            length = random.randint(1, 64)
            alphabet = string.ascii_letters + string.digits + "_-"
            name = "".join(random.choice(alphabet) for _ in range(length))
            self.assertTrue(badapple_mcp_marketplace._is_safe_name(name))

    def test_path_traversal_in_catalog_path(self) -> None:
        """Catalog paths with parent-directory references are rejected."""
        bad_path = self._home_tmp / ".." / "evil.json"
        self.assertEqual(badapple_mcp_marketplace.list_catalog_servers(bad_path), [])

    def test_path_traversal_in_commands(self) -> None:
        """Command/path arguments outside the user home are rejected."""
        # Outside home.
        self.assertFalse(badapple_mcp_marketplace._is_safe_path_arg("/etc/passwd"))
        self.assertFalse(badapple_mcp_marketplace._is_safe_path_arg("~/../.bashrc"))

        # Parent-directory escape.
        self.assertFalse(badapple_mcp_marketplace._is_safe_path_arg(".."))

        # Safe home paths.
        self.assertTrue(badapple_mcp_marketplace._is_safe_path_arg("~"))
        self.assertTrue(badapple_mcp_marketplace._is_safe_path_arg("~/Documents"))
        self.assertTrue(badapple_mcp_marketplace._is_safe_path_arg("Documents/secret"))

        # Absolute executable outside allowed bases.
        with self.assertRaises(badapple_mcp_marketplace.SecurityError):
            badapple_mcp_marketplace._resolve_command(["/tmp/evil"])

        # Path traversal in an argument.
        with self.assertRaises(badapple_mcp_marketplace.SecurityError):
            badapple_mcp_marketplace._resolve_command(["/bin/echo", "~/../.bashrc"])

        # Shell interpreters and python -c are blocked.
        with self.assertRaises(badapple_mcp_marketplace.SecurityError):
            badapple_mcp_marketplace._resolve_command(["/bin/sh", "-c", "ls"])
        with self.assertRaises(badapple_mcp_marketplace.SecurityError):
            badapple_mcp_marketplace._resolve_command(["python", "-c", "print(1)"])

        # Via public API.
        result = badapple_mcp_marketplace.add_mcp_server("bad", "/bin/echo /etc/passwd")
        self.assertIn("rejected", result)

        # MCPClient construction also rejects unsafe commands.
        with self.assertRaises(badapple_mcp_marketplace.SecurityError):
            badapple_mcp_marketplace.MCPClient("bad", ["/bin/sh", "-c", "ls"])

    def test_oversized_input(self) -> None:
        """Oversized JSON payloads and command arguments are rejected."""
        huge = self._home_tmp / "huge.json"
        huge.write_bytes(b"x" * (badapple_mcp_marketplace.MAX_JSON_BYTES + 1))
        self.assertEqual(badapple_mcp_marketplace._load_catalog(huge), {"servers": []})

        huge_arg = "x" * (badapple_mcp_marketplace.MAX_ARG_LEN + 1)
        with self.assertRaises(badapple_mcp_marketplace.SecurityError):
            badapple_mcp_marketplace._resolve_command(["/bin/echo", huge_arg])

        long_name = "a" * (badapple_mcp_marketplace.MAX_SERVER_NAME_LEN + 1)
        self.assertFalse(badapple_mcp_marketplace._is_safe_name(long_name))

    def test_env_sanitization(self) -> None:
        """Dangerous environment keys are stripped, safe keys are kept."""
        env = {
            "PATH": "/tmp",
            "LD_PRELOAD": "/tmp/lib.so",
            "DYLD_INSERT_LIBRARIES": "/tmp/lib.dylib",
            "PYTHONPATH": "/tmp",
            "API_KEY": "secret",
        }
        safe = badapple_mcp_marketplace._sanitize_env(env)
        self.assertEqual(safe, {"API_KEY": "secret"})
        self.assertIsNotNone(badapple_mcp_marketplace._validate_env({"PATH": "/tmp"}))


if __name__ == "__main__":
    unittest.main()
