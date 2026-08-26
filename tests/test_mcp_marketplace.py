#!/usr/bin/env python3
"""Unit tests for the MCP marketplace catalog and registry helpers."""

import json
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
        self.assertIn(".bad_apple/mcp_data", catalog)
        self.assertIn("filesystem", catalog)
        self.assertIn("sqlite", catalog)

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
        """_mcp_workspace_dir points at the user's Documents folder."""
        path = badapple_mcp_marketplace._mcp_workspace_dir()
        self.assertEqual(path.name, "Documents")


if __name__ == "__main__":
    unittest.main()
