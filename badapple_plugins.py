#!/usr/bin/env python3
"""Signed, subprocess-isolated local tool plugin registry."""

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class PluginError(RuntimeError):
    pass


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PluginRegistry:
    """Loads only manifests signed by a locally trusted Ed25519 publisher."""

    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "plugins"
        self.trust_dir = Path(data_dir) / "trusted_publishers"
        self.root.mkdir(parents=True, exist_ok=True)
        self.trust_dir.mkdir(parents=True, exist_ok=True)
        self.plugins: dict[str, dict[str, Any]] = {}
        self.tools: dict[str, dict[str, Any]] = {}
        self.reload()

    def _trusted_key(self, publisher: str) -> Ed25519PublicKey:
        key_path = self.trust_dir / f"{publisher}.pub"
        if not key_path.is_file():
            raise PluginError(f"publisher '{publisher}' is not trusted")
        try:
            raw = base64.b64decode(key_path.read_text(encoding="ascii").strip(), validate=True)
            return Ed25519PublicKey.from_public_bytes(raw)
        except Exception as e:
            raise PluginError(f"invalid trusted key for '{publisher}': {e}") from e

    def verify(self, plugin_dir: Path) -> dict[str, Any]:
        manifest_path = plugin_dir / "plugin.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise PluginError(f"invalid manifest {manifest_path}: {e}") from e
        required = {"name", "version", "publisher", "entrypoint", "tools", "files", "signature"}
        missing = required - set(manifest)
        if missing:
            raise PluginError(f"manifest missing: {', '.join(sorted(missing))}")
        if Path(manifest["entrypoint"]).is_absolute() or ".." in Path(manifest["entrypoint"]).parts:
            raise PluginError("entrypoint must stay inside the plugin directory")
        for relative, expected in manifest["files"].items():
            path = (plugin_dir / relative).resolve()
            if plugin_dir.resolve() not in path.parents or not path.is_file():
                raise PluginError(f"unsafe or missing plugin file: {relative}")
            if _hash(path) != expected:
                raise PluginError(f"plugin file hash mismatch: {relative}")
        unsigned = dict(manifest)
        signature = base64.b64decode(unsigned.pop("signature"), validate=True)
        self._trusted_key(manifest["publisher"]).verify(signature, _canonical(unsigned))
        return manifest

    def reload(self) -> dict[str, str]:
        self.plugins.clear()
        self.tools.clear()
        errors: dict[str, str] = {}
        for plugin_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            try:
                manifest = self.verify(plugin_dir)
                self.plugins[manifest["name"]] = {"dir": plugin_dir, "manifest": manifest}
                for tool in manifest["tools"]:
                    name = tool.get("name")
                    if not name or name in self.tools:
                        raise PluginError(f"duplicate or invalid tool name: {name}")
                    self.tools[name] = {"plugin": manifest["name"], "schema": tool}
            except (LookupError, TypeError, ValueError) as e:
                errors[plugin_dir.name] = str(e)
        return errors

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": item["schema"].get("description", "Signed local plugin tool."),
                    "parameters": item["schema"].get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for name, item in sorted(self.tools.items())
        ]

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def invoke(self, name: str, arguments: dict[str, Any], timeout: int = 30) -> str:
        item = self.tools.get(name)
        if item is None:
            raise PluginError(f"unknown plugin tool: {name}")
        plugin = self.plugins[item["plugin"]]
        plugin_dir: Path = plugin["dir"]
        entrypoint = plugin_dir / plugin["manifest"]["entrypoint"]
        env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(Path.home()),
            "BADAPPLE_PLUGIN": plugin["manifest"]["name"],
        }
        result = subprocess.run(
            [str(entrypoint)],
            input=json.dumps({"tool": name, "arguments": arguments}) + "\n",
            capture_output=True,
            text=True,
            cwd=plugin_dir,
            env=env,
            timeout=min(max(timeout, 1), 120),
        check=False)
        if result.returncode != 0:
            raise PluginError((result.stderr or result.stdout or "plugin failed")[:2000])
        return result.stdout.strip()
