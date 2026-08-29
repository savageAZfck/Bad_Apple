#!/usr/bin/env python3
"""Secure Enclave identity and biometric approval bridge.

Prefers a long-lived identity agent (running in the user's Aqua session) over
spawning the one-shot helper directly. This keeps Secure Enclave access inside
a GUI session and makes it available to Background / LaunchDaemon processes.
"""

import base64
import json
import os
import pwd
import socket
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_HELPER = Path(__file__).resolve().parent / "target" / "release" / "badapple-identity"
DEFAULT_AGENT_SOCKET = Path("/var/run/badapple/identity.sock")


class IdentityError(RuntimeError):
    pass


def _helper() -> Path:
    return Path(os.environ.get("BADAPPLE_IDENTITY_HELPER", str(DEFAULT_HELPER))).expanduser()


def _agent_socket() -> Path | None:
    raw = os.environ.get("BADAPPLE_IDENTITY_AGENT_SOCKET")
    if raw == "":
        return None
    return Path(raw) if raw else DEFAULT_AGENT_SOCKET


def _identity_blob_path() -> str:
    if "BADAPPLE_IDENTITY_BLOB" in os.environ:
        return os.environ["BADAPPLE_IDENTITY_BLOB"]
    home = os.path.expanduser("~")
    if os.getuid() == 0:
        try:
            user = subprocess.run(
                ["stat", "-f", "%Su", "/dev/console"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            ).stdout.strip()
            home = pwd.getpwnam(user).pw_dir
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            print(f"[identity] strip failed: {e}", flush=True)
    return str(Path(home) / "Library/Application Support/BadApple/identity.sekey")


def _agent_request(request: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    """Send a single JSON request to the identity agent and return its response."""
    sock_path = _agent_socket()
    if sock_path is None or not sock_path.exists():
        raise IdentityError("identity agent socket not found")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall((json.dumps(request) + "\n").encode())
        with s.makefile("r") as f:
            line = f.readline()
    if not line:
        raise IdentityError("identity agent closed connection without response")
    try:
        response = json.loads(line)
    except json.JSONDecodeError as e:
        raise IdentityError(f"identity agent returned invalid json: {e}") from e
    if not response.get("ok"):
        raise IdentityError(response.get("error", "identity agent request failed"))
    return response


def _run(command: str, argument: str | None = None, timeout: int = 60) -> str:
    helper = _helper()
    if not helper.is_file():
        raise IdentityError(f"Secure Enclave helper is missing: {helper}")
    args = [str(helper), command]
    if argument is not None:
        args.append(argument)
    env = os.environ.copy()
    env.setdefault("BADAPPLE_IDENTITY_BLOB", _identity_blob_path())
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    if result.returncode != 0:
        raise IdentityError((result.stderr or result.stdout or "identity helper failed").strip())
    return result.stdout.strip()


def ensure_identity() -> str:
    try:
        return _agent_request({"command": "public_key"})["public_key"]
    except IdentityError:
        return _run("ensure")


def public_key() -> str:
    try:
        return _agent_request({"command": "public_key"})["public_key"]
    except IdentityError:
        return _run("public-key")


def sign(message: bytes) -> str:
    try:
        return _agent_request({
            "command": "sign",
            "message_b64": base64.b64encode(message).decode("ascii"),
        })["signature"]
    except IdentityError:
        return _run("sign", base64.b64encode(message).decode("ascii"))


def biometric_gate(reason: str = "Approve a sensitive Bad Apple action") -> bool:
    try:
        response = _agent_request({"command": "biometric_gate", "reason": reason}, timeout=120.0)
        return response.get("result") == "approved"
    except IdentityError:
        return _run("biometric-gate", reason, timeout=120) == "approved"


def status() -> str:
    try:
        return _agent_request({"command": "status"})["status"]
    except IdentityError:
        return _run("status")
