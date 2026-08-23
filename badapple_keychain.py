#!/usr/bin/env python3
"""macOS Keychain-backed SLICKS secret storage for Bad Apple.

Stores the 32-byte hex SLICKS secret in the macOS Keychain instead of a plain
file. Generates a random secret if one does not exist. This keeps the secret
out of the filesystem and inside Apple's keychain (FileVault / Secure Enclave
protected when the keychain is configured accordingly).
"""

import os
import secrets
import shutil
import subprocess
from typing import Optional

DEFAULT_SERVICE = "com.badapple.slicks"
DEFAULT_ACCOUNT = "mlx-server"


def _run(args: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def _keychain_file() -> str:
    """Use the system keychain if available, otherwise the login keychain."""
    for path in ("/Library/Keychains/System.keychain", "~/Library/Keychains/login.keychain-db"):
        p = os.path.expanduser(path)
        if os.path.isfile(p):
            return p
    return ""


def get_secret(service: str = DEFAULT_SERVICE, account: str = DEFAULT_ACCOUNT) -> Optional[bytes]:
    """Load the SLICKS secret from the macOS Keychain."""
    keychain = _keychain_file()
    cmd = ["security", "find-generic-password", "-s", service, "-a", account, "-w"]
    if keychain:
        cmd.extend(["-k", keychain])
    rc, out, err = _run(cmd, timeout=10)
    if rc != 0:
        # No existing item.
        return None
    trimmed = out.strip()
    if all(c in "0123456789abcdefABCDEF" for c in trimmed) and len(trimmed) >= 32:
        return bytes.fromhex(trimmed)
    return trimmed.encode()


def store_secret(secret: bytes, service: str = DEFAULT_SERVICE, account: str = DEFAULT_ACCOUNT) -> str:
    """Store or update the SLICKS secret in the macOS Keychain."""
    if shutil.which("security") is None:
        return "Error: `security` CLI not found."
    keychain = _keychain_file()
    hex_secret = secret.hex() if isinstance(secret, bytes) else secret

    # Remove any existing item first.
    _run(["security", "delete-generic-password", "-s", service, "-a", account], timeout=5)

    cmd = [
        "security", "add-generic-password",
        "-s", service,
        "-a", account,
        "-w", hex_secret,
        "-T", "",  # no specific app access; prompt on use
    ]
    if keychain:
        cmd.extend(["-k", keychain])
    else:
        cmd.append("-U")  # Update if exists
    rc, _out, err = _run(cmd, timeout=10)
    if rc != 0:
        return f"Keychain store error: {err}"
    return f"SLICKS secret stored in keychain (service={service}, account={account})."


def get_or_create_secret(service: str = DEFAULT_SERVICE, account: str = DEFAULT_ACCOUNT) -> bytes:
    """Load the SLICKS secret from keychain, or create a new 32-byte one."""
    existing = get_secret(service, account)
    if existing is not None:
        return existing
    new_secret = secrets.token_bytes(32)
    msg = store_secret(new_secret, service, account)
    if "Error" in msg:
        raise RuntimeError(msg)
    return new_secret


def rotate_secret(service: str = DEFAULT_SERVICE, account: str = DEFAULT_ACCOUNT) -> str:
    """Generate and store a new SLICKS secret."""
    new_secret = secrets.token_bytes(32)
    return store_secret(new_secret, service, account)
