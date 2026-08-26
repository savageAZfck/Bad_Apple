#!/usr/bin/env python3
"""Secure Enclave identity and biometric approval bridge."""

import base64
import os
import pwd
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_HELPER = Path(__file__).resolve().parent / "target" / "release" / "badapple-identity"


class IdentityError(RuntimeError):
    pass


def _helper() -> Path:
    return Path(os.environ.get("BADAPPLE_IDENTITY_HELPER", str(DEFAULT_HELPER))).expanduser()


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
        except Exception:
            pass
    return str(Path(home) / "Library/Application Support/BadApple/identity.sekey")


def _run(command: str, argument: Optional[str] = None, timeout: int = 60) -> str:
    helper = _helper()
    if not helper.is_file():
        raise IdentityError(f"Secure Enclave helper is missing: {helper}")
    args = [str(helper), command]
    if argument is not None:
        args.append(argument)
    env = os.environ.copy()
    env.setdefault("BADAPPLE_IDENTITY_BLOB", _identity_blob_path())
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode != 0:
        raise IdentityError((result.stderr or result.stdout or "identity helper failed").strip())
    return result.stdout.strip()


def ensure_identity() -> str:
    return _run("ensure")


def public_key() -> str:
    return _run("public-key")


def sign(message: bytes) -> str:
    return _run("sign", base64.b64encode(message).decode("ascii"))


def biometric_gate(reason: str = "Approve a sensitive Bad Apple action") -> bool:
    return _run("biometric-gate", reason, timeout=120) == "approved"


def status() -> str:
    return _run("status")
