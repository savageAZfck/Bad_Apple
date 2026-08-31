#!/usr/bin/env python3
"""SLICKS and SLICKS 2.0 authentication helpers.

SLICKS v1 uses a shared HMAC secret.
SLICKS v2 uses the Secure Enclave to sign the server challenge and the client
execute frame; both sides verify the signature against the local device public
key. The private key never leaves the Secure Enclave.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import threading
import time

import badapple_identity
import badapple_keychain

SLICKS_VERSION = 1
SLICKS_VERSION_2 = 2
DEFAULT_KEY_PATH = "/var/lib/bad_apple/slicks.key"
HANDSHAKE_MAX_SKEW_MS = 60_000


def _is_hex(s: str, length: int | None = None) -> bool:
    if length is not None and len(s) != length:
        return False
    return bool(re.fullmatch(r"[0-9a-fA-F]+" if length is None else rf"[0-9a-fA-F]{{{length}}}", s))


def nonce_is_valid(nonce: str) -> bool:
    return _is_hex(nonce, 64)


def timestamp_is_fresh(timestamp_ms: int) -> bool:
    return abs(int(time.time() * 1000) - timestamp_ms) <= HANDSHAKE_MAX_SKEW_MS


# =============================================================================
# SLICKS v1 (shared HMAC)
# =============================================================================


def load_slicks_secret() -> bytes:
    if "BADAPPLE_SLICKS_SECRET" in os.environ:
        raw = os.environ["BADAPPLE_SLICKS_SECRET"]
    elif os.environ.get("BADAPPLE_SLICKS_KEYCHAIN", "0") == "1":
        try:
            return badapple_keychain.get_or_create_secret()
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            print(f"[slicks] keychain load failed: {e}; falling back to key file", flush=True)
            key_path = os.environ.get("BADAPPLE_SLICKS_KEY_PATH", DEFAULT_KEY_PATH)
            raw = _read_secret_file(key_path)
    else:
        key_path = os.environ.get("BADAPPLE_SLICKS_KEY_PATH", DEFAULT_KEY_PATH)
        raw = _read_secret_file(key_path)
    trimmed = raw.strip()
    if all(c in "0123456789abcdefABCDEF" for c in trimmed) and len(trimmed) >= 32:
        return bytes.fromhex(trimmed)
    return trimmed.encode()


def _read_secret_file(key_path: str) -> str:
    """Read the SLICKS secret from a file, enforcing strict permissions."""
    import stat
    st = os.stat(key_path)
    mode = stat.S_IMODE(st.st_mode)
    # Reject world- or group-readable key files.
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        raise PermissionError(
            f"SLICKS key file {key_path} is group/world readable (mode {oct(mode)}); "
            f"refusing to load. Fix with: chmod 600 {key_path}"
        )
    with open(key_path) as f:
        return f.read()


def _sign(secret: bytes, material: bytes) -> str:
    mac = hmac.new(secret, material, hashlib.sha256)
    return mac.hexdigest()


def _verify(secret: bytes, material: bytes, proof: str) -> bool:
    if not _is_hex(proof, 64):
        return False
    return hmac.compare_digest(_sign(secret, material).lower(), proof.lower())


def server_material_v1(timestamp_ms: int, client_nonce: str, server_nonce: str) -> bytes:
    return f"BADAPPLE-SLICKS/{SLICKS_VERSION}|server|{timestamp_ms}|{client_nonce}|{server_nonce}".encode()


def client_material_v1(timestamp_ms: int, client_nonce: str, server_nonce: str, prompt: str, max_new_tokens: int) -> bytes:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return f"BADAPPLE-SLICKS/{SLICKS_VERSION}|client|{timestamp_ms}|{client_nonce}|{server_nonce}|{max_new_tokens}|{prompt_hash}".encode()


def v1_server_proof(secret: bytes, timestamp_ms: int, client_nonce: str, server_nonce: str) -> str:
    return _sign(secret, server_material_v1(timestamp_ms, client_nonce, server_nonce))


def v1_client_proof(secret: bytes, timestamp_ms: int, client_nonce: str, server_nonce: str, prompt: str, max_new_tokens: int) -> str:
    return _sign(secret, client_material_v1(timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens))


def v1_verify_server_proof(secret: bytes, timestamp_ms: int, client_nonce: str, server_nonce: str, proof: str) -> bool:
    return _verify(secret, server_material_v1(timestamp_ms, client_nonce, server_nonce), proof)


def v1_verify_client_proof(secret: bytes, timestamp_ms: int, client_nonce: str, server_nonce: str, prompt: str, max_new_tokens: int, proof: str) -> bool:
    return _verify(secret, client_material_v1(timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens), proof)


# =============================================================================
# SLICKS v2 (Secure Enclave ECDSA)
# =============================================================================


class Slicks2State:
    """Caches the SE public key and badapple-identity availability."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._public_key_b64: str | None = None
        self._available: bool | None = None

    def available(self) -> bool:
        with self._lock:
            if self._available is None:
                try:
                    badapple_identity.public_key()
                    self._available = True
                except Exception:  # noqa: BLE001
                    self._available = False
            return self._available

    def public_key_b64(self) -> str | None:
        with self._lock:
            if self._public_key_b64 is None:
                try:
                    self._public_key_b64 = badapple_identity.public_key()
                    self._available = True
                except Exception:  # noqa: BLE001
                    pass
            return self._public_key_b64

    def public_key(self) -> bytes | None:
        b64 = self.public_key_b64()
        if b64 is None:
            return None
        try:
            return base64.b64decode(b64)
        except Exception:  # noqa: BLE001
            return None


_state = Slicks2State()
_se_sign_lock = threading.Lock()


def v2_available() -> bool:
    return _state.available()


def v2_public_key_b64() -> str | None:
    return _state.public_key_b64()


def v2_public_key() -> bytes | None:
    return _state.public_key()


def _sign_with_identity(message: bytes) -> str:
    """Sign a message using the SE helper. Returns base64 DER ECDSA signature.

    Serialized with a lock because the Secure Enclave helper can race when
    invoked concurrently from the same process.
    """
    with _se_sign_lock:
        return badapple_identity.sign(message)


def _verify_with_pubkey(message: bytes, signature_b64: str, public_key: bytes) -> bool:
    """Verify a base64 DER ECDSA signature over SHA-256(message)."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), public_key)
        sig = base64.b64decode(signature_b64)
        pub.verify(sig, message, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, Exception):
        return False


def server_material_v2(timestamp_ms: int, client_nonce: str, server_nonce: str) -> bytes:
    return f"BADAPPLE-SLICKS/{SLICKS_VERSION_2}|server|{timestamp_ms}|{client_nonce}|{server_nonce}".encode()


def client_material_v2(timestamp_ms: int, client_nonce: str, server_nonce: str, prompt: str, max_new_tokens: int) -> bytes:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return f"BADAPPLE-SLICKS/{SLICKS_VERSION_2}|client|{timestamp_ms}|{client_nonce}|{server_nonce}|{max_new_tokens}|{prompt_hash}".encode()


def v2_server_proof(timestamp_ms: int, client_nonce: str, server_nonce: str) -> str:
    return _sign_with_identity(server_material_v2(timestamp_ms, client_nonce, server_nonce))


def v2_verify_client_proof(timestamp_ms: int, client_nonce: str, server_nonce: str, prompt: str, max_new_tokens: int, proof: str, public_key: bytes | None = None) -> bool:
    pk = public_key or v2_public_key()
    if pk is None:
        return False
    return _verify_with_pubkey(client_material_v2(timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens), proof, pk)


def v2_client_proof(timestamp_ms: int, client_nonce: str, server_nonce: str, prompt: str, max_new_tokens: int) -> str:
    return _sign_with_identity(client_material_v2(timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens))


def v2_verify_server_proof(timestamp_ms: int, client_nonce: str, server_nonce: str, proof: str, public_key: bytes | None = None) -> bool:
    pk = public_key or v2_public_key()
    if pk is None:
        return False
    return _verify_with_pubkey(server_material_v2(timestamp_ms, client_nonce, server_nonce), proof, pk)


def v2_sign_message(message: bytes) -> str | None:
    """Sign an arbitrary message with the Secure Enclave identity.

    Returns a base64 DER ECDSA signature, or None if no identity is available.
    """
    if v2_public_key() is None:
        return None
    return _sign_with_identity(message)


def v2_verify_message(message: bytes, signature_b64: str, public_key_b64: str) -> bool:
    """Verify an arbitrary Secure Enclave signature over a message.

    The public key is expected as a base64-encoded P-256 uncompressed point.
    """
    try:
        public_key = base64.b64decode(public_key_b64)
    except Exception:  # noqa: BLE001
        return False
    return _verify_with_pubkey(message, signature_b64, public_key)


# =============================================================================
# Unified dispatch
# =============================================================================


def version_for_request(client_version: int, require_v2: bool) -> int:
    """Return the negotiated version."""
    if require_v2 and client_version != SLICKS_VERSION_2:
        raise ValueError("SLICKS v2 is required")
    if client_version == SLICKS_VERSION_2 and v2_available():
        return SLICKS_VERSION_2
    return SLICKS_VERSION
