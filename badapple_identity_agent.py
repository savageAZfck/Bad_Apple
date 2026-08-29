#!/usr/bin/env python3
"""Long-lived identity agent that keeps Secure Enclave signing in the user
Aqua / GUI session.

The Bad Apple MLX daemon and CLI connect to this agent over a Unix socket
instead of calling the one-shot `badapple-identity` helper directly. Because
this agent runs in the user's Aqua session, `CryptoTokenKit` / Secure Enclave
key operations succeed reliably.

The agent serializes signing requests because the one-shot helper is not safe
for immediate back-to-back invocations from the same process.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import badapple_identity


DEFAULT_SOCKET_PATH = Path("/var/run/badapple/identity.sock")
SIGN_COOLDOWN_S = 0.25


class IdentityAgent:
    """Unix-socket identity agent.

    Protocol (JSON line per connection):
        {"command": "public_key"}
        -> {"ok": True, "public_key": "<base64>"}

        {"command": "sign", "message_b64": "<base64>"}
        -> {"ok": True, "signature": "<base64>"}

        {"command": "status"}
        -> {"ok": True, "status": "secure-enclave:<fingerprint>|missing|unavailable"}

        {"command": "biometric_gate", "reason": "..."}
        -> {"ok": True, "result": "approved"} or {"ok": False, "error": "..."}
    """

    def __init__(self, sock_path: Path | None = None) -> None:
        self.sock_path = sock_path or DEFAULT_SOCKET_PATH
        self._sign_lock = threading.Lock()
        self._pub_key_b64: str | None = None
        self._running = False
        self._server: socket.socket | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.sock_path.unlink()
        except FileNotFoundError:
            pass
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.sock_path))
        self._server.listen(64)
        os.chmod(self.sock_path, 0o660)
        self._running = True
        accept_thread = threading.Thread(target=self._accept_loop, name="identity-accept", daemon=True)
        accept_thread.start()
        self._threads.append(accept_thread)

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=timeout / max(len(self._threads), 1))
        try:
            self.sock_path.unlink()
        except FileNotFoundError:
            pass

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            t.start()

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            with conn.makefile("r") as f:
                line = f.readline()
            if not line:
                return
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self._send(conn, {"ok": False, "error": f"invalid json: {e}"})
                return
            self._dispatch(conn, request)
        finally:
            conn.close()

    def _send(self, conn: socket.socket, payload: dict[str, Any]) -> None:
        try:
            conn.sendall((json.dumps(payload) + "\n").encode())
        except OSError:
            pass

    def _dispatch(self, conn: socket.socket, request: dict[str, Any]) -> None:
        command = request.get("command")
        if command == "public_key":
            self._send(conn, self._public_key())
        elif command == "sign":
            self._send(conn, self._sign(request))
        elif command == "verify":
            self._send(conn, self._verify(request))
        elif command == "status":
            self._send(conn, self._status())
        elif command == "biometric_gate":
            self._send(conn, self._biometric_gate(request))
        else:
            self._send(conn, {"ok": False, "error": f"unknown command: {command}"})

    def _public_key(self) -> dict[str, Any]:
        try:
            if self._pub_key_b64 is None:
                self._pub_key_b64 = badapple_identity.public_key()
            return {"ok": True, "public_key": self._pub_key_b64}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def _sign(self, request: dict[str, Any]) -> dict[str, Any]:
        message_b64 = request.get("message_b64", "")
        try:
            message = base64.b64decode(message_b64)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"invalid message_b64: {e}"}
        with self._sign_lock:
            try:
                signature = badapple_identity.sign(message)
                time.sleep(SIGN_COOLDOWN_S)
                return {"ok": True, "signature": signature}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": str(e)}

    def _verify(self, request: dict[str, Any]) -> dict[str, Any]:
        message_b64 = request.get("message_b64", "")
        signature_b64 = request.get("signature", "")
        public_key_b64 = request.get("public_key", "")
        try:
            message = base64.b64decode(message_b64)
            signature = base64.b64decode(signature_b64)
            public_key = base64.b64decode(public_key_b64)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"invalid base64: {e}"}
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import hashes
            from cryptography.exceptions import InvalidSignature

            pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), public_key)
            pub.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            return {"ok": True, "valid": True}
        except InvalidSignature:
            return {"ok": True, "valid": False}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def _status(self) -> dict[str, Any]:
        try:
            return {"ok": True, "status": badapple_identity.status()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "status": f"unavailable: {e}"}

    def _biometric_gate(self, request: dict[str, Any]) -> dict[str, Any]:
        reason = request.get("reason", "Approve a sensitive Bad Apple action")
        try:
            ok = badapple_identity.biometric_gate(reason)
            return {"ok": True, "result": "approved" if ok else "denied"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}


def main() -> int:
    sock_path = Path(os.environ.get("BADAPPLE_IDENTITY_AGENT_SOCKET") or DEFAULT_SOCKET_PATH)
    # Disable the agent fallback inside this process so the agent talks
    # directly to the one-shot helper and does not recurse into itself.
    os.environ["BADAPPLE_IDENTITY_AGENT_SOCKET"] = ""
    os.environ.setdefault("BADAPPLE_IDENTITY_BLOB", str(Path.home() / "Library/Application Support/BadApple/identity.sekey"))
    agent = IdentityAgent(sock_path)
    agent.start()
    print(f"[identity_agent] listening on {sock_path}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
