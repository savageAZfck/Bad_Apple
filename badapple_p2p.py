#!/usr/bin/env python3
"""Encrypted peer-to-peer sync for the Bad Apple daemon.

Uses AES-256-GCM for confidentiality and HMAC-SHA256 for origin
authentication, both keyed from the SLICKS secret via HKDF. Peer discovery
is link-local UDP broadcast; sync payloads are sent over TCP.

This is intentionally air-gapped: it only ever binds to local interfaces,
never dials cloud endpoints, and discards any peer whose HMAC fails.
"""

import asyncio
import base64
import datetime
import hashlib
import hmac
import json
import os
import shutil
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

P2P_VERSION = 1
P2P_BROADCAST_PORT = int(os.environ.get("BADAPPLE_P2P_UDP_PORT", "9999"))
P2P_SYNC_PORT = int(os.environ.get("BADAPPLE_P2P_TCP_PORT", "10000"))
P2P_BROADCAST_INTERVAL = 30.0
P2P_BEACON_TTL = 120.0


def _derive_keys(secret: bytes) -> tuple:
    """Derive an encryption key and a MAC key from the SLICKS secret."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=64,
        salt=b"BADAPPLE-P2P-v1",
        info=b"peer-sync",
    )
    okm = hkdf.derive(secret)
    return okm[:32], okm[32:]


def _origin_id(secret: bytes) -> str:
    return hashlib.sha256(b"badapple-origin:" + secret).hexdigest()[:16]


@dataclass
class Peer:
    host: str
    port: int
    origin_id: str
    last_seen: float = field(default_factory=time.time)


@dataclass
class P2PFrame:
    frame_type: str
    origin_id: str
    timestamp_ms: int
    nonce_b64: str
    payload_b64: str
    proof: str

    def canonical(self) -> bytes:
        return (
            f"{P2P_VERSION}|{self.frame_type}|{self.origin_id}|"
            f"{self.timestamp_ms}|{self.nonce_b64}|{self.payload_b64}"
        ).encode()

    def to_bytes(self) -> bytes:
        return json.dumps({
            "v": P2P_VERSION,
            "t": self.frame_type,
            "o": self.origin_id,
            "ts": self.timestamp_ms,
            "n": self.nonce_b64,
            "p": self.payload_b64,
            "h": self.proof,
        }, ensure_ascii=True).encode() + b"\n"

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["P2PFrame"]:
        try:
            obj = json.loads(data.decode())
            if obj.get("v") != P2P_VERSION:
                return None
            return cls(
                frame_type=obj["t"],
                origin_id=obj["o"],
                timestamp_ms=obj["ts"],
                nonce_b64=obj["n"],
                payload_b64=obj["p"],
                proof=obj["h"],
            )
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, LookupError):
            return None


class P2PDaemon:
    """A minimal, encrypted, link-local P2P sync daemon."""

    def __init__(
        self,
        secret: bytes,
        data_dir: Path,
        memory: Any | None = None,
        broadcast_port: int = P2P_BROADCAST_PORT,
        sync_port: int = P2P_SYNC_PORT,
        workspace: Any | None = None,
    ):
        self.enc_key, self.mac_key = _derive_keys(secret)
        self.origin_id = _origin_id(secret)
        self.data_dir = data_dir
        self.memory = memory
        self.workspace = workspace
        self.broadcast_port = broadcast_port
        self.sync_port = sync_port
        self.peers: dict[str, Peer] = {}
        self._seen_nonces: set[str] = set()
        self._nonce_window: int = 10000
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._aes = AESGCM(self.enc_key)

    # ------------------------------------------------------------------
    # Crypto
    # ------------------------------------------------------------------
    def _proof(self, frame: P2PFrame) -> str:
        mac = hmac.new(self.mac_key, frame.canonical(), hashlib.sha256)
        return mac.hexdigest()

    def _verify(self, frame: P2PFrame) -> bool:
        return hmac.compare_digest(self._proof(frame), frame.proof)

    def _encrypt(self, plaintext: bytes) -> tuple:
        nonce = os.urandom(12)
        ct = self._aes.encrypt(nonce, plaintext, None)
        return base64.b64encode(nonce).decode(), base64.b64encode(ct).decode()

    def _decrypt(self, nonce_b64: str, payload_b64: str) -> bytes | None:
        try:
            nonce = base64.b64decode(nonce_b64)
            ct = base64.b64decode(payload_b64)
            return self._aes.decrypt(nonce, ct, None)
        except Exception:  # noqa: BLE001 - catch-all wrapper
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self):
        if self._running:
            return
        self._running = True
        loop = asyncio.get_running_loop()

        # UDP broadcast listener for peer discovery.
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp_sock.setblocking(False)
        try:
            self._udp_sock.bind(("0.0.0.0", self.broadcast_port))
        except OSError as e:
            print(f"[p2p] could not bind broadcast port {self.broadcast_port}: {e}", flush=True)
            self._running = False
            return

        # TCP listener for sync payloads.
        self._tcp_server = await asyncio.start_server(
            self._handle_sync_client, host="0.0.0.0", port=self.sync_port
        )

        self._tasks = [
            loop.create_task(self._beacon_loop()),
            loop.create_task(self._prune_peers_loop()),
            loop.create_task(self._read_udp_loop()),
        ]
        print(f"[p2p] listening on udp {self.broadcast_port} / tcp {self.sync_port}", flush=True)

    def is_running(self) -> bool:
        return self._running

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
        if getattr(self, "_tcp_server", None):
            self._tcp_server.close()
            await self._tcp_server.wait_closed()
        if getattr(self, "_udp_sock", None):
            self._udp_sock.close()

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------
    async def _beacon_loop(self):
        while self._running:
            try:
                await self._send_beacon()
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                print(f"[p2p] beacon error: {e}", flush=True)
            await asyncio.sleep(P2P_BROADCAST_INTERVAL)

    async def _send_beacon(self):
        timestamp_ms = int(time.time() * 1000)
        # Include our TCP sync port and workspace/project context in the beacon
        # so peers can see the same project on the local mesh.
        workspace = ""
        workspace_summary = ""
        if self.workspace is not None:
            workspace = str(getattr(self.workspace, "path", ""))
            workspace_summary = getattr(self.workspace, "summary", lambda: "")() or ""
        plaintext = json.dumps({
            "sync_port": self.sync_port,
            "msg": "hello",
            "workspace": workspace,
            "workspace_summary": workspace_summary,
        }).encode()
        nonce_b64, payload_b64 = self._encrypt(plaintext)
        frame = P2PFrame(
            frame_type="beacon",
            origin_id=self.origin_id,
            timestamp_ms=timestamp_ms,
            nonce_b64=nonce_b64,
            payload_b64=payload_b64,
            proof="",
        )
        frame.proof = self._proof(frame)
        loop = asyncio.get_running_loop()
        await loop.sock_sendto(self._udp_sock, frame.to_bytes(), ("255.255.255.255", self.broadcast_port))

    async def _read_udp_loop(self):
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                data, addr = await loop.sock_recvfrom(self._udp_sock, 4096)
                await self._handle_udp(data, addr)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                print(f"[p2p] udp read error: {e}", flush=True)

    async def _handle_udp(self, data: bytes, addr):
        frame = P2PFrame.from_bytes(data)
        if frame is None:
            return
        if not self._verify(frame):
            print(f"[p2p] bad HMAC from {addr[0]}", flush=True)
            return
        if frame.origin_id == self.origin_id:
            return
        if not self._timestamp_fresh(frame.timestamp_ms):
            return
        if not self._nonce_fresh(frame.nonce_b64):
            return

        if frame.frame_type == "beacon":
            remote_sync_port = self.sync_port
            plaintext = self._decrypt(frame.nonce_b64, frame.payload_b64)
            if plaintext:
                try:
                    remote_sync_port = int(json.loads(plaintext.decode()).get("sync_port", self.sync_port))
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
                    print(f"[p2p] int failed: {e}", flush=True)
            self._remember_peer(addr[0], frame.origin_id, remote_sync_port)

    def _remember_peer(self, host: str, origin_id: str, sync_port: int):
        peer_id = f"{origin_id}@{host}"
        self.peers[peer_id] = Peer(
            host=host,
            port=sync_port,
            origin_id=origin_id,
            last_seen=time.time(),
        )

    def add_peer(self, host: str, port: int) -> str:
        """Manually add a static peer for testing or known neighbors."""
        peer_id = f"manual@{host}:{port}"
        self.peers[peer_id] = Peer(
            host=host,
            port=port,
            origin_id="manual",
            last_seen=time.time(),
        )
        return f"Added peer {host}:{port}."

    def remove_peer(self, peer_id: str) -> None:
        self.peers.pop(peer_id, None)
        for key in list(self.peers.keys()):
            if peer_id in key:
                self.peers.pop(key, None)

    async def _prune_peers_loop(self):
        while self._running:
            await asyncio.sleep(P2P_BEACON_TTL)
            now = time.time()
            stale = [k for k, p in self.peers.items() if now - p.last_seen > P2P_BEACON_TTL]
            for k in stale:
                del self.peers[k]

    async def _handle_sync_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            data = await reader.readline()
            if not data:
                return
            frame = P2PFrame.from_bytes(data)
            if frame is None or not self._verify(frame):
                return
            if frame.frame_type not in ("sync", "adapter") or not self._timestamp_fresh(frame.timestamp_ms) or not self._nonce_fresh(frame.nonce_b64):
                return
            plaintext = self._decrypt(frame.nonce_b64, frame.payload_b64)
            if plaintext is None:
                return
            if frame.frame_type == "sync":
                packet = json.loads(plaintext.decode())
                facts = packet.get("facts", [])
                if self.memory is not None:
                    for f in facts:
                        self.memory.remember(f, source="peer")
                    # Merge incoming project context if present and newer.
                    remote_ctx = packet.get("project_context")
                    if remote_ctx and isinstance(remote_ctx, dict):
                        local_ctx = self.memory._state.get("project_context", {})
                        remote_ts = remote_ctx.get("updated_at", "")
                        local_ts = local_ctx.get("updated_at", "")
                        if not local_ts or remote_ts > local_ts:
                            self.memory._state["project_context"] = remote_ctx
                            self.memory._save()
                writer.write(b'{"ok":true}\n')
                await writer.drain()
            elif frame.frame_type == "adapter":
                packet = json.loads(plaintext.decode())
                adapter_name = packet.get("name", " unnamed")
                adapters_dir = Path(packet.get("adapters_dir", str(self.data_dir / "lora_adapters")))
                target = adapters_dir / adapter_name
                try:
                    target.mkdir(parents=True, exist_ok=True)
                    import io
                    import zipfile
                    zip_bytes = base64.b64decode(packet.get("data_b64", ""))
                    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                        zf.extractall(target)
                    print(f"[p2p] received adapter '{adapter_name}' into {target}", flush=True)
                    writer.write(b'{"ok":true}\n')
                    await writer.drain()
                except Exception as e:  # noqa: BLE001 - catch-all wrapper
                    print(f"[p2p] adapter receive error: {e}", flush=True)
                    writer.write(json.dumps({"ok": False, "error": str(e)}).encode() + b"\n")
                    await writer.drain()
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError, LookupError) as e:
            print(f"[p2p] sync handler error: {e}", flush=True)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001,S110 - cleanup
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def sync_memory(self) -> str:
        if not self.memory:
            return "P2P sync unavailable: no memory store."
        if not self.peers:
            return "No peers discovered yet."
        facts = [f["text"] for f in self.memory._state.get("facts", [])[-20:]]
        if not facts:
            return "No facts to sync."

        workspace = ""
        workspace_summary = ""
        if self.workspace is not None:
            workspace = str(getattr(self.workspace, "path", ""))
            workspace_summary = getattr(self.workspace, "summary", lambda: "")() or ""
        project_context = self.memory._state.get("project_context", {}) if self.memory else {}
        plaintext = json.dumps({
            "origin_id": self.origin_id,
            "ts": int(time.time() * 1000),
            "facts": facts,
            "workspace": workspace,
            "workspace_summary": workspace_summary,
            "project_context": project_context,
        }).encode()
        nonce_b64, payload_b64 = self._encrypt(plaintext)
        frame = P2PFrame(
            frame_type="sync",
            origin_id=self.origin_id,
            timestamp_ms=int(time.time() * 1000),
            nonce_b64=nonce_b64,
            payload_b64=payload_b64,
            proof="",
        )
        frame.proof = self._proof(frame)

        sent = 0
        for peer in list(self.peers.values()):
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(peer.host, peer.port),
                    timeout=5,
                )
                writer.write(frame.to_bytes())
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line and b"ok" in line:
                    sent += 1
                writer.close()
                await writer.wait_closed()
            except (OSError, ValueError, TypeError) as e:
                print(f"[p2p] sync to {peer.host} failed: {e}", flush=True)
        return f"Synced {len(facts)} facts to {sent} peer(s)."

    def get_peers(self) -> str:
        if not self.peers:
            return "No peers on the local network."
        lines = [
            f"{p.origin_id} at {p.host}:{p.port} (last seen {datetime.datetime.fromtimestamp(p.last_seen, tz=datetime.timezone.utc).astimezone().isoformat()})"
            for p in self.peers.values()
        ]
        return "Discovered peers:\n" + "\n".join(lines)

    def send_adapter_sync(self, peer_origin_id: str, adapter_name: str, adapters_dir: Path) -> str:
        """Sync a LoRA adapter directory to a peer over the existing P2P TCP sync port."""
        if not self.peers:
            return "No peers discovered. Broadcast a beacon first or list peers."
        peer = None
        for p in self.peers.values():
            if p.origin_id == peer_origin_id or f"{p.origin_id}@{p.host}" == peer_origin_id:
                peer = p
                break
        if peer is None:
            return f"Peer {peer_origin_id} not found."

        adapter_path = Path(adapters_dir).expanduser() / adapter_name
        if not adapter_path.is_dir():
            return f"Adapter '{adapter_name}' not found at {adapter_path}"

        try:
            base = shutil.make_archive(str(adapter_path), 'zip', str(adapter_path))
            with open(base, "rb") as f:
                zip_bytes = f.read()
            Path(base).unlink(missing_ok=True)
        except (OSError, ValueError) as e:
            return f"Error packaging adapter: {e}"

        plaintext = json.dumps({
            "origin_id": self.origin_id,
            "name": adapter_name,
            "data_b64": base64.b64encode(zip_bytes).decode(),
        }).encode()
        nonce_b64, payload_b64 = self._encrypt(plaintext)
        frame = P2PFrame(
            frame_type="adapter",
            origin_id=self.origin_id,
            timestamp_ms=int(time.time() * 1000),
            nonce_b64=nonce_b64,
            payload_b64=payload_b64,
            proof="",
        )
        frame.proof = self._proof(frame)

        async def _send():
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(peer.host, peer.port),
                    timeout=30,
                )
                writer.write(frame.to_bytes())
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=60)
                writer.close()
                await writer.wait_closed()
                if line and b'"ok":true' in line:
                    return f"Sent adapter '{adapter_name}' to {peer.origin_id}"
                return f"Peer rejected adapter: {line.decode().strip()}"
            except (OSError, ValueError, TypeError) as e:
                return f"P2P adapter send failed: {e}"

        try:
            return asyncio.run(_send())
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            return f"P2P adapter send error: {e}"

    def list_local_adapters(self, adapters_dir: Path) -> str:
        d = Path(adapters_dir).expanduser()
        if not d.is_dir():
            return f"No adapters directory: {d}"
        names = [x.name for x in d.iterdir() if x.is_dir()]
        return "Local adapters: " + ", ".join(names) if names else "No local adapters."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _timestamp_fresh(self, timestamp_ms: int) -> bool:
        now = int(time.time() * 1000)
        return abs(now - timestamp_ms) <= 120_000

    def _nonce_fresh(self, nonce_b64: str) -> bool:
        # Simple replay window: keep the last N nonces.
        if nonce_b64 in self._seen_nonces:
            return False
        self._seen_nonces.add(nonce_b64)
        if len(self._seen_nonces) > self._nonce_window:
            # crude: clear old half
            keep = list(self._seen_nonces)[self._nonce_window // 2:]
            self._seen_nonces = set(keep)
        return True


# Global reference set by the main server so standalone `run_tool` can reach it.
_p2p_daemon: P2PDaemon | None = None


def set_p2p_daemon(daemon: P2PDaemon):
    global _p2p_daemon
    _p2p_daemon = daemon


def get_p2p_daemon() -> P2PDaemon | None:
    return _p2p_daemon
