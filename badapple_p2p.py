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
import ipaddress
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

import badapple_model_provenance
import badapple_slicks

P2P_VERSION = 1
P2P_BROADCAST_PORT = int(os.environ.get("BADAPPLE_P2P_UDP_PORT", "9999"))
P2P_SYNC_PORT = int(os.environ.get("BADAPPLE_P2P_TCP_PORT", "10000"))
P2P_BROADCAST_INTERVAL = 30.0
P2P_BEACON_TTL = 120.0
P2P_RATE_LIMIT_INTERVAL = 5.0
P2P_MAX_BEACON_SIZE = 4096
# 1 MB allows encrypted model_chunk frames built from a 256 KB raw chunk.
P2P_MAX_SYNC_SIZE = 1024 * 1024
BADAPPLE_CHUNK_SIZE = 256 * 1024

FRAME_MODEL_REQUEST = "model_request"
FRAME_MODEL_CHUNK = "model_chunk"
FRAME_MODEL_DONE = "model_done"


def _is_local_peer_address(ip: str) -> bool:
    """Return True if `ip` is on a private, link-local, or loopback network.

    The P2P mesh's own threat model (see module docstring) is "only ever
    binds to local interfaces" -- but `host="0.0.0.0"` on the TCP listener
    and a UDP broadcast socket both actually accept traffic from *any*
    reachable address, not just the local network. HMAC/AES-GCM framing
    already stops an attacker without the shared SLICKS secret from
    forging a valid frame, but this check makes the code's behavior match
    its documented claim and rejects non-local traffic before spending any
    CPU on parsing or cryptography.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_private or addr.is_link_local or addr.is_loopback


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
        model_registry: Any | None = None,
    ):
        self.enc_key, self.mac_key = _derive_keys(secret)
        self._v2_identity = badapple_slicks.v2_available()
        if self._v2_identity:
            self.origin_id = badapple_slicks.v2_public_key_b64() or _origin_id(secret)
        else:
            self.origin_id = _origin_id(secret)
        self.data_dir = data_dir
        self.memory = memory
        self.workspace = workspace
        self.model_registry = model_registry
        self.broadcast_port = broadcast_port
        self.sync_port = sync_port
        self.peers: dict[str, Peer] = {}
        self._seen_nonces: set[str] = set()
        self._nonce_window: int = 10000
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._aes = AESGCM(self.enc_key)
        # Rate-limit incoming beacons and syncs per origin to avoid floods.
        self._rate_limit: dict[str, float] = {}
        # Optional allowlist/blocklist of origin public-key fingerprints or IDs.
        self._allowlist: set[str] = self._load_id_set(data_dir / "p2p_allowlist.txt")
        self._blocklist: set[str] = self._load_id_set(data_dir / "p2p_blocklist.txt")
        # Model manifests received from remote peers (peer_id -> [manifests]).
        self._remote_models: dict[str, list[dict[str, Any]]] = {}
        # Status of models received from peers (model_id -> info).
        self._received_models: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Peer identity lists
    # ------------------------------------------------------------------
    def _load_id_set(self, path: Path) -> set[str]:
        """Load a list of origin IDs (public keys or HMAC origin IDs)."""
        if not path.is_file():
            return set()
        try:
            return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        except OSError:
            return set()

    def _is_allowed(self, origin_id: str) -> bool:
        if origin_id in self._blocklist:
            return False
        if self._allowlist and origin_id not in self._allowlist:
            return False
        return True

    def _rate_limited(self, origin_id: str) -> bool:
        now = time.time()
        last = self._rate_limit.get(origin_id, 0.0)
        if now - last < P2P_RATE_LIMIT_INTERVAL:
            return True
        self._rate_limit[origin_id] = now
        return False

    # ------------------------------------------------------------------
    # Crypto
    # ------------------------------------------------------------------
    def _proof(self, frame: P2PFrame) -> str:
        if self._v2_identity:
            return badapple_slicks.v2_sign_message(frame.canonical()) or ""
        mac = hmac.new(self.mac_key, frame.canonical(), hashlib.sha256)
        return mac.hexdigest()

    def _verify(self, frame: P2PFrame) -> bool:
        if badapple_slicks.v2_public_key_b64() and len(frame.proof) > 64:
            return badapple_slicks.v2_verify_message(
                frame.canonical(), frame.proof, frame.origin_id
            )
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
        # TCP listener for sync payloads — start even if UDP discovery fails.
        # Binding to all interfaces (rather than a single known local IP) is
        # required so this works regardless of which interface the user's
        # LAN is actually on (Wi-Fi, Ethernet, etc.) -- the source-address
        # checks in _handle_udp/_handle_sync_client (_is_local_peer_address)
        # are what actually enforce "local interfaces only", not the bind
        # address itself.
        self._tcp_server = await asyncio.start_server(
            self._handle_sync_client,
            host="0.0.0.0",  # noqa: S104 - see _is_local_peer_address for the real enforcement
            port=self.sync_port,
            limit=P2P_MAX_SYNC_SIZE,
        )

        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (OSError, AttributeError):
            pass
        self._udp_sock.setblocking(False)
        try:
            self._udp_sock.bind(("0.0.0.0", self.broadcast_port))  # noqa: S104 - see _is_local_peer_address
        except OSError as e:
            print(f"[p2p] could not bind broadcast port {self.broadcast_port}: {e}", flush=True)

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
        if not _is_local_peer_address(addr[0]):
            print(f"[p2p] rejected non-local UDP source {addr[0]}", flush=True)
            return
        if len(data) > P2P_MAX_BEACON_SIZE:
            print(f"[p2p] oversized beacon from {addr[0]}", flush=True)
            return
        frame = P2PFrame.from_bytes(data)
        if frame is None:
            return
        if not self._is_allowed(frame.origin_id):
            print(f"[p2p] blocked origin {frame.origin_id[:16]}...", flush=True)
            return
        if self._rate_limited(frame.origin_id):
            return
        if not self._verify(frame):
            print(f"[p2p] bad proof from {addr[0]}", flush=True)
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
        peer_addr = writer.get_extra_info("peername")
        if peer_addr and not _is_local_peer_address(peer_addr[0]):
            print(f"[p2p] rejected non-local TCP connection from {peer_addr[0]}", flush=True)
            writer.close()
            return
        try:
            while True:
                result = await self._read_frame(reader)
                if result is None:
                    break
                frame, plaintext = result
                try:
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
                        remote_models = packet.get("models")
                        if remote_models and isinstance(remote_models, list):
                            peername = writer.get_extra_info("peername")
                            peer_host = peername[0] if peername and len(peername) > 0 else ""
                            peer_id = f"{frame.origin_id}@{peer_host}" if peer_host else frame.origin_id
                            self._remote_models[peer_id] = remote_models
                        writer.write(b'{"ok":true}\n')
                        await writer.drain()
                        break
                    if frame.frame_type == "pull_request":
                        packet = json.loads(plaintext.decode())
                        model_id = packet.get("model_id", "")
                        if not self.model_registry or not model_id:
                            await self._send_pull_error(writer, "The model could not be shared because the model name was missing.")
                            break
                        match = self._find_registry_model(model_id)
                        if match is None:
                            await self._send_pull_error(writer, f"Bad Apple could not find {model_id} in the local model list.")
                            break
                        manifest = self._provenance_manifest(model_id, match)
                        response = {"model": match, "manifest": manifest if manifest is not None else {}}
                        response_plaintext = json.dumps(response).encode()
                        nonce_b64, payload_b64 = self._encrypt(response_plaintext)
                        resp_frame = P2PFrame(
                            frame_type="pull_manifest",
                            origin_id=self.origin_id,
                            timestamp_ms=int(time.time() * 1000),
                            nonce_b64=nonce_b64,
                            payload_b64=payload_b64,
                            proof="",
                        )
                        resp_frame.proof = self._proof(resp_frame)
                        writer.write(resp_frame.to_bytes())
                        await writer.drain()
                        break
                    if frame.frame_type == "adapter":
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
                        break
                    if frame.frame_type == FRAME_MODEL_REQUEST:
                        await self._handle_model_request(frame, plaintext, reader, writer)
                        break
                    break
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError, LookupError) as e:
                    print(f"[p2p] sync handler error: {e}", flush=True)
                    break
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001,S110 - cleanup
                pass

    # ------------------------------------------------------------------
    # Model file transfer helpers
    # ------------------------------------------------------------------
    async def _read_frame(self, reader: asyncio.StreamReader) -> tuple[P2PFrame, bytes] | None:
        """Read, verify, and decrypt a single P2P frame."""
        try:
            data = await reader.readline()
        except Exception:
            return None
        if not data:
            return None
        if len(data) > P2P_MAX_SYNC_SIZE:
            print("[p2p] oversized sync payload", flush=True)
            return None
        frame = P2PFrame.from_bytes(data)
        if frame is None:
            return None
        if not self._is_allowed(frame.origin_id):
            print(f"[p2p] blocked origin {frame.origin_id[:16]}...", flush=True)
            return None
        if not self._verify(frame):
            print("[p2p] bad proof", flush=True)
            return None
        if not self._timestamp_fresh(frame.timestamp_ms) or not self._nonce_fresh(frame.nonce_b64):
            return None
        plaintext = self._decrypt(frame.nonce_b64, frame.payload_b64)
        if plaintext is None:
            return None
        return frame, plaintext

    def _build_frame(self, frame_type: str, plaintext: bytes) -> P2PFrame:
        """Build an encrypted and signed P2P frame."""
        nonce_b64, payload_b64 = self._encrypt(plaintext)
        frame = P2PFrame(
            frame_type=frame_type,
            origin_id=self.origin_id,
            timestamp_ms=int(time.time() * 1000),
            nonce_b64=nonce_b64,
            payload_b64=payload_b64,
            proof="",
        )
        frame.proof = self._proof(frame)
        return frame

    @staticmethod
    def _safe_model_name(model_id: str) -> str:
        """Return a filesystem-safe name for a model_id."""
        safe = model_id.replace("/", "--").replace("..", "")
        if not safe or safe.startswith(("/", "\\")):
            return "invalid"
        return safe

    def _list_model_files(self, path: Path) -> list[str]:
        """Return non-hidden, regular file paths relative to *path*."""
        files: list[str] = []
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return files
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name
            if name.startswith(".") or name == "desktop.ini" or name == "Thumbs.db":
                continue
            try:
                resolved = p.resolve()
                if resolved.is_file():
                    rel = str(p.relative_to(root))
                    if ".." in rel:
                        continue
                    files.append(rel)
            except (OSError, ValueError):
                continue
        return sorted(files)

    async def _send_model_ok(self, writer: asyncio.StreamWriter, message: str = "ok") -> None:
        payload = json.dumps({"ok": True, "message": message}).encode()
        frame = self._build_frame("ok", payload)
        writer.write(frame.to_bytes())
        await writer.drain()

    async def _send_model_error(self, writer: asyncio.StreamWriter, message: str) -> None:
        payload = json.dumps({"error": message}).encode()
        frame = self._build_frame("error", payload)
        writer.write(frame.to_bytes())
        await writer.drain()

    async def _handle_model_request(
        self,
        frame: P2PFrame,
        plaintext: bytes,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming model_request frame (push or pull)."""
        try:
            packet = json.loads(plaintext.decode())
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError, LookupError):
            await self._send_model_error(writer, "The request could not be read.")
            return
        model_id = str(packet.get("model_id", ""))
        direction = packet.get("direction")
        if not model_id or direction not in ("push", "pull"):
            await self._send_model_error(writer, "The request needs a model name and a valid direction.")
            return
        if direction == "pull":
            match = self._find_registry_model(model_id)
            if match is None:
                await self._send_model_error(writer, f"Bad Apple could not find {model_id} in the local model list.")
                return
            path = Path(match.get("path", ""))
            manifest = self._provenance_manifest(model_id, match)
            if not path or not path.is_dir():
                await self._send_model_error(writer, f"The model files for {model_id} are missing.")
                return
            await self._stream_model_files(model_id, path, manifest, writer)
            return
        # Push: tell the sender we are ready and then receive the file stream.
        await self._send_model_ok(writer, "ready")
        await self._receive_model_stream(reader, model_id, writer)

    async def _stream_model_files(
        self,
        model_id: str,
        path: Path,
        manifest: dict[str, Any] | None,
        writer: asyncio.StreamWriter,
    ) -> tuple[bool, str]:
        """Send model files as encrypted model_chunk frames, finishing with model_done."""
        files = self._list_model_files(path)
        if not files:
            return False, f"There are no files to send for model {model_id!r}."
        for rel in files:
            full = path / rel
            try:
                file_size = full.stat().st_size
            except OSError as e:
                return False, f"Could not read file {rel}: {e}"
            with open(full, "rb") as f:
                offset = 0
                while True:
                    chunk = f.read(BADAPPLE_CHUNK_SIZE)
                    if not chunk:
                        break
                    is_last = (offset + len(chunk)) >= file_size
                    payload = json.dumps({
                        "model_id": model_id,
                        "filename": rel,
                        "offset": offset,
                        "data_b64": base64.b64encode(chunk).decode(),
                        "is_last": is_last,
                    }).encode()
                    frame = self._build_frame(FRAME_MODEL_CHUNK, payload)
                    writer.write(frame.to_bytes())
                    await writer.drain()
                    offset += len(chunk)
        done_payload = json.dumps({
            "model_id": model_id,
            "manifest": manifest if manifest is not None else {},
        }).encode()
        done_frame = self._build_frame(FRAME_MODEL_DONE, done_payload)
        writer.write(done_frame.to_bytes())
        await writer.drain()
        return True, ""

    async def _receive_model_stream(
        self,
        reader: asyncio.StreamReader,
        model_id: str,
        writer: asyncio.StreamWriter | None = None,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Receive model_chunk and model_done frames and finalize the transfer."""
        safe_id = self._safe_model_name(model_id)
        incoming = self.data_dir / "incoming_models" / safe_id
        try:
            if incoming.exists():
                if incoming.is_dir():
                    shutil.rmtree(incoming)
                else:
                    incoming.unlink()
            incoming.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"Could not prepare the incoming model directory: {e}"
            if writer:
                await self._send_model_error(writer, msg)
            return False, msg, None

        open_files: dict[str, Any] = {}
        manifest: dict[str, Any] | None = None
        try:
            while True:
                try:
                    result = await asyncio.wait_for(self._read_frame(reader), timeout=60.0)
                except asyncio.TimeoutError:
                    msg = "The model transfer timed out while waiting for the next chunk."
                    if writer:
                        await self._send_model_error(writer, msg)
                    return False, msg, None
                if result is None:
                    msg = "The connection closed before the model transfer finished."
                    if writer:
                        await self._send_model_error(writer, msg)
                    return False, msg, None
                frame, plaintext = result
                if frame.frame_type == "error":
                    try:
                        packet = json.loads(plaintext.decode())
                        err = packet.get("error", "The peer returned an error.")
                    except Exception:
                        err = "The peer returned an error."
                    return False, err, None
                if frame.frame_type == FRAME_MODEL_CHUNK:
                    packet = json.loads(plaintext.decode())
                    filename = str(packet.get("filename", ""))
                    if not filename:
                        continue
                    if ".." in filename or filename.startswith(("/", "\\")):
                        msg = f"The chunk contains an invalid file name: {filename}"
                        if writer:
                            await self._send_model_error(writer, msg)
                        return False, msg, None
                    target = (incoming / filename).resolve()
                    if not target.is_relative_to(incoming.resolve()):
                        msg = "The chunk file path is not inside the incoming model directory."
                        if writer:
                            await self._send_model_error(writer, msg)
                        return False, msg, None
                    target.parent.mkdir(parents=True, exist_ok=True)
                    offset = int(packet.get("offset", 0))
                    data = base64.b64decode(packet.get("data_b64", ""))
                    if filename not in open_files:
                        open_files[filename] = open(target, "w+b")
                    f = open_files[filename]
                    f.seek(offset)
                    f.write(data)
                    if packet.get("is_last"):
                        f.flush()
                        f.close()
                        del open_files[filename]
                elif frame.frame_type == FRAME_MODEL_DONE:
                    packet = json.loads(plaintext.decode())
                    manifest = packet.get("manifest")
                    break
                else:
                    msg = f"Unexpected frame type during model transfer: {frame.frame_type}"
                    if writer:
                        await self._send_model_error(writer, msg)
                    return False, msg, None
        finally:
            for f in open_files.values():
                f.close()

        ok, msg, result = self._finalize_received_model(model_id, incoming, manifest)
        if writer:
            if ok:
                await self._send_model_ok(writer, msg)
            else:
                await self._send_model_error(writer, msg)
        return ok, msg, result

    def _finalize_received_model(
        self,
        model_id: str,
        incoming: Path,
        manifest: dict[str, Any] | None,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Move a received model into the registry and record its provenance."""
        safe_id = self._safe_model_name(model_id)
        final = self.data_dir / "models" / safe_id
        try:
            if final.exists():
                if final.is_dir():
                    shutil.rmtree(final)
                else:
                    final.unlink()
            shutil.move(str(incoming), str(final))
        except OSError as e:
            return False, f"Could not move the model into place: {e}", None

        result: dict[str, Any] = {}
        if self.model_registry is not None and hasattr(self.model_registry, "add_model"):
            try:
                record = self.model_registry.add_model(str(final), model_id)
                result["registry"] = record
            except Exception as e:  # noqa: BLE001
                return False, f"Could not add the model to the registry: {e}", None
        else:
            try:
                provenance = badapple_model_provenance.ModelProvenance(self.data_dir)
                record = provenance.record(model_id, model_id, str(final))
                result["provenance"] = record
            except Exception as e:  # noqa: BLE001
                return False, f"Could not record the provenance manifest: {e}", None

        self._received_models[model_id] = {
            "model_id": model_id,
            "path": str(final),
            "status": "received",
        }
        return True, f"Model {model_id} received and recorded.", result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _local_models_for_sync(self) -> list[dict[str, Any]] | None:
        """Return a minimal, P2P-safe slice of the local model registry."""
        if self.model_registry is None:
            return None
        if hasattr(self.model_registry, "get_manifests_for_p2p") and callable(self.model_registry.get_manifests_for_p2p):
            models = self.model_registry.get_manifests_for_p2p()
        else:
            models = getattr(self.model_registry, "_state", {}).get("models", [])
        if not models:
            return []
        fields = ("id", "size_gb", "quantization", "architecture", "provenance", "signature")
        return [{k: m[k] for k in fields if k in m} for m in models]

    async def sync_memory(self) -> str:
        if not self.memory:
            return "P2P sync is not ready because the memory store is not available."
        if not self.peers:
            return "No peers discovered yet."
        facts = [f["text"] for f in self.memory._state.get("facts", [])[-20:]]
        if not facts:
            return "There is nothing to share right now."

        workspace = ""
        workspace_summary = ""
        if self.workspace is not None:
            workspace = str(getattr(self.workspace, "path", ""))
            workspace_summary = getattr(self.workspace, "summary", lambda: "")() or ""
        project_context = self.memory._state.get("project_context", {}) if self.memory else {}
        payload: dict[str, Any] = {
            "origin_id": self.origin_id,
            "ts": int(time.time() * 1000),
            "facts": facts,
            "workspace": workspace,
            "workspace_summary": workspace_summary,
            "project_context": project_context,
        }
        local_models = self._local_models_for_sync()
        if local_models is not None:
            payload["models"] = local_models
        plaintext = json.dumps(payload).encode()
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
                    asyncio.open_connection(peer.host, peer.port, limit=P2P_MAX_SYNC_SIZE),
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
        if sent == 0:
            return f"Shared {len(facts)} memory fact(s) with 0 peer(s). No peers were reachable. Make sure P2P is turned on and the other Mac is nearby."
        if sent == 1:
            return f"Shared {len(facts)} memory fact(s) with 1 peer(s)."
        return f"Shared {len(facts)} memory fact(s) with {sent} peer(s)."

    def get_peers(self) -> str:
        if not self.peers:
            return "No peers on the local network. Make sure P2P is turned on and the other Mac is nearby."
        lines = [
            f"{p.origin_id} at {p.host}:{p.port} (last seen {datetime.datetime.fromtimestamp(p.last_seen, tz=datetime.timezone.utc).astimezone().isoformat()})"
            for p in self.peers.values()
        ]
        return "Discovered peers:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Model manifest gossip
    # ------------------------------------------------------------------
    def _find_registry_model(self, model_id: str) -> dict[str, Any] | None:
        """Look up a model in the optional local model registry."""
        if self.model_registry is None:
            return None
        if hasattr(self.model_registry, "get_manifests_for_p2p") and callable(self.model_registry.get_manifests_for_p2p):
            models = self.model_registry.get_manifests_for_p2p()
        else:
            models = getattr(self.model_registry, "_state", {}).get("models", [])
        return next((m for m in models if m.get("id") == model_id), None)

    def _provenance_manifest(self, model_id: str, model: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Return the provenance manifest JSON for a model, if available."""
        if model is not None and isinstance(model, dict) and "manifest" in model:
            return model["manifest"]
        if self.model_registry is None:
            return None
        provenance = getattr(self.model_registry, "provenance", None)
        if provenance is None:
            return None
        try:
            path = provenance._manifest_path(model_id)
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - manifest lookup best-effort
            return None

    def _resolve_peer(self, peer_id: str) -> Peer | None:
        """Resolve a peer by id, origin_id, or host:port."""
        for key, p in self.peers.items():
            if (key == peer_id
                or p.origin_id == peer_id
                or f"{p.origin_id}@{p.host}" == peer_id
                or f"{p.host}:{p.port}" == peer_id):
                return p
        return None

    async def _send_pull_error(self, writer: asyncio.StreamWriter, message: str) -> None:
        """Respond to a pull_request with an encrypted error frame."""
        plaintext = json.dumps({"error": message}).encode()
        nonce_b64, payload_b64 = self._encrypt(plaintext)
        frame = P2PFrame(
            frame_type="error",
            origin_id=self.origin_id,
            timestamp_ms=int(time.time() * 1000),
            nonce_b64=nonce_b64,
            payload_b64=payload_b64,
            proof="",
        )
        frame.proof = self._proof(frame)
        writer.write(frame.to_bytes())
        await writer.drain()

    def remote_models(self) -> dict[str, list[dict[str, Any]]]:
        """Return model manifests received from remote peers."""
        return self._remote_models

    async def pull_model_manifest(self, peer_id: str, model_id: str) -> dict[str, Any]:
        """Request a model's provenance manifest from a peer."""
        peer = self._resolve_peer(peer_id)
        if peer is None:
            return {"error": "Bad Apple could not find that peer on your local network. Make sure P2P is turned on and the other Mac is nearby."}

        plaintext = json.dumps({"model_id": model_id}).encode()
        nonce_b64, payload_b64 = self._encrypt(plaintext)
        frame = P2PFrame(
            frame_type="pull_request",
            origin_id=self.origin_id,
            timestamp_ms=int(time.time() * 1000),
            nonce_b64=nonce_b64,
            payload_b64=payload_b64,
            proof="",
        )
        frame.proof = self._proof(frame)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.host, peer.port, limit=P2P_MAX_SYNC_SIZE),
                timeout=5,
            )
            writer.write(frame.to_bytes())
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            writer.close()
            await writer.wait_closed()

            if not line:
                return {"error": "The other Mac did not answer. Make sure it is still nearby and P2P is on."}
            resp = P2PFrame.from_bytes(line)
            if resp is None:
                return {"error": "The other Mac sent an unreadable response."}
            if not self._is_allowed(resp.origin_id) or not self._verify(resp):
                return {"error": "The other Mac could not be verified. It may not be trusted."}
            if not self._timestamp_fresh(resp.timestamp_ms) or not self._nonce_fresh(resp.nonce_b64):
                return {"error": "The response from the other Mac was too old or already used."}
            resp_plaintext = self._decrypt(resp.nonce_b64, resp.payload_b64)
            if resp_plaintext is None:
                return {"error": "Bad Apple could not read the response from the other Mac."}
            packet = json.loads(resp_plaintext.decode())
            if resp.frame_type == "error":
                return {"error": packet.get("error", "The other Mac returned an error.")}
            if resp.frame_type != "pull_manifest":
                return {"error": "The other Mac sent an unexpected response."}

            manifest = packet.get("manifest")
            if manifest and isinstance(manifest, dict):
                peername = writer.get_extra_info("peername")
                peer_host = peername[0] if peername and len(peername) > 0 else ""
                remote_peer_id = f"{resp.origin_id}@{peer_host}" if peer_host else resp.origin_id
                self._remote_models.setdefault(remote_peer_id, []).append(manifest)
            return packet
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            return {"error": f"pull failed: {e}"}

    async def send_model(self, peer_id: str, model_id: str) -> str:
        """Send a local model (weights + manifest) to a peer."""
        peer = self._resolve_peer(peer_id)
        if peer is None:
            return f"Bad Apple could not find peer {peer_id!r} on your local network."
        if self.model_registry is None:
            return "No local model registry is available."
        model = self._find_registry_model(model_id)
        if model is None:
            return f"Model {model_id!r} is not in the local registry."
        path = Path(model.get("path", ""))
        if not path.is_dir():
            return f"The model files for {model_id!r} are missing at {path}."
        manifest = self._provenance_manifest(model_id, model)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.host, peer.port, limit=P2P_MAX_SYNC_SIZE),
                timeout=10,
            )
        except Exception as e:  # noqa: BLE001
            return f"Could not connect to the peer: {e}"

        request_payload = json.dumps({"model_id": model_id, "direction": "push"}).encode()
        request_frame = self._build_frame(FRAME_MODEL_REQUEST, request_payload)
        writer.write(request_frame.to_bytes())
        await writer.drain()

        try:
            result = await asyncio.wait_for(self._read_frame(reader), timeout=10)
        except asyncio.TimeoutError:
            writer.close()
            await writer.wait_closed()
            return "The peer did not respond to the send request."
        if result is None:
            writer.close()
            await writer.wait_closed()
            return "The peer closed the connection before accepting the model."
        frame, plaintext = result
        if frame.frame_type == "error":
            try:
                packet = json.loads(plaintext.decode())
                err = packet.get("error", "The peer refused the model.")
            except Exception:
                err = "The peer refused the model."
            writer.close()
            await writer.wait_closed()
            return f"The peer refused the model: {err}"
        if frame.frame_type != "ok":
            writer.close()
            await writer.wait_closed()
            return f"The peer sent an unexpected response: {frame.frame_type}"

        try:
            ok, msg = await self._stream_model_files(model_id, path, manifest, writer)
            if not ok:
                return msg

            try:
                final = await asyncio.wait_for(self._read_frame(reader), timeout=30)
            except asyncio.TimeoutError:
                final = None
            if final is not None:
                final_frame, final_text = final
                if final_frame.frame_type == "error":
                    try:
                        packet = json.loads(final_text.decode())
                        err = packet.get("error", "The peer reported an error after the transfer.")
                    except Exception:
                        err = "The peer reported an error after the transfer."
                    return f"Transfer finished with an error: {err}"

            return f"Sent model {model_id} to {peer.origin_id}."
        except Exception as e:  # noqa: BLE001
            return f"Could not finish sending the model: {e}"
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def receive_model(self, peer_id: str = "", model_id: str = "") -> dict[str, Any] | str:
        """Receive a model from a peer, or return the status of recent receives."""
        if not peer_id and not model_id:
            if not self._received_models:
                return {"status": "ready", "message": "No models have been received yet."}
            return {"status": "ready", "received": list(self._received_models.values())}
        if not peer_id or not model_id:
            return {"error": "Both peer_id and model_id are required for a directed receive."}

        peer = self._resolve_peer(peer_id)
        if peer is None:
            return {"error": f"Bad Apple could not find peer {peer_id!r} on your local network."}

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.host, peer.port, limit=P2P_MAX_SYNC_SIZE),
                timeout=10,
            )
        except Exception as e:  # noqa: BLE001
            return {"error": f"Could not connect to the peer: {e}"}

        request_payload = json.dumps({"model_id": model_id, "direction": "pull"}).encode()
        request_frame = self._build_frame(FRAME_MODEL_REQUEST, request_payload)
        writer.write(request_frame.to_bytes())
        await writer.drain()

        try:
            ok, msg, result = await self._receive_model_stream(reader, model_id, writer=None)
            if not ok:
                return {"error": msg}
            return {"status": "received", "message": msg, "model_id": model_id, "result": result}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Could not finish receiving the model: {e}"}
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

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
            return "Bad Apple could not find that peer on your local network. Make sure P2P is turned on and the other Mac is nearby."

        adapter_path = Path(adapters_dir).expanduser() / adapter_name
        if not adapter_path.is_dir():
            return f"Bad Apple could not find an adapter called '{adapter_name}'."

        try:
            base = shutil.make_archive(str(adapter_path), "zip", str(adapter_path))
            with open(base, "rb") as f:
                zip_bytes = f.read()
            Path(base).unlink(missing_ok=True)
        except (OSError, ValueError) as e:
            return f"Bad Apple could not package the adapter: {e}"

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
                    asyncio.open_connection(peer.host, peer.port, limit=P2P_MAX_SYNC_SIZE),
                    timeout=30,
                )
                writer.write(frame.to_bytes())
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=60)
                writer.close()
                await writer.wait_closed()
                if line and b'"ok":true' in line:
                    return f"Sent the '{adapter_name}' adapter to {peer.origin_id}."
                return f"The other Mac did not accept the adapter: {line.decode().strip()}"
            except (OSError, ValueError, TypeError) as e:
                return f"Could not send the adapter: {e}"

        try:
            return asyncio.run(_send())
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            return f"Could not send the adapter: {e}"

    def list_local_adapters(self, adapters_dir: Path) -> str:
        d = Path(adapters_dir).expanduser()
        if not d.is_dir():
            return "Bad Apple could not find the adapters folder."
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
