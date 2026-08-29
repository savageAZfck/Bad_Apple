#!/usr/bin/env python3
"""Actor wrapper for the Bad Apple P2P daemon.

The P2P daemon is an asyncio service, so this actor runs it on a dedicated
event-loop thread and dispatches async calls with
`asyncio.run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from pathlib import Path
from typing import Any

import badapple_actor
import badapple_lora
import badapple_p2p


WHITELIST = frozenset({
    "start",
    "stop",
    "is_running",
    "peers",
    "get_peers",
    "sync",
    "sync_memory",
    "add_peer",
    "remove_peer",
    "send_adapter",
    "send_adapter_sync",
    "list_adapters",
    "list_local_adapters",
    "p2p_models",
    "p2p_pull_model",
    "remote_models",
    "pull_model_manifest",
    "p2p_send_model",
    "p2p_receive_model",
    "send_model",
    "receive_model",
})


class P2PActor(badapple_actor.Actor):
    """Actor that owns a P2PDaemon and its asyncio event-loop thread."""

    def __init__(
        self,
        secret: bytes,
        data_dir: Path,
        memory: Any | None = None,
        workspace: Any | None = None,
        broadcast_port: int | None = None,
        sync_port: int | None = None,
        model_registry: Any | None = None,
    ) -> None:
        super().__init__("p2p")
        self._daemon = badapple_p2p.P2PDaemon(
            secret,
            data_dir,
            memory=memory,
            workspace=workspace,
            model_registry=model_registry,
            broadcast_port=broadcast_port or badapple_p2p.P2P_BROADCAST_PORT,
            sync_port=sync_port or badapple_p2p.P2P_SYNC_PORT,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._loop_thread: threading.Thread | None = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            try:
                tasks = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                for t in tasks:
                    t.cancel()
                if tasks:
                    self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                self._loop.close()
            except Exception:  # noqa: BLE001,S110 - cleanup
                pass

    def _run(self) -> None:
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="p2p-event-loop", daemon=True
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=5.0):
            print("[p2p_actor] event loop failed to start", flush=True)
            return
        super()._run()

    def _schedule(self, coro: Any, timeout: float = 30.0) -> Any:
        if self._loop is None or self._loop.is_closed() or not self._loop.is_running():
            return {"error": "P2P event loop is not running"}
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {"error": "P2P call timed out"}

    def stop(self, timeout: float = 5.0) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                self._schedule(self._daemon.stop(), timeout=timeout)
            except Exception:  # noqa: BLE001,S110 - cleanup
                pass
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:  # noqa: BLE001,S110 - cleanup
                pass
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=timeout)
        super().stop(timeout=timeout)

    def receive(self, message: Any) -> Any:
        if isinstance(message, badapple_actor.Ask):
            payload = message.payload
        else:
            payload = message
        if not isinstance(payload, dict):
            return None
        method = payload.get("method")
        if method not in WHITELIST:
            return {"error": f"method {method!r} not whitelisted"}
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        try:
            if method == "start":
                return self._schedule(self._daemon.start(), timeout=30.0)
            if method == "stop":
                return self._schedule(self._daemon.stop(), timeout=30.0)
            if method in ("sync", "sync_memory"):
                return self._schedule(self._daemon.sync_memory(), timeout=60.0)
            if method in ("peers", "get_peers"):
                return self._daemon.get_peers()
            if method in ("send_adapter", "send_adapter_sync"):
                peer_id = args[0] if args else kwargs.get("peer_id", "")
                adapter = args[1] if len(args) > 1 else kwargs.get("adapter", "")
                adapters_dir = args[2] if len(args) > 2 else kwargs.get("adapters_dir", badapple_lora.LORA_ADAPTERS_DIR)
                return self._daemon.send_adapter_sync(
                    str(peer_id), str(adapter), Path(adapters_dir)
                )
            if method in ("list_adapters", "list_local_adapters"):
                adapters_dir = args[0] if args else kwargs.get("adapters_dir", badapple_lora.LORA_ADAPTERS_DIR)
                return self._daemon.list_local_adapters(Path(adapters_dir))
            if method in ("p2p_models", "remote_models"):
                return self._daemon.remote_models()
            if method in ("p2p_pull_model", "pull_model_manifest"):
                peer_id = args[0] if args else kwargs.get("peer_id", "")
                model_id = args[1] if len(args) > 1 else kwargs.get("model_id", "")
                return self._schedule(
                    self._daemon.pull_model_manifest(str(peer_id), str(model_id)),
                    timeout=60.0,
                )
            if method in ("p2p_send_model", "send_model"):
                peer_id = args[0] if args else kwargs.get("peer_id", "")
                model_id = args[1] if len(args) > 1 else kwargs.get("model_id", "")
                return self._schedule(
                    self._daemon.send_model(str(peer_id), str(model_id)),
                    timeout=600.0,
                )
            if method in ("p2p_receive_model", "receive_model"):
                peer_id = args[0] if args else kwargs.get("peer_id", "")
                model_id = args[1] if len(args) > 1 else kwargs.get("model_id", "")
                return self._schedule(
                    self._daemon.receive_model(str(peer_id), str(model_id)),
                    timeout=600.0,
                )
            target = getattr(self._daemon, method, None)
            if target is None:
                return {"error": f"method {method!r} not found"}
            if callable(target):
                return target(*args, **kwargs)
            return target
        except Exception as e:  # noqa: BLE001 - actor boundary
            return {"error": str(e)}


class P2PActorProxy:
    """Synchronous proxy for the P2P actor."""

    def __init__(self, actor: P2PActor) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"P2PActorProxy has no attribute {name!r}")

        timeout = 600.0 if name in ("send_model", "receive_model", "p2p_send_model", "p2p_receive_model") else 60.0

        def _call(*args: Any, **kwargs: Any) -> Any:
            payload = {"method": name, "args": list(args), "kwargs": kwargs}
            return self._actor.ask(payload, timeout=timeout)

        return _call
