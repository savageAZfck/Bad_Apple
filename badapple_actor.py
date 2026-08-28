#!/usr/bin/env python3
"""Minimal actor framework for Bad Apple.

Actors run in dedicated daemon threads, have a bounded inbox, and can be
supervised for restart. The framework is intentionally small and dependency-free
so it can be introduced into the existing monolith without pulling in an entire
actor library.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar


T = TypeVar("T")

DEFAULT_INBOX_SIZE = 1_000


class ActorShutdown:
    """Sentinel to request clean actor shutdown."""


@dataclass
class Ask:
    """A request that expects a reply."""

    payload: Any
    reply_to: queue.Queue = field(default_factory=queue.Queue)
    timeout: float = 10.0


@dataclass
class Envelope:
    """Message envelope with optional correlation and sender."""

    message: Any
    sender: str | None = None
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])


class Inbox:
    """Thread-safe bounded inbox with backpressure."""

    def __init__(self, maxsize: int = DEFAULT_INBOX_SIZE) -> None:
        self._queue: queue.Queue[Envelope] = queue.Queue(maxsize=maxsize)

    def put(self, envelope: Envelope, timeout: float | None = None) -> bool:
        try:
            self._queue.put(envelope, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float | None = None) -> Envelope | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def size(self) -> int:
        return self._queue.qsize()

    def done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()


class Actor:
    """Base actor class.

    Subclass must implement `receive(self, message)`.
    """

    def __init__(self, name: str | None = None, inbox: Inbox | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.inbox = inbox or Inbox()
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._running = False

    def receive(self, message: Any) -> Any:
        """Override to handle one message. Return value is ignored for tell."""
        raise NotImplementedError

    def _run(self) -> None:
        while not self._shutdown.is_set():
            envelope = self.inbox.get(timeout=0.1)
            if envelope is None:
                continue
            done = False
            try:
                if envelope.message is ActorShutdown:
                    self._shutdown.set()
                    done = True
                    break
                reply = self.receive(envelope.message)
                if isinstance(envelope.message, Ask):
                    try:
                        envelope.message.reply_to.put(reply, timeout=envelope.message.timeout)
                    except queue.Full:
                        pass
            except Exception as e:  # noqa: BLE001 - actor boundary
                print(f"[actor] {self.name} error handling {envelope.correlation_id}: {e}", flush=True)
            finally:
                if not done:
                    self.inbox.done()

    def start(self) -> None:
        if self._running:
            return
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._run, name=f"actor-{self.name}", daemon=True)
        self._thread.start()
        self._running = True

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        self._shutdown.set()
        self.inbox.put(Envelope(message=ActorShutdown, sender="supervisor"), timeout=timeout)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._running = False

    def tell(self, message: Any, sender: str | None = None, timeout: float = 1.0) -> bool:
        """Send a one-way message. Returns False if inbox is full."""
        return self.inbox.put(Envelope(message=message, sender=sender), timeout=timeout)

    def ask(self, message: Any, timeout: float = 10.0) -> Any | None:
        """Send a message and block for a reply."""
        req = Ask(payload=message, timeout=timeout)
        if not self.inbox.put(Envelope(message=req, sender="ask"), timeout=timeout):
            return None
        try:
            return req.reply_to.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())


class Supervisor:
    """Supervise a set of actors and restart them on failure."""

    def __init__(self, restart: bool = True, max_restarts: int = 5, window_s: float = 60.0) -> None:
        self.restart = restart
        self.max_restarts = max_restarts
        self.window_s = window_s
        self.actors: dict[str, tuple[type[Actor], Callable[[], Actor]]] = {}
        self.instances: dict[str, Actor] = {}
        self.restarts: dict[str, list[float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, name: str, factory: Callable[[], Actor]) -> None:
        self.actors[name] = (factory.__class__, factory)

    def start_all(self) -> None:
        for name, (_, factory) in self.actors.items():
            self._start(name, factory)
        self._thread = threading.Thread(target=self._watcher, name="actor-supervisor", daemon=True)
        self._thread.start()

    def _start(self, name: str, factory: Callable[[], Actor]) -> None:
        actor = factory()
        actor.name = name
        actor.start()
        self.instances[name] = actor

    def _watcher(self) -> None:
        while not self._stop.is_set():
            for name, actor in list(self.instances.items()):
                if not actor.is_alive():
                    now = time.time()
                    self.restarts.setdefault(name, [])
                    self.restarts[name] = [t for t in self.restarts[name] if now - t < self.window_s]
                    if not self.restart or len(self.restarts[name]) >= self.max_restarts:
                        print(f"[supervisor] {name} exceeded restart limit", flush=True)
                        continue
                    self.restarts[name].append(now)
                    _, factory = self.actors[name]
                    print(f"[supervisor] restarting {name}", flush=True)
                    self._start(name, factory)
            time.sleep(1.0)

    def stop_all(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for actor in self.instances.values():
            actor.stop(timeout=timeout)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def get(self, name: str) -> Actor | None:
        return self.instances.get(name)
