#!/usr/bin/env python3
"""Tests for the minimal actor framework."""

import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import badapple_health_actor
import badapple_task_actor
from badapple_actor import Actor, Ask, Envelope, Inbox, Supervisor


class CounterActor(Actor):
    def __init__(self) -> None:
        super().__init__("counter")
        self.count = 0
        self.seen: list[Any] = []

    def receive(self, message: Any) -> int:
        if isinstance(message, Ask):
            if message.payload == "get":
                return self.count
            if message.payload == "ping":
                return "pong"
        if message == "inc":
            self.count += 1
            self.seen.append(message)
        if message == "dec":
            self.count -= 1
            self.seen.append(message)
        if message == "reset":
            self.count = 0
        return self.count


class ActorFrameworkTests(unittest.TestCase):
    def test_inbox_put_and_get(self) -> None:
        inbox = Inbox(maxsize=2)
        self.assertTrue(inbox.put(Envelope(message="a")))
        self.assertTrue(inbox.put(Envelope(message="b")))
        self.assertFalse(inbox.put(Envelope(message="c"), timeout=0.01))
        self.assertEqual(inbox.get(timeout=0.1).message, "a")
        self.assertEqual(inbox.get(timeout=0.1).message, "b")

    def test_counter_tell_and_ask(self) -> None:
        actor = CounterActor()
        actor.start()
        try:
            self.assertTrue(actor.tell("inc"))
            self.assertTrue(actor.tell("inc"))
            self.assertEqual(actor.ask("get"), 2)
            self.assertEqual(actor.ask("ping"), "pong")
            actor.tell("dec")
            self.assertEqual(actor.ask("get"), 1)
        finally:
            actor.stop()

    def test_actor_shutdown_is_clean(self) -> None:
        actor = CounterActor()
        actor.start()
        actor.stop(timeout=1.0)
        self.assertFalse(actor.is_alive())

    def test_backpressure_on_full_inbox(self) -> None:
        inbox = Inbox(maxsize=1)
        inbox.put(Envelope(message="x"))
        self.assertFalse(inbox.put(Envelope(message="y"), timeout=0.01))

    def test_supervisor_restarts_failed_actor(self) -> None:
        class FailingActor(Actor):
            def __init__(self) -> None:
                super().__init__("failing")
                self.restarts = 0

            def receive(self, message: Any) -> None:
                if message == "fail":
                    self._shutdown.set()
                    return
                if message == "restart_count":
                    self.restarts += 1

        sup = Supervisor(restart=True, max_restarts=2, window_s=10)
        counter: list[int] = [0]

        def factory() -> FailingActor:
            counter[0] += 1
            return FailingActor()

        sup.register("failing", factory)
        sup.start_all()
        try:
            time.sleep(0.1)
            actor = sup.get("failing")
            self.assertIsNotNone(actor)
            actor.tell("fail")
            time.sleep(1.5)
            self.assertGreaterEqual(counter[0], 2)
            new_actor = sup.get("failing")
            self.assertIsNotNone(new_actor)
            self.assertTrue(new_actor.is_alive())
        finally:
            sup.stop_all()

    def test_supervisor_stop_all(self) -> None:
        sup = Supervisor()
        actor = CounterActor()
        sup.register("counter", lambda: actor)
        sup.start_all()
        time.sleep(0.1)
        sup.stop_all()
        self.assertFalse(actor.is_alive())

    def test_health_actor_register_and_snapshot(self) -> None:
        actor = badapple_health_actor.HealthActor()
        actor.start()
        try:
            proxy = badapple_health_actor.HealthActorProxy(actor)
            proxy.register("test", "liveness", lambda: True)
            snap = proxy.snapshot()
            self.assertIn("test", snap.get("checks", {}))
        finally:
            actor.stop()

    def test_task_actor_create_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            actor = badapple_task_actor.TaskActor(Path(td))
            actor.start()
            try:
                proxy = badapple_task_actor.TaskActorProxy(actor)
                task = proxy.create("test goal", 5)
                self.assertIsNotNone(task)
                self.assertEqual(task.status, "queued")
                self.assertTrue(proxy.cancel(task.task_id))
                self.assertEqual(proxy.get(task.task_id).status, "cancelled")
            finally:
                actor.stop()


if __name__ == "__main__":
    unittest.main()
