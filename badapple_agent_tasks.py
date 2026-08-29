#!/usr/bin/env python3
"""Persistent, background agent task manager for Bad Apple.

The OS can queue multi-step goals, execute them in the MLX executor, and let
the dashboard or CLI observe, pause, and resume them. Tasks are stored in
/var/lib/bad_apple/agent_tasks/ so they survive daemon restarts.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class AgentStep:
    thought: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentStep:
        return cls(**data)


@dataclass
class AgentTask:
    task_id: str
    goal: str
    status: str  # queued, running, paused, completed, failed, cancelled
    max_steps: int = 10
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    steps: list[AgentStep] = field(default_factory=list)
    summary: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status,
            "max_steps": self.max_steps,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTask:
        steps = [AgentStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            task_id=data["task_id"],
            goal=data["goal"],
            status=data["status"],
            max_steps=data.get("max_steps", 10),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            steps=steps,
            summary=data.get("summary", ""),
            error=data.get("error", ""),
        )


class AgentTaskManager:
    """Store, queue, and observe autonomous agent tasks."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(
            data_dir or os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple")
        ).expanduser()
        self.tasks_dir = self.data_dir / "agent_tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, AgentTask] = {}
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[AgentTask], None]] = []
        self._load_all()

    def _is_safe_id(self, task_id: str) -> bool:
        return bool(SAFE_ID_RE.match(task_id)) and len(task_id) <= 64

    def _path_for(self, task_id: str) -> Path | None:
        if not self._is_safe_id(task_id):
            return None
        path = (self.tasks_dir / f"{task_id}.json").resolve()
        if not path.is_relative_to(self.tasks_dir.resolve()):
            return None
        return path

    def _load_all(self) -> None:
        if not self.tasks_dir.is_dir():
            return
        for p in self.tasks_dir.glob("*.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                task = AgentTask.from_dict(data)
                if not self._is_safe_id(task.task_id):
                    print(f"[agent_tasks] skipping invalid task id in {p}: {task.task_id}", flush=True)
                    continue
                if p.resolve() != self._path_for(task.task_id):
                    print(f"[agent_tasks] ignoring mismatched task file {p}", flush=True)
                    continue
                self._tasks[task.task_id] = task
            except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
                print(f"[agent_tasks] could not load {p}: {e}", flush=True)

    def _save(self, task: AgentTask) -> None:
        try:
            with self._lock:
                task.updated_at = time.time()
            path = self._path_for(task.task_id)
            if path is None:
                print(f"[agent_tasks] invalid task_id for save: {task.task_id}", flush=True)
                return
            with open(path, "w") as f:
                json.dump(task.to_dict(), f, indent=2, default=str)
        except OSError as e:
            print(f"[agent_tasks] could not save {task.task_id}: {e}", flush=True)

    def create(self, goal: str, max_steps: int = 10) -> AgentTask:
        task_id = uuid.uuid4().hex[:12]
        task = AgentTask(
            task_id=task_id,
            goal=goal,
            status="queued",
            max_steps=max(1, min(max_steps, 50)),
        )
        with self._lock:
            self._tasks[task_id] = task
        self._save(task)
        return task

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[AgentTask]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    def status(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.list()]

    def update_status(self, task_id: str, status: str, summary: str = "", error: str = "") -> AgentTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task.status = status
            if summary:
                task.summary = summary
            if error:
                task.error = error
        self._save(task)
        self._notify(task)
        return task

    def add_step(self, task_id: str, step: AgentStep) -> AgentTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task.steps.append(step)
            task.updated_at = time.time()
        self._save(task)
        self._notify(task)
        return task

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task.status in ("completed", "failed"):
                return False
            task.status = "cancelled"
            task.updated_at = time.time()
        self._save(task)
        self._notify(task)
        return True

    def pause(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != "running":
                return False
            task.status = "paused"
            task.updated_at = time.time()
        self._save(task)
        self._notify(task)
        return True

    def resume(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != "paused":
                return False
            task.status = "queued"
            task.updated_at = time.time()
        self._save(task)
        self._notify(task)
        return True

    def delete(self, task_id: str) -> bool:
        path = self._path_for(task_id)
        if path is None:
            return False
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return False
        with self._lock:
            self._tasks.pop(task_id, None)
        return True

    def register_callback(self, cb: Callable[[AgentTask], None]) -> None:
        with self._lock:
            self._callbacks.append(cb)

    def _notify(self, task: AgentTask) -> None:
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(task)
            except Exception as e:  # noqa: BLE001
                print(f"[agent_tasks] callback error: {e}", flush=True)
