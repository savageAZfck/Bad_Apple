#!/usr/bin/env python3
"""Long-term memory graph for Bad Apple.

Stores facts, entities, relations, episodic summaries, and project context.
Embeds facts with a provided encoder so retrieval is semantic, not just keyword
matching. Split out from badapple_extras.py to reduce the monolith.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np


class MemoryGraph:
    """A simple, local, semantic memory graph for the user."""

    MAX_FACTS = 200
    MAX_EPISODES = 50
    MAX_ENTITIES = 100
    MAX_WORKFLOWS = 50

    ENTITY_PATTERNS = [
        re.compile(r"\bmy ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b"),
        re.compile(r"\bmy ([a-z]+(?:\s+[a-z]+){0,2}) is\b"),
        re.compile(r"\bi (?:like|love|prefer|hate) ([^,.!?:;]+)", re.IGNORECASE),
        re.compile(r"\bremember that ([^,.!?:;]+)", re.IGNORECASE),
        re.compile(r"\bmy name is ([A-Z][a-zA-Z]+)\b", re.IGNORECASE),
    ]

    def __init__(self, data_dir: Path, encoder: Any | None = None):
        self.data_dir = data_dir
        self.memory_file = data_dir / "memory_graph.json"
        self.encoder = encoder
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "facts": [],
            "entities": [],
            "relations": [],
            "episodes": [],
            "workflows": [],
            "project_context": {},
        }
        self._load()

    def _load(self):
        if not self.memory_file.is_file():
            return
        try:
            with open(self.memory_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._state.update(data)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
            print(f"[memory] could not load: {e}", flush=True)

    def _save(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.memory_file.with_name(f".{self.memory_file.name}.{os.getpid()}.tmp")
            with self._lock, open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.memory_file)
        except (TypeError, ValueError, OSError) as e:
            print(f"[memory] could not save: {e}", flush=True)

    def _embed(self, text: str) -> list[float] | None:
        if self.encoder is None:
            return None
        try:
            return self.encoder([text])[0].tolist()
        except (LookupError, TypeError, ValueError) as e:
            print(f"[memory] embedding failed: {e}", flush=True)
            return None

    def _extract_fact(self, text: str) -> str | None:
        # Keep just the first sentence; must be a statement about the user.
        low = text.lower()
        if not any(k in low for k in ("my ", "i like", "i love", "i prefer", "i hate", "remember that", "my name is")):
            return None
        sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
        if 5 < len(sentence) < 240:
            return sentence
        return None

    def _extract_entities(self, text: str):
        for pat in self.ENTITY_PATTERNS:
            for m in pat.finditer(text):
                entity = m.group(1).strip().lower()
                if len(entity) > 1 and entity not in self._state["entities"]:
                    self._state["entities"].append(entity)

    def _trim(self):
        self._state["facts"] = self._state["facts"][-self.MAX_FACTS:]
        self._state["episodes"] = self._state["episodes"][-self.MAX_EPISODES:]
        self._state["entities"] = self._state["entities"][-self.MAX_ENTITIES:]
        self._state["workflows"] = self._state["workflows"][-self.MAX_WORKFLOWS:]

    def remember(self, text: str, source: str = "user"):
        """Extract and store a fact from user or assistant text."""
        fact = self._extract_fact(text)
        if not fact:
            return
        if any(f["text"] == fact for f in self._state["facts"]):
            return
        embedding = self._embed(fact)
        self._state["facts"].append({
            "text": fact,
            "source": source,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "embedding": embedding,
        })
        self._extract_entities(fact)
        self._trim()
        self._save()

    def add_episode(self, user_text: str, assistant_text: str, context: dict[str, Any] | None = None):
        """Store a brief episodic record of a turn."""
        self._state["episodes"].append({
            "user": user_text[:300],
            "assistant": assistant_text[:300],
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "context": context or {},
            "embedding": self._embed(user_text[:300]),
        })
        self._trim()
        self._save()

    def add_relation(self, subject: str, relation: str, target: str):
        subject, relation, target = subject.lower().strip(), relation.lower().strip(), target.lower().strip()
        if not all((subject, relation, target)):
            return
        triple = {"subject": subject, "relation": relation, "target": target}
        if triple not in self._state["relations"]:
            self._state["relations"].append(triple)
            self._save()

    def learn_workflow(self, name: str, trigger: str, steps: list[dict[str, Any]]) -> str:
        name, trigger = name.strip(), trigger.strip()
        if not name or not trigger or not steps:
            return "Workflow name, trigger, and at least one step are required."
        if any(not isinstance(step, dict) or not step.get("tool") for step in steps):
            return "Every workflow step must contain a tool name."
        workflow = {
            "name": name,
            "trigger": trigger,
            "steps": steps,
            "enabled": False,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "embedding": self._embed(trigger),
        }
        self._state["workflows"] = [item for item in self._state["workflows"] if item.get("name") != name]
        self._state["workflows"].append(workflow)
        self._trim()
        self._save()
        return f"Learned workflow '{name}' in disabled review mode."

    def set_workflow_enabled(self, name: str, enabled: bool) -> str:
        for workflow in self._state["workflows"]:
            if workflow.get("name") == name:
                workflow["enabled"] = bool(enabled)
                self._save()
                return f"Workflow '{name}' enabled={bool(enabled)}."
        return f"Workflow '{name}' not found."

    def workflows(self) -> list[dict[str, Any]]:
        return [dict(workflow) for workflow in self._state["workflows"]]

    def search(self, query: str, k: int = 3) -> list[str]:
        """Return the most relevant fact and episode texts for a query."""
        if not (self._state["facts"] or self._state["episodes"]):
            return []

        low = query.lower()
        # Direct fact patterns should always win over episode recall.
        direct_patterns = [
            (r"\bmy name\b", r"\bmy name is\b"),
            (r"\bmy (?:favorite|favourite)\b", r"\bmy (?:favorite|favourite)\s+\w+\s+is\b"),
            (r"\bwhat do i (?:like|love|prefer|hate)\b", r"\bi (?:like|love|prefer|hate)\b"),
            (r"\bwhat do i\b", r"\bmy\s+\w+\s+is\b"),
        ]
        for query_pat, fact_pat in direct_patterns:
            if re.search(query_pat, low):
                for f in self._state["facts"]:
                    if re.search(fact_pat, f.get("text", ""), re.IGNORECASE):
                        return [f.get("text", "")]

        combined = []
        for f in self._state["facts"]:
            combined.append(("fact", f.get("text", ""), f.get("embedding")))
        for e in self._state["episodes"]:
            # Treat the user side of an episode as a memory target.
            combined.append(("episode", f"You previously asked: {e['user']}", e.get("embedding")))

        if self.encoder and any(emb for _, _, emb in combined):
            try:
                q_emb = self.encoder([query])[0]
                q_norm = np.linalg.norm(q_emb)
                scored = []
                for _, text, emb in combined:
                    if not emb:
                        continue
                    f_emb = np.array(emb)
                    f_norm = np.linalg.norm(f_emb)
                    if f_norm == 0:
                        continue
                    sim = float((q_emb @ f_emb) / (q_norm * f_norm))
                    scored.append((sim, text))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [t for _, t in scored[:k]]
            except (LookupError, TypeError, ValueError) as e:
                print(f"[memory] semantic search failed: {e}", flush=True)

        # Fallback keyword search.
        q_words = set(w for w in re.findall(r"\b\w+\b", query.lower()) if len(w) > 2)
        scored = []
        for _, text, _ in combined:
            text_words = set(w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 2)
            overlap = len(q_words & text_words)
            if overlap:
                scored.append((overlap, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    def context_for_prompt(self, query: str) -> str:
        relevant = self.search(query, k=5)
        if not relevant:
            return ""
        return "Things you remember about the user and past turns:\n" + "\n".join(f"- {r}" for r in relevant)

    def set_project_context(self, name: str, description: str, goals: list[str] | None = None, tags: list[str] | None = None) -> str:
        """Store a long-horizon project profile for the user."""
        self._state["project_context"] = {
            "name": name,
            "description": description,
            "goals": goals or [],
            "tags": tags or [],
            "updated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
        self._save()
        return f"Project context set: {name}."

    def get_project_context(self) -> str:
        """Return the project context as a prompt-ready string."""
        ctx = self._state.get("project_context", {})
        if not ctx:
            return ""
        goals = "\n- " + "\n- ".join(ctx.get("goals", [])) if ctx.get("goals") else ""
        tags = "\nTags: " + ", ".join(ctx.get("tags", [])) if ctx.get("tags") else ""
        return (
            f"Active project: {ctx.get('name', '')}\n"
            f"Description: {ctx.get('description', '')}{goals}{tags}"
        )

    def get_summary(self) -> str:
        return (
            f"Memory graph: {len(self._state['facts'])} facts, "
            f"{len(self._state['entities'])} entities, "
            f"{len(self._state['relations'])} relations, "
            f"{len(self._state['episodes'])} episodes, "
            f"{len(self._state['workflows'])} workflows."
        )

    def consolidate(self) -> str:
        """Offline dream pass: deduplicate facts, prune orphaned entities, refresh embeddings."""
        with self._lock:
            # Deduplicate facts by text (keep newest).
            seen: set = set()
            unique = []
            for f in reversed(self._state["facts"]):
                text = f.get("text", "").strip()
                if text and text not in seen:
                    seen.add(text)
                    unique.append(f)
            self._state["facts"] = list(reversed(unique))

            # Re-embed stale facts.
            for f in self._state["facts"]:
                if not f.get("embedding") and self.encoder:
                    f["embedding"] = self._embed(f["text"])

            # Prune entities no longer referenced by any fact.
            referenced = set()
            for f in self._state["facts"]:
                for pat in self.ENTITY_PATTERNS:
                    for m in pat.finditer(f["text"]):
                        referenced.add(m.group(1).strip().lower())
            self._state["entities"] = [
                e for e in self._state["entities"]
                if (e if isinstance(e, str) else e.get("name", "")).lower() in referenced
            ]

            self._trim()
            self._save()
        return self.get_summary()
