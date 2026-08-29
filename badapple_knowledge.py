"""Bad Apple local knowledge / RAG layer.

Uses a small sentence-transformer style embedding model (BGE-small) on CPU to
index and search the user's local text files. All vectors and chunks are stored
on disk under /var/lib/bad_apple/knowledge so the assistant can answer from the
user's own documents without the cloud.
"""
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norm + 1e-10)


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)


class BadAppleKnowledge:
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_INDEX_DIR = Path("/var/lib/bad_apple/knowledge")
    DEFAULT_INDEX_NAME = "local_docs.json"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        index_dir: Path = DEFAULT_INDEX_DIR,
        max_chunk_chars: int = 800,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.max_chunk_chars = max_chunk_chars
        self._lock = threading.RLock()
        self.batch_size = batch_size

        self.tokenizer = None
        self.model = None
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.embeddings: np.ndarray | None = None

        self._load_index()

    def _load_model(self):
        if self.tokenizer is None:
            print("Loading embedding model...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print("Embedding model loaded.", flush=True)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                model_output = self.model(**encoded)
            embeddings = _mean_pooling(model_output, encoded["attention_mask"])
            all_embeddings.append(embeddings.numpy())
        return _normalize(np.vstack(all_embeddings))

    def _chunk_file(self, path: Path) -> list[tuple[str, str]]:
        """Return (source_label, chunk_text) for a file."""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, ValueError):
            return []
        if not text.strip():
            return []

        # Try paragraph splitting, fall back to fixed-size windows
        parts = re.split(r"\n\s*\n", text)
        if len(parts) <= 1:
            parts = [text[i : i + self.max_chunk_chars] for i in range(0, len(text), self.max_chunk_chars)]

        results = []
        for part in parts:
            part = " ".join(part.split())
            if len(part) > 30:
                label = f"[{path}] {part}".strip()
                results.append((str(path), label))
        return results

    SKIP_DIRS = {".git", ".svn", ".venv", ".env", "venv", "env", "node_modules", "target", "build", "dist", "__pycache__", ".pytest_cache"}

    def _should_skip_path(self, f: Path) -> bool:
        return any(part in self.SKIP_DIRS for part in f.parts)

    def index_paths(self, paths: list[Path], extensions: set | None = None) -> int:
        """Index the given files or directories. Returns number of chunks."""
        with self._lock:
            return self._index_paths_unsafe(paths, extensions)

    def _index_paths_unsafe(self, paths: list[Path], extensions: set | None = None) -> int:
        extensions = extensions or {".txt", ".md", ".rs", ".swift", ".py", ".sh", ".toml"}

        new_chunks = []
        new_sources = []
        for p in paths:
            if p.is_file() and p.suffix.lower() in extensions and not self._should_skip_path(p):
                for src, chunk in self._chunk_file(p):
                    new_chunks.append(chunk)
                    new_sources.append(src)
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in extensions:
                        if self._should_skip_path(f) or f.stat().st_size > 512 * 1024:
                            continue
                        for src, chunk in self._chunk_file(f):
                            new_chunks.append(chunk)
                            new_sources.append(src)

        if not new_chunks:
            return 0

        # Remove stale chunks for any source we are about to re-index.
        new_sources_set = set(new_sources)
        if self.sources and self.chunks:
            keep = [i for i, s in enumerate(self.sources) if s not in new_sources_set]
            if len(keep) < len(self.sources):
                self.chunks = [self.chunks[i] for i in keep]
                self.sources = [self.sources[i] for i in keep]
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[keep, :]

        start = time.perf_counter()
        embeddings = self._encode_texts(new_chunks)
        print(f"Indexed {len(new_chunks)} chunks in {time.perf_counter() - start:.2f}s", flush=True)

        self.chunks.extend(new_chunks)
        self.sources.extend(new_sources)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self._save_index()
        return len(new_chunks)

    def search(self, query: str, k: int = 3, threshold: float = 0.45) -> list[tuple[str, float]]:
        with self._lock:
            if self.embeddings is None or not self.chunks:
                return []
            self._load_model()
            q_emb = self._encode_texts([query])
            scores = (q_emb @ self.embeddings.T)[0]
            k = min(k, len(scores))
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            results = []
            for idx in top_idx:
                if scores[idx] >= threshold:
                    results.append((f"[{self.sources[idx]}] {self.chunks[idx]}", float(scores[idx])))
            return results

    def _save_index(self):
        path = self.index_dir / self.DEFAULT_INDEX_NAME
        try:
            data = {
                "chunks": self.chunks,
                "sources": self.sources,
                "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
        except (TypeError, ValueError, OSError) as e:
            print(f"Warning: could not save index: {e}", flush=True)

    def _load_index(self):
        path = self.index_dir / self.DEFAULT_INDEX_NAME
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            self.sources = data.get("sources", [])
            if data.get("embeddings"):
                self.embeddings = _normalize(np.array(data["embeddings"]))
            print(f"Loaded {len(self.chunks)} chunks from knowledge index.", flush=True)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError, LookupError) as e:
            print(f"Warning: could not load index: {e}", flush=True)
