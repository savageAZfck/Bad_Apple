#!/usr/bin/env python3
"""Local document extraction and Q&A for Bad Apple.

Reads PDF and EPUB files on device, extracts text, and exposes it for
indexing, search, and question-answering through the existing RAG pipeline.
"""

import re
from pathlib import Path
from typing import Any

try:
    from badapple_knowledge import BadAppleKnowledge
except ImportError:
    BadAppleKnowledge = Any


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber
        parts: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text() or ""
                    if text.strip():
                        parts.append(f"--- Page {i} ---\n{text}")
                except Exception as e:  # noqa: BLE001 - logged
                    print(f"[documents] extract_text failed: {e}", flush=True)
        return "\n\n".join(parts)
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        return f"PDF extraction error: {e}"


def _extract_epub(path: Path) -> str:
    try:
        from ebooklib import epub
        book = epub.read_epub(str(path))
        parts: list[str] = []
        for item in book.get_items():
            if item.get_type() == 9:  # ITEM_DOCUMENT / DOCUMENT
                try:
                    content = item.get_content().decode("utf-8", errors="ignore")
                    text = re.sub(r"<[^>]+>", " ", content)
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        parts.append(text)
                except Exception as e:  # noqa: BLE001 - logged
                    print(f"[documents] decode failed: {e}", flush=True)
        return "\n\n".join(parts)
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        return f"EPUB extraction error: {e}"


def _extract_plain(path: Path, limit: int = 100000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except (OSError, ValueError) as e:
        return f"Text read error: {e}"


def extract_document(path: Path, limit: int = 100000) -> tuple[str, int]:
    """Extract text from a local PDF, EPUB, or plain text file.

    Returns (text, pages_or_chunks).
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(path)[:limit]
        pages = text.count("--- Page")
        return text, pages
    if suffix == ".epub":
        text = _extract_epub(path)[:limit]
        chunks = max(1, len(text) // 4000)
        return text, chunks
    if suffix in {".txt", ".md", ".py", ".rs", ".swift", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".c", ".cpp", ".h", ".go", ".java", ".sh", ".bash"}:
        text = _extract_plain(path, limit)
        return text, 1
    return f"Unsupported file type: {suffix}", 0


def read_document(path: str, max_chars: int = 10000) -> str:
    """Extract text from a local PDF/EPUB/plain file and return a preview."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"Error: {p} not found"
    text, count = extract_document(p, limit=max_chars)
    if text.startswith("Error") or text.startswith("Unsupported"):
        return text
    header = f"Document: {p.name} ({count} pages/chunks)\n---\n"
    return header + text[:max_chars]


def index_document(path: str, knowledge: BadAppleKnowledge) -> str:
    """Extract a PDF/EPUB and feed it into the RAG index."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"Error: {p} not found"
    text, count = extract_document(p)
    if text.startswith("Error") or text.startswith("Unsupported"):
        return text
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"badapple_doc_{p.stem}.txt"
    try:
        tmp.write_text(text, encoding="utf-8")
        indexed = knowledge.index_paths([tmp])
        return f"Indexed {p.name} ({count} pages/chunks, {indexed} chunks)."
    except (OSError, ValueError) as e:
        return f"Index error: {e}"
