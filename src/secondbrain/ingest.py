"""Ingestion: discover files -> read -> chunk -> embed -> store.

Chunking is a character window with overlap that prefers to break on newlines/spaces —
simple, robust, and easy to explain. (Semantic/recursive chunking is a documented next step.)
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .config import cfg
from .llm import embed
from .store import Store

SUPPORTED = {
    ".md", ".markdown", ".txt", ".rst",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml",
    ".pdf",
}
SKIP_DIRS = {".venv", "node_modules", ".git", "__pycache__", ".next", "data"}


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # try a clean break near the window edge
            cut = text.rfind("\n", start + size - overlap, end)
            if cut == -1:
                cut = text.rfind(" ", start + size - overlap, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def discover(root: str | Path) -> list[Path]:
    root = Path(root).expanduser()
    if root.is_file():
        return [root]
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED and not (SKIP_DIRS & set(p.parts)):
            files.append(p)
    return files


def ingest_paths(path: str | Path, reset: bool = False) -> Iterator[tuple[Path, int]]:
    """Yield (file, n_chunks) as each file is ingested, for live progress."""
    store = Store()
    if reset:
        store.reset()
    for f in discover(path):
        text = read_file(f)
        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        src = str(f)
        # Replace any earlier ingest of this file. Without this, editing a file so it
        # produces *fewer* chunks would leave the old higher-index chunks behind as
        # orphans. (Skipped on reset=True, which already wiped the whole collection.)
        if not reset:
            store.delete_source(src)
        if not chunks:
            continue
        ids = [f"{src}#{i}" for i in range(len(chunks))]
        metadatas = [{"source": src, "name": f.name, "chunk": i} for i in range(len(chunks))]
        store.upsert(ids=ids, embeddings=embed(chunks), documents=chunks, metadatas=metadatas)
        yield f, len(chunks)
