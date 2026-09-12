"""Ingestion: discover files -> read -> chunk -> embed -> store.

Chunking is a character window with overlap that prefers to break on newlines/spaces —
simple, robust, and easy to explain. (Semantic/recursive chunking is a documented next step.)
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .config import cfg
from .keyword_index import KeywordIndex
from .llm import embed
from .store import Store

# The keyword index mirrors the store: every upsert and every delete that
# happens there happens here too, or keyword search starts answering from a
# corpus that no longer exists.
_index = KeywordIndex()

SUPPORTED = {
    ".md", ".markdown", ".txt", ".rst",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml",
    ".pdf",
}
# Matched by name at any depth. "data" is deliberately NOT here: it used to be,
# and since the match is per path segment it skipped every folder named `data`
# anywhere in the world — including the owner's own notes. This program's data
# directory is excluded by path instead, below.
SKIP_DIRS = {".venv", "node_modules", ".git", "__pycache__", ".next"}

# Encodings tried in order before a file is declared undecodable. `errors="ignore"`
# is not on this list and must not come back: it does not fall back, it deletes the
# bytes it cannot read and hands on the wreckage, which then gets embedded and
# cited as though it were what the owner wrote.
_ENCODINGS = ("utf-8", "utf-16", "cp1252", "latin-1")

_report: dict[str, list[dict[str, str]]] = {"skipped": []}


def last_report() -> dict[str, list[dict[str, str]]]:
    """What the most recent discover/ingest left behind, and why.

    A run that drops the owner's file has to be able to say so. Reset at the
    start of each discovery so one run's losses are never reported as another's.
    """
    return {"skipped": list(_report["skipped"])}


def _note_skip(path: Path, reason: str) -> None:
    _report["skipped"].append({"path": str(path), "reason": reason})


def _own_data_dir() -> Path:
    return cfg.state_db.parent.resolve()


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
    """Text from a file, decoded honestly or not at all."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raw = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        # A successful cp1252/latin-1 decode of UTF-16 leaves interleaved NULs;
        # they are not text, and a chunk of them is worse than no chunk at all.
        if "\x00" in text:
            continue
        return text
    raise ValueError(f"cannot decode {path} as any of: {', '.join(_ENCODINGS)}")


def _is_skipped_dir(path: Path, own_data: Path) -> str | None:
    if SKIP_DIRS & set(path.parts):
        return "skipped-dir"
    try:
        if own_data in path.resolve().parents:
            return "own-data-dir"
    except OSError:
        return None
    return None


def discover(root: str | Path) -> list[Path]:
    root = Path(root).expanduser()
    _report["skipped"] = []
    if root.is_file():
        return [root]
    own_data = _own_data_dir()
    files = []
    excluded: dict[str, int] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        reason = _is_skipped_dir(p, own_data)
        if reason:
            # A whole excluded tree is expected noise, not a per-file finding.
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        if p.suffix.lower() not in SUPPORTED:
            _note_skip(p, "unsupported-suffix")
            continue
        files.append(p)
    if not files and excluded:
        # Finding nothing because an exclusion rule ate everything is the one case
        # where the rule has to explain itself: otherwise a misconfigured
        # SB_STATE_DB under the ingest root looks exactly like an empty folder.
        for reason, count in sorted(excluded.items()):
            _note_skip(root, f"{reason} ({count} file(s), nothing left to ingest)")
    return files


def ingest_paths(path: str | Path, reset: bool = False, collection: str | None = None) -> Iterator[tuple[Path, int]]:
    """Yield (file, n_chunks) as each file is ingested, for live progress."""
    store = Store(collection=collection)
    index_name = getattr(store, "collection_name", None) or collection or cfg.collection
    if reset:
        store.reset()
        _index.reset(index_name)
    for f in discover(path):
        try:
            text = read_file(f)
        except ValueError as exc:
            # One unreadable file must not end the run, but it must not vanish either.
            _note_skip(f, f"undecodable: {exc}")
            continue
        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        src = str(f)
        # Replace any earlier ingest of this file. Without this, editing a file so it
        # produces *fewer* chunks would leave the old higher-index chunks behind as
        # orphans. (Skipped on reset=True, which already wiped the whole collection.)
        if not reset:
            store.delete_source(src)
            _index.delete_source(index_name, src)
        if not chunks:
            _note_skip(f, "no-content")
            continue
        ids = [f"{src}#{i}" for i in range(len(chunks))]
        metadatas = [{"source": src, "name": f.name, "chunk": i} for i in range(len(chunks))]
        store.upsert(ids=ids, embeddings=embed(chunks), documents=chunks, metadatas=metadatas)
        _index.upsert_chunks(
            index_name,
            [
                {"source": src, "chunk": i, "name": f.name, "document": chunk}
                for i, chunk in enumerate(chunks)
            ],
        )
        yield f, len(chunks)
