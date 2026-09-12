"""Ingestion: discover files -> read -> chunk -> embed -> store.

Chunking is a character window with overlap that prefers to break on newlines/spaces —
simple, robust, and easy to explain. (Semantic/recursive chunking is a documented next step.)
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
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
#
# UTF-16 is deliberately NOT in this ladder. It decodes almost any byte sequence
# of even length without error, so trying it speculatively turned roughly half of
# all Latin-1 notes into CJK mojibake — "Über den Wolken" became 拜牥搠湥圠汯敫确 —
# which was then embedded and cited as the owner's own words. It is attempted only
# when a byte-order mark says the file really is UTF-16. The first version of this
# ladder shipped with the bug and its test passed only because the sample string
# happened to be an odd number of bytes.
_ENCODINGS = ("utf-8", "cp1252", "latin-1")
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")

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


def _chunk_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """The windows `chunk_text` cuts, as offsets into the stripped text.

    Split out so the header logic can ask *where* a chunk came from. The
    arithmetic is unchanged and `chunk_text` is now a projection of this, which
    is what keeps the bodies byte-identical to what shipped before.
    """
    spans: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # try a clean break near the window edge
            cut = text.rfind("\n", start + size - overlap, end)
            if cut == -1:
                cut = text.rfind(" ", start + size - overlap, end)
            if cut > start:
                end = cut
        spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return spans


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces = (text[start:end].strip() for start, end in _chunk_spans(text, size, overlap))
    return [piece for piece in pieces if piece]


# ATX headings only, and only H1 through H3. A `#####` is too fine to orient
# anyone, and a `#` is the document's title rather than a section within it.
_HEADING = re.compile(r"^(#{1,3})[ \t]+(\S.*?)[ \t]*#*$", re.MULTILINE)
_FENCE = re.compile(r"^[ \t]*(```|~~~)", re.MULTILINE)


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges inside fenced code blocks.

    `# comment` at the start of a line in a shell block is a comment, and
    treating it as a heading put "not a heading" on real chunks.
    """
    ranges: list[tuple[int, int]] = []
    opening: int | None = None
    for match in _FENCE.finditer(text):
        if opening is None:
            opening = match.start()
        else:
            ranges.append((opening, match.end()))
            opening = None
    if opening is not None:  # an unclosed fence swallows the rest of the file
        ranges.append((opening, len(text)))
    return ranges


@dataclass(frozen=True)
class DocumentChunk:
    """One chunk, plus where in its document it came from.

    `body` is the source text a citation quotes. `embedded` is what retrieval
    sees: the header and the body. They are deliberately different — the header
    orients the retriever and the model, and putting it in the quoted text would
    mean citing words the owner never wrote.
    """

    body: str
    title: str
    section: str | None
    page: int | None

    @property
    def header(self) -> str:
        return f"{self.title} › {self.section}" if self.section else self.title

    @property
    def embedded(self) -> str:
        return f"{self.header}\n\n{self.body}"


def chunk_document(
    text: str,
    size: int,
    overlap: int,
    *,
    name: str,
    page_starts: list[int] | None = None,
) -> list[DocumentChunk]:
    """Chunks carrying the title and section they sit under."""
    stripped = text.strip()
    if not stripped:
        return []
    offset = text.index(stripped[0]) if stripped else 0

    fenced = _fenced_ranges(stripped)

    def in_code(position: int) -> bool:
        return any(start <= position < end for start, end in fenced)

    headings = [
        (match.start(), len(match.group(1)), match.group(2).strip())
        for match in _HEADING.finditer(stripped)
        if not in_code(match.start())
    ]
    title = next((text for _, level, text in headings if level == 1), None) or Path(name).stem
    sections = [(position, text) for position, level, text in headings if level in (2, 3)]

    chunks: list[DocumentChunk] = []
    for start, end in _chunk_spans(stripped, size, overlap):
        body = stripped[start:end].strip()
        if not body:
            continue
        # Two cases, and the first document ever tested got the second one.
        # A chunk that *follows* headings belongs to the last one before it —
        # that is the section it opens in. A chunk that starts before any
        # heading, which includes the common case of a short document arriving
        # whole, belongs to the first heading it actually contains. Asking only
        # the first question labelled every such document with no section at all.
        section = next((t for position, t in reversed(sections) if position <= start), None)
        if section is None:
            section = next((t for position, t in sections if start < position < end), None)
        page = None
        if page_starts:
            # Offsets are into the original text; the spans are into the
            # stripped copy, so the leading whitespace has to be added back.
            absolute = start + offset
            page = sum(1 for boundary in page_starts if boundary <= absolute) or 1
        chunks.append(DocumentChunk(body=body, title=title, section=section, page=page))
    return chunks


_ID_PREFIX = "c:"
_ID_LENGTH = 32


def normalised(text: str) -> str:
    """The form a chunk is hashed in.

    Leading and trailing space stripped and internal runs of whitespace
    collapsed to one, so the same sentence reflowed by an editor keeps its
    identity. Case is deliberately preserved: "Gateway" and "gateway" are
    different text and a citation should be able to tell them apart.
    """
    return " ".join(str(text).split())


def chunk_id(text: str) -> str:
    """Identity derived from content, not from where the content was found.

    The path was doing this job, which made a copied file a second corpus and
    made every web ingest a fresh document under a temporary directory that no
    longer existed by the time anyone read the citation.
    """
    digest = hashlib.sha256(normalised(text).encode("utf-8")).hexdigest()
    return f"{_ID_PREFIX}{digest[:_ID_LENGTH]}"


def _payload(chunk, index: int, *, path: str, doc_type: str, mtime: float) -> dict:
    meta = {
        "path": path,
        # Kept as an alias of `path` so hybrid, ask, citations and the web UI
        # keep reading the key they have always read. Removing it is a separate
        # change to every read site, not a side effect of this one.
        "source": path,
        "name": Path(path).name or path,
        "chunk": index,
        "doc_type": doc_type,
        "mtime": mtime,
        "ingested_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "title": chunk.title,
        "header": chunk.header,
    }
    if chunk.section:
        meta["section"] = chunk.section
    if chunk.page:
        meta["page"] = chunk.page
    return meta


def ingest_text(
    text: str,
    *,
    source: str,
    doc_type: str = "text",
    collection: str | None = None,
) -> int:
    """Ingest text the caller already holds, under the logical source it names.

    The web endpoint used to write the body to a temporary file and ingest that,
    so the stored source was a `/var/folders/...` path unique to one request.
    Nothing could ever match it again: re-posting the same document added a
    second copy, and the citation a reader clicked pointed at a file deleted
    when the request ended.
    """
    store = Store(collection=collection)
    index_name = getattr(store, "collection_name", None) or collection or cfg.collection
    chunks = chunk_document(text, cfg.chunk_size, cfg.chunk_overlap, name=Path(source).name or source)
    store.delete_source(source)
    _index.delete_source(index_name, source)
    if not chunks:
        return 0
    _write_chunks(store, index_name, chunks, path=source, doc_type=doc_type, mtime=0.0)
    return len(chunks)


def _write_chunks(store, index_name: str, chunks, *, path: str, doc_type: str, mtime: float) -> None:
    """The one place a chunk becomes a point, so both stores always agree."""
    store.upsert(
        ids=[chunk_id(chunk.body) for chunk in chunks],
        # Embedded with the header, stored without it.
        embeddings=embed([chunk.embedded for chunk in chunks]),
        documents=[chunk.body for chunk in chunks],
        metadatas=[
            _payload(chunk, i, path=path, doc_type=doc_type, mtime=mtime)
            for i, chunk in enumerate(chunks)
        ],
    )
    _index.upsert_chunks(
        index_name,
        [
            {
                "source": path,
                "chunk": i,
                "name": Path(path).name or path,
                "header": chunk.header,
                "document": chunk.body,
            }
            for i, chunk in enumerate(chunks)
        ],
    )


def read_pages(path: Path) -> tuple[str, list[int]]:
    """A PDF's text plus the offset each page starts at.

    Same joined text `read_file` has always produced; the offsets are what lets
    a chunk say which page it came from, so a citation into a 200-page document
    points somewhere a person can actually turn to.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    starts: list[int] = []
    parts: list[str] = []
    position = 0
    for page in reader.pages:
        starts.append(position)
        text = page.extract_text() or ""
        parts.append(text)
        position += len(text) + 2  # the "\n\n" the join inserts
    return "\n\n".join(parts), starts


def read_file(path: Path) -> str:
    """Text from a file, decoded honestly or not at all."""
    if path.suffix.lower() == ".pdf":
        return read_pages(path)[0]
    raw = path.read_bytes()
    attempted = []
    if raw[:2] in _UTF16_BOMS:
        attempted.append("utf-16")
        try:
            return raw.decode("utf-16")
        except (UnicodeDecodeError, UnicodeError):
            pass
    for encoding in _ENCODINGS:
        attempted.append(encoding)
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        # A single-byte decode of UTF-16 that slipped past the BOM check leaves
        # interleaved NULs. They are not text, and a chunk of them is worse than
        # no chunk at all.
        if "\x00" in text:
            continue
        return text
    raise ValueError(f"cannot decode {path} as any of: {', '.join(attempted)}")


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
        page_starts: list[int] | None = None
        try:
            if f.suffix.lower() == ".pdf":
                text, page_starts = read_pages(f)
            else:
                text = read_file(f)
        except ValueError as exc:
            # One unreadable file must not end the run, but it must not vanish either.
            _note_skip(f, f"undecodable: {exc}")
            continue
        chunks = chunk_document(
            text, cfg.chunk_size, cfg.chunk_overlap, name=f.name, page_starts=page_starts
        )
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
        _write_chunks(
            store,
            index_name,
            chunks,
            path=src,
            doc_type=f.suffix.lstrip(".").lower() or "text",
            mtime=f.stat().st_mtime,
        )
        yield f, len(chunks)
