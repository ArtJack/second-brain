"""Durable learned memory.

Learning is explicit: the user teaches a fact, we write it to a small Markdown file,
then ingest that file through the same retrieval pipeline as any other source.
"""
from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import cfg
from .ingest import ingest_paths


def _slug(text: str, max_len: int = 48) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    slug = "-".join(words)[:max_len].strip("-")
    return slug or "memory"


def write_memory(text: str, source: str = "user") -> Path:
    text = text.strip()
    if not text:
        raise ValueError("memory text cannot be empty")

    cfg.memory_dir.mkdir(parents=True, exist_ok=True)
    learned_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    name = f"{learned_at.replace(':', '').replace('+0000', 'Z')}-{_slug(text)}-{uuid.uuid4().hex[:8]}.md"
    path = cfg.memory_dir / name
    path.write_text(
        "\n".join(
            [
                "---",
                f"learned_at: {learned_at}",
                f"source: {source}",
                "type: learned_memory",
                "---",
                "",
                text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def learn(text: str, source: str = "user", collection: str | None = None) -> dict:
    path = write_memory(text, source=source)
    chunks = 0
    for _, n in ingest_paths(path, collection=collection):
        chunks += n
    return {"path": path, "chunks": chunks}
