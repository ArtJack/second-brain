"""Retrieve relevant chunks and produce a grounded, cited answer."""
from __future__ import annotations

from .citations import invalid_citations
from .config import cfg
from .llm import answer, embed
from .store import Store


def ask(question: str, k: int | None = None) -> dict:
    store = Store()
    if store.count() == 0:
        return {
            "answer": "Nothing ingested yet — run `sb ingest <path>` first.",
            "sources": [],
            "invalid_citations": [],
        }

    qvec = embed([question])[0]
    hits = store.query(qvec, k or cfg.top_k)

    context_parts, sources = [], []
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        context_parts.append(f"[{i}] (from {meta.get('name', meta.get('source'))})\n{h['document']}")
        sources.append({"n": i, "source": meta.get("source", "?"), "distance": h["distance"]})

    answer_text = answer(question, "\n\n".join(context_parts))
    return {
        "answer": answer_text,
        "sources": sources,
        "invalid_citations": invalid_citations(answer_text, len(sources)),
    }
