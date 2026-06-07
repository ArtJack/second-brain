"""Retrieve relevant chunks and produce a grounded, cited answer."""
from __future__ import annotations

from contextlib import nullcontext

from .citations import invalid_citations
from .config import cfg
from .hybrid import hybrid_retrieve
from .llm import answer, embed
from .store import Store
from .tracing import TraceRecorder


def _span(
    trace: TraceRecorder | None,
    name: str,
    *,
    kind: str,
    parent_span_id: str | None,
):
    return (
        trace.span(name, kind=kind, parent_span_id=parent_span_id)
        if trace
        else nullcontext()
    )


def ask(
    question: str,
    k: int | None = None,
    *,
    collection: str | None = None,
    trace: TraceRecorder | None = None,
    parent_span_id: str | None = None,
) -> dict:
    store = Store(collection=collection)
    with _span(trace, "store_count", kind="store", parent_span_id=parent_span_id) as count_span:
        chunks = store.count()
        if count_span:
            count_span.add_attributes({"chunks": chunks})
            count_span.set_summary({"chunks": chunks})
    if chunks == 0:
        return {
            "answer": "Nothing ingested yet — run `sb ingest <path>` first.",
            "sources": [],
            "invalid_citations": [],
        }

    with _span(trace, "embed_question", kind="embedding", parent_span_id=parent_span_id) as embed_span:
        qvec = embed([question])[0]
        if embed_span:
            embed_span.add_attributes({"vector_dimensions": len(qvec)})
            embed_span.set_summary({"vector_dimensions": len(qvec)})
    with _span(trace, "retrieve_context", kind="retrieval", parent_span_id=parent_span_id) as retrieve_span:
        limit = k or cfg.top_k
        hits = hybrid_retrieve(store, question, qvec, limit, enabled=cfg.hybrid_enabled)
        if retrieve_span:
            retrieve_span.add_attributes({"hit_count": len(hits), "top_k": k or cfg.top_k})
            retrieve_span.set_summary({"hit_count": len(hits)})

    context_parts, sources = [], []
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        context_parts.append(f"[{i}] (from {meta.get('name', meta.get('source'))})\n{h['document']}")
        sources.append(
            {
                "n": i,
                "source": meta.get("source", "?"),
                "distance": h["distance"],
                "retrieval": h.get("retrieval", "vector"),
            }
        )

    with _span(trace, "generate_answer", kind="llm", parent_span_id=parent_span_id) as answer_span:
        answer_text = answer(question, "\n\n".join(context_parts))
        if answer_span:
            answer_span.add_attributes({"answer_chars": len(answer_text)})
            answer_span.set_summary({"answer_chars": len(answer_text)})
    return {
        "answer": answer_text,
        "sources": sources,
        "invalid_citations": invalid_citations(answer_text, len(sources)),
    }


def recall(query: str, top_k: int = 0, collection: str | None = None) -> dict:
    """Retrieve raw matching chunks without calling the chat model."""
    store = Store(collection=collection)
    if store.count() == 0:
        return {"count": 0, "hits": []}
    k = top_k if top_k and top_k > 0 else cfg.top_k
    qvec = embed([query])[0]
    hits = store.query(qvec, k)
    return {
        "count": len(hits),
        "hits": [
            {
                "source": h["metadata"].get("source", "?"),
                "distance": round(h["distance"], 4),
                "text": h["document"],
            }
            for h in hits
        ],
    }
