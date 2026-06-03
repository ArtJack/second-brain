"""Thin wrapper over the OpenAI-compatible endpoint (Ollama locally, or the gateway).

Kept deliberately small: embeddings + chat. Same code works against any provider that
speaks the OpenAI API — that portability is the whole point of routing through a gateway.
"""
from __future__ import annotations

from openai import OpenAI

from .config import cfg

_client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Tries one batched call; falls back to per-item if the
    backend rejects batches (some Ollama builds do)."""
    if not texts:
        return []
    try:
        resp = _client.embeddings.create(model=cfg.embed_model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        out: list[list[float]] = []
        for t in texts:
            resp = _client.embeddings.create(model=cfg.embed_model, input=t)
            out.append(resp.data[0].embedding)
        return out


def answer(question: str, context: str) -> str:
    """Generate an answer grounded ONLY in the supplied context, with [n] citations."""
    system = (
        "You are the user's personal knowledge assistant. Answer the question using ONLY "
        "the numbered context below. Cite the sources you used inline as [1], [2], etc. "
        "If the context does not contain the answer, say so plainly — do not invent facts."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    resp = _client.chat.completions.create(
        model=cfg.chat_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""
