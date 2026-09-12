"""Thin wrapper over the OpenAI-compatible endpoint (Ollama locally, or the gateway).

Kept deliberately small: embeddings + chat. Same code works against any provider that
speaks the OpenAI API — that portability is the whole point of routing through a gateway.
"""
from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterator
from pathlib import Path

from openai import BadRequestError, OpenAI

from .config import cfg

_client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.llm_timeout_s, max_retries=1)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, in one batched call where the backend allows it.

    The per-item fallback exists for one reason: some Ollama builds reject a
    batched `input`. It is now scoped to exactly that — a 4xx saying the request
    shape was wrong. It used to catch every exception, so a gateway that was
    simply unreachable turned one failed call into N more, each waiting out the
    client timeout: the slowest possible way to discover the lab is asleep.
    """
    if not texts:
        return []
    try:
        resp = _client.embeddings.create(model=cfg.embed_model, input=texts)
        return [d.embedding for d in resp.data]
    except BadRequestError:
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


def answer_stream(question: str, context: str) -> Iterator[str]:
    """Stream an answer grounded ONLY in the supplied context, yielding text deltas."""
    system = (
        "You are the user's personal knowledge assistant. Answer the question using ONLY "
        "the numbered context below. Cite the sources you used inline as [1], [2], etc. "
        "If the context does not contain the answer, say so plainly — do not invent facts."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    stream = _client.chat.completions.create(
        model=cfg.chat_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


def see(image: str | Path, question: str = "Describe this image in detail.") -> str:
    """Look at an image and answer a question about it, using the vision route
    (llava on the lab gateway). Accepts a local file path or an http(s) URL —
    local files are inlined as base64, so nothing leaves the OpenAI call."""
    src = str(image)
    if src.startswith(("http://", "https://")):
        url = src
    else:
        path = Path(src).expanduser()
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode()
        url = f"data:{mime};base64,{data}"
    resp = _client.chat.completions.create(
        model=cfg.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""
