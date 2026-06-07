"""Small local keyword retriever to complement semantic vector search."""
from __future__ import annotations

import math
import re

STOPWORDS = {
    "about",
    "all",
    "and",
    "are",
    "can",
    "did",
    "does",
    "for",
    "from",
    "how",
    "into",
    "is",
    "its",
    "me",
    "my",
    "of",
    "or",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _fold_token(token: str) -> str:
    if len(token) >= 4 and token.endswith("s"):
        return token[:-1]
    return token


def _keywords(text: str) -> list[str]:
    keywords = []
    for raw in _words(text):
        if raw in STOPWORDS:
            continue
        word = _fold_token(raw)
        if len(word) > 2:
            keywords.append(word)
    return keywords


def _normalize_words(text: str) -> list[str]:
    return [_fold_token(word) for word in _words(text)]


def _phrases(tokens: list[str]) -> list[str]:
    phrases: list[str] = []
    for size in (2, 3):
        phrases.extend(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))
    return phrases


def _has_list_intent(query: str, tokens: list[str]) -> bool:
    lower = query.lower()
    list_terms = {"list", "enumerate", "step", "way", "type", "kind", "example"}
    return bool(list_terms & set(tokens)) or "how many" in lower or "what are" in lower


def _hit_key(hit: dict) -> tuple[object, object, str]:
    meta = hit.get("metadata") or {}
    return meta.get("source"), meta.get("chunk"), hit.get("document", "")


def _merge_hits(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[object, object, str]] = set()
    for group in groups:
        for hit in group:
            key = _hit_key(hit)
            if key in seen:
                continue
            merged.append(hit)
            seen.add(key)
    return merged


def _expand_keyword_neighbors(store, hits: list[dict], right: int = 2) -> list[dict]:
    expanded: list[dict] = []
    for hit in hits:
        expanded.append(hit)
        if hit.get("retrieval") != "keyword":
            continue
        meta = hit.get("metadata") or {}
        source = meta.get("source")
        chunk = meta.get("chunk")
        if source is None or chunk is None or not hasattr(store, "get_source_chunk"):
            continue
        try:
            chunk_num = int(chunk)
        except (TypeError, ValueError):
            continue
        for offset in range(1, right + 1):
            neighbor = store.get_source_chunk(source, chunk_num + offset)
            if not neighbor:
                continue
            neighbor["retrieval"] = "keyword-adjacent"
            neighbor["distance"] = hit["distance"]
            expanded.append(neighbor)
    return _merge_hits(expanded)


def keyword_query(store, query: str, limit: int = 3) -> list[dict]:
    """Rank stored chunks with a tiny BM25-style scorer.

    This is intentionally simple and local: it rescues exact section/list lookups that
    embeddings sometimes miss, while the answer still has to cite retrieved chunks.
    """
    if limit <= 0:
        return []
    tokens = _keywords(query)
    if not tokens:
        return []
    try:
        docs = store.documents()
    except AttributeError:
        return []
    if not docs:
        return []

    tokenized = [_normalize_words(hit.get("document", "")) for hit in docs]
    n_docs = len(docs)
    avg_len = sum(len(words) for words in tokenized) / max(n_docs, 1)
    doc_freq = {
        token: sum(1 for words in tokenized if token in set(words))
        for token in set(tokens)
    }
    phrases = _phrases(tokens)
    wants_list = _has_list_intent(query, tokens)
    scored: list[tuple[float, dict]] = []

    for hit, words in zip(docs, tokenized):
        if not words:
            continue
        doc = hit.get("document", "")
        normalized_doc = " ".join(words)
        doc_len = len(words)
        score = 0.0
        for token in tokens:
            tf = words.count(token)
            if not tf:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * ((tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * doc_len / max(avg_len, 1))))
        for phrase in phrases:
            score += 1.5 * normalized_doc.count(phrase)
        if wants_list and re.search(r"(?:^|\n)\s*1\.\s+[A-Z]", doc):
            score += 8.0
        if score:
            enriched = {
                **hit,
                "distance": 1 / (1 + score),
                "retrieval": "keyword",
            }
            scored.append((score, enriched))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in scored[:limit]]


def hybrid_retrieve(
    store,
    query: str,
    qvec: list[float],
    limit: int,
    *,
    enabled: bool = True,
    keyword_limit: int = 3,
) -> list[dict]:
    """Vector search, optionally fused with local BM25 keyword hits + neighbor expansion."""
    vector_hits = store.query(qvec, limit)
    if not enabled:
        return vector_hits
    keyword_hits = keyword_query(store, query, limit=min(keyword_limit, limit))
    return _merge_hits(_expand_keyword_neighbors(store, keyword_hits), vector_hits)
