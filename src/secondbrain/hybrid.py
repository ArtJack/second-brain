"""Keyword retrieval alongside semantic vector search, and the fusion of the two."""
from __future__ import annotations

import logging
import math
import re
import sqlite3

from .keyword_index import STOPWORDS, KeywordIndex

# One index for the process. Tests replace this attribute directly.
log = logging.getLogger("secondbrain.hybrid")
_index = KeywordIndex()

# Reciprocal-rank fusion's smoothing constant, from Cormack et al. (2009). It
# damps the top of each list so one retriever's first result cannot dominate the
# other's entirely — which matters here because the two scores are not
# commensurable: cosine distance and 1/(1+bm25) share no scale. Fusing on rank
# sidesteps that instead of pretending the numbers can be compared.
RRF_K = 60

# The shared stopword set lives with the index that also needs it.


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


def _intent_boost(hits: list[dict], query: str, tokens: list[str]) -> list[dict]:
    """Nudge a numbered list to the top when the question asks for one.

    FTS5 ranks by term statistics and has no idea that "what are the steps" wants
    the chunk that contains `1.`, `2.`, `3.`. This preserves the one piece of
    judgement the hand-rolled scorer had that a general-purpose index does not.
    """
    if not _has_list_intent(query, tokens):
        return hits
    listed, rest = [], []
    for hit in hits:
        target = listed if re.search(r"(?:^|\n)\s*1\.\s+[A-Z]", hit.get("document", "")) else rest
        target.append(hit)
    return listed + rest


def keyword_query(store, query: str, limit: int = 3) -> list[dict]:
    """Rank stored chunks by keyword, from the persisted index where one exists.

    Rescues the exact section and list lookups that embeddings miss, while the
    answer still has to cite whatever comes back. A collection ingested before
    the index existed has no rows in it, and falls back to the original scan so
    that data keeps working rather than silently returning nothing.
    """
    if limit <= 0:
        return []
    tokens = _keywords(query)
    collection = getattr(store, "collection_name", None)
    if collection:
        try:
            if _index.count(collection):
                return _intent_boost(_index.query(collection, query, limit=limit), query, tokens)
        except sqlite3.Error as exc:
            # Keyword search widens a hybrid answer; it is not a dependency of
            # one. An unreadable index costs the keyword half, where letting the
            # error through costs the whole answer. But never silently: this is
            # exactly how every answer went vector-only for hours on 2026-09-12
            # without a single line in any log.
            log.warning("keyword index unusable for %s, answering vector-only: %s", collection, exc)
            return []
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

    for hit, words in zip(docs, tokenized, strict=True):
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


def _reciprocal_rank_fusion(groups: list[list[dict]], limit: int) -> list[dict]:
    """Combine ranked lists by position rather than by score.

    Each list votes 1/(k + rank) for the documents it ranked. Two retrievers that
    both like a chunk beat one that likes it a lot, and no calibration between
    cosine distance and a BM25-derived distance is needed — which is the point,
    since there is none to be had.
    """
    scores: dict[tuple, float] = {}
    best: dict[tuple, dict] = {}
    for group in groups:
        for rank, hit in enumerate(group, start=1):
            key = _hit_key(hit)
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
            # Keep the first sighting, so a hit found by both retrievers keeps
            # the `retrieval` label and distance of whichever ranked it first.
            best.setdefault(key, hit)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [best[key] for key in ordered[:limit]]


def hybrid_retrieve(
    store,
    query: str,
    qvec: list[float],
    limit: int,
    *,
    enabled: bool = True,
    keyword_limit: int = 3,
) -> list[dict]:
    """Vector search, optionally fused with keyword hits and their neighbours.

    Returns at most `limit` hits. It used to return more: keyword hits and up to
    two neighbours each were prepended to a full page of vector hits and never
    truncated, so `limit=5` could return fourteen chunks and the caller's k was a
    floor rather than a cap.
    """
    vector_hits = store.query(qvec, limit)
    if not enabled:
        return vector_hits
    keyword_hits = keyword_query(store, query, limit=min(keyword_limit, limit))

    # Fuse what the two retrievers actually matched, and nothing else. Neighbours
    # used to be spliced in before fusion, which handed each of them a retrieved
    # rank they had not earned: at the default k=5 an answer was built from one
    # keyword hit, two of its neighbours and two vector hits, and a chunk that
    # matched nothing outscored the second real keyword match. Adjacency is
    # context, not evidence.
    fused = _reciprocal_rank_fusion([keyword_hits, vector_hits], limit)
    if len(fused) >= limit:
        return fused

    # Room left over: widen the surviving keyword hits with their neighbours,
    # which is what adjacency was for before the result set had a cap.
    padded = _merge_hits(fused, _expand_keyword_neighbors(store, fused))
    return padded[:limit]
