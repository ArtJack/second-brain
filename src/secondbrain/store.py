"""Vector store. Chroma (local, embedded) for v1 — zero infra, exactly the "local first"
step in the curriculum. The interface is small so swapping to the lab's Qdrant later is
a drop-in. We pass embeddings explicitly, so Chroma never downloads its own model.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from .config import cfg


class Store:
    def __init__(self, persist_dir: Path | None = None, collection: str | None = None) -> None:
        path = persist_dir or cfg.persist_dir
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=collection or cfg.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        self._col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: list[float], k: int):
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        # Flatten Chroma's per-query nesting into a simple list of hits.
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({"document": doc, "metadata": meta, "distance": dist})
        return hits

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(self._col.name)
        self._col = self._client.get_or_create_collection(
            name=cfg.collection, metadata={"hnsw:space": "cosine"}
        )
