"""Vector store adapters.

Chroma remains the zero-infra default. Qdrant is available when `SB_STORE=qdrant`,
which lets the same CLI use the hosted vector DB already running in the AI lab.
Embeddings are always passed explicitly, so neither store downloads its own model.
"""
from __future__ import annotations

import uuid
import warnings
from pathlib import Path

import chromadb

from .config import cfg


class ChromaStore:
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

    def delete_source(self, source: str) -> None:
        """Drop every chunk previously ingested from `source` (matched on metadata)."""
        self._col.delete(where={"source": source})

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


class QdrantStore:
    def __init__(self, collection: str | None = None) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant backend requires qdrant-client. Run `uv sync` after updating dependencies."
            ) from exc

        self._models = models
        self._collection = collection or cfg.collection
        if cfg.qdrant_url == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Api key is used with an insecure connection.")
                self._client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    @staticmethod
    def _point_id(source_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))

    def _exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(collection_name=self._collection))
        except Exception as exc:
            if "401" in str(exc) or "Unauthorized" in str(exc):
                raise RuntimeError(
                    "Qdrant rejected the request. Set QDRANT_API_KEY in .env "
                    "or switch back to SB_STORE=chroma."
                ) from exc
            raise

    def _ensure_collection(self, vector_size: int) -> None:
        if self._exists():
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=self._models.VectorParams(
                size=vector_size,
                distance=self._models.Distance.COSINE,
            ),
        )

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        if not embeddings:
            return
        self._ensure_collection(len(embeddings[0]))
        points = []
        for source_id, vector, document, metadata in zip(ids, embeddings, documents, metadatas):
            payload = dict(metadata)
            payload["source_id"] = source_id
            payload["document"] = document
            points.append(
                self._models.PointStruct(
                    id=self._point_id(source_id),
                    vector=vector,
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)

    def delete_source(self, source: str) -> None:
        """Drop every point previously ingested from `source` (matched on payload)."""
        if not self._exists():
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._models.FilterSelector(
                filter=self._models.Filter(
                    must=[
                        self._models.FieldCondition(
                            key="source",
                            match=self._models.MatchValue(value=source),
                        )
                    ]
                )
            ),
        )

    def query(self, embedding: list[float], k: int):
        if not self._exists():
            return []
        response = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=k,
            with_payload=True,
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            metadata = {key: value for key, value in payload.items() if key != "document"}
            results.append(
                {
                    "document": payload.get("document", ""),
                    "metadata": metadata,
                    "distance": 1 - float(hit.score),
                }
            )
        return results

    def count(self) -> int:
        if not self._exists():
            return 0
        return int(self._client.count(collection_name=self._collection, exact=True).count)

    def reset(self) -> None:
        if self._exists():
            self._client.delete_collection(self._collection)


def Store(persist_dir: Path | None = None, collection: str | None = None):
    if cfg.store_backend == "chroma":
        return ChromaStore(persist_dir=persist_dir, collection=collection)
    if cfg.store_backend == "qdrant":
        return QdrantStore(collection=collection)
    raise ValueError(f"Unsupported SB_STORE={cfg.store_backend!r}; expected 'chroma' or 'qdrant'.")
