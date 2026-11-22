"""The one test fake for the vector store (BUILD-STANDARD.md, section 2.4).

FakeStore keeps what it is given the way ChromaStore and QdrantStore do, and
records every call. tests/test_store_contract.py runs the same assertions
against it and against both real stores, so a rule only the fake has shows up
there. It never sleeps and returns nothing random.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .config import cfg


class FakeStore:
    """An in-memory store that answers like the real ones and records each call.

    - `points` seeds what it holds, as `{"id", "document", "metadata"}` rows.
      `upsert` replaces by id, `delete_source` drops every chunk whose metadata
      `source` matches, and `reset` empties it. `documents`, `sources`,
      `get_source_chunk` and `count` read what it holds.
    - `query` returns `hits` exactly as given: which stored chunks are nearest is
      the test's to say, so give them nearest first and no more than `k`.
    - `count`, when given, is what `count()` answers whatever it holds, for tests
      where the content is beside the point.
    - `error`, when given, is raised by every call once the call is recorded.
    - `calls` records every call as `(method, *arguments)`; `deleted` lists the
      sources `delete_source` was given.
    """

    def __init__(
        self,
        collection: str | None = None,
        *,
        points: Iterable[Mapping[str, Any]] = (),
        hits: list[dict[str, Any]] | None = None,
        count: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._collection = collection or cfg.collection
        self._points: dict[str, dict[str, Any]] = {
            point["id"]: {"document": point["document"], "metadata": dict(point["metadata"])} for point in points
        }
        self._hits = hits if hits is not None else []
        self._count = count
        self._error = error
        self.calls: list[tuple[Any, ...]] = []

    @property
    def collection_name(self) -> str:
        return self._collection

    @property
    def deleted(self) -> list[str]:
        return [call[1] for call in self.calls if call[0] == "delete_source"]

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        self._record("upsert", list(ids), list(embeddings), list(documents), [dict(m) for m in metadatas])
        for point_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self._points[point_id] = {"document": document, "metadata": dict(metadata)}

    def delete_source(self, source: str) -> None:
        self._record("delete_source", source)
        for point_id in [pid for pid, point in self._points.items() if point["metadata"].get("source") == source]:
            del self._points[point_id]

    def query(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        self._record("query", embedding, k)
        return self._hits

    def documents(self) -> list[dict[str, Any]]:
        self._record("documents")
        return [self._hit(point) for point in self._points.values()]

    def get_source_chunk(self, source: str, chunk: int) -> dict[str, Any] | None:
        self._record("get_source_chunk", source, chunk)
        for point in self._points.values():
            if point["metadata"].get("source") == source and point["metadata"].get("chunk") == chunk:
                return self._hit(point)
        return None

    def sources(self) -> dict[str, int]:
        self._record("sources")
        counts: dict[str, int] = {}
        for point in self._points.values():
            source = point["metadata"].get("source")
            if source:
                counts[source] = counts.get(source, 0) + 1
        return counts

    def count(self) -> int:
        self._record("count")
        return len(self._points) if self._count is None else self._count

    def reset(self) -> None:
        self._record("reset")
        self._points.clear()

    def _record(self, method: str, *arguments: Any) -> None:
        self.calls.append((method, *arguments))
        if self._error is not None:
            raise self._error

    @staticmethod
    def _hit(point: dict[str, Any]) -> dict[str, Any]:
        return {"document": point["document"], "metadata": dict(point["metadata"]), "distance": 1.0}


class FakeStoreFactory:
    """Stands in for `store.Store` where the code under test builds its own store.

    `collections` records the collection each construction asked for, in order.
    Each collection gets one FakeStore: the one `stores` holds for that name, or
    else one built with `defaults` and kept for the next construction.
    """

    def __init__(self, stores: Mapping[str | None, FakeStore] | None = None, **defaults: Any) -> None:
        self.collections: list[str | None] = []
        self.stores: dict[str | None, FakeStore] = dict(stores or {})
        self._defaults = defaults

    def __call__(self, persist_dir: Any = None, collection: str | None = None) -> FakeStore:
        self.collections.append(collection)
        if collection not in self.stores:
            self.stores[collection] = FakeStore(collection, **self._defaults)
        return self.stores[collection]
