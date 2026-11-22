"""The shared FakeStore is held to the real stores: the same assertions, run on each.

Every test here runs against `fakes.FakeStore`, against ChromaStore on a
`tmp_path` database, and against QdrantStore in its in-memory mode, which needs
no server. A rule only the fake has fails here instead of passing everywhere
else: the identity tests' old fake deleted on metadata `path`, where both real
stores delete on `source`.

Two differences between the real stores shape the assertions:
- QdrantStore adds its own `source_id` to the metadata it returns, so a hit
  carries the metadata that went in rather than equalling it.
- QdrantStore's distance for an exact match can come back a hair below zero, so
  it is compared with a tolerance.
"""
from __future__ import annotations

import inspect

import pytest

from secondbrain.config import cfg
from secondbrain.fakes import FakeStore
from secondbrain.store import ChromaStore, QdrantStore, VectorStore

POINTS = [
    {"id": "/a.md#0", "embedding": [0.1, 0.2], "document": "a zero", "metadata": {"source": "/a.md", "chunk": 0}},
    {"id": "/a.md#1", "embedding": [0.2, 0.1], "document": "a one", "metadata": {"source": "/a.md", "chunk": 1}},
    {"id": "/b.md#0", "embedding": [0.9, 0.1], "document": "b zero", "metadata": {"source": "/b.md", "chunk": 0}},
]

# ChromaStore's answer to `query([0.1, 0.2], 2)` over POINTS, recorded. The fake
# computes no similarity, so it is given the answer a real store computes.
NEAREST_TWO = [
    {"document": "a zero", "metadata": {"source": "/a.md", "chunk": 0}, "distance": 0.0},
    {"document": "a one", "metadata": {"source": "/a.md", "chunk": 1}, "distance": 0.20000004768371582},
]


@pytest.fixture(params=["fake", "chroma", "qdrant"])
def make_store(request, tmp_path, monkeypatch):
    """Opens an empty store of the implementation under test.

    `hits` reaches only the fake: its query answers with what it is given.
    """
    if request.param == "qdrant":
        monkeypatch.setattr(cfg, "qdrant_url", ":memory:")

    def make(collection: str | None = None, *, hits: list[dict] | None = None) -> VectorStore:
        if request.param == "fake":
            return FakeStore(collection, hits=hits)
        if request.param == "chroma":
            return ChromaStore(persist_dir=tmp_path / "chroma", collection=collection)
        return QdrantStore(collection=collection)

    return make


def _fill(store: VectorStore, points: list[dict] = POINTS) -> None:
    store.upsert(
        ids=[point["id"] for point in points],
        embeddings=[point["embedding"] for point in points],
        documents=[point["document"] for point in points],
        metadatas=[point["metadata"] for point in points],
    )


def _carries(hit: dict, point: dict) -> bool:
    """`hit` is `point` handed back: its text, and every metadata key that went in."""
    return hit["document"] == point["document"] and hit["metadata"].items() >= point["metadata"].items()


def test_a_store_is_opened_on_the_collection_it_was_asked_for(make_store):
    assert make_store("contract").collection_name == "contract"
    assert make_store().collection_name == cfg.collection


def test_a_new_store_holds_nothing(make_store):
    store = make_store("contract")

    assert store.count() == 0
    assert store.documents() == []
    assert store.sources() == {}
    assert store.get_source_chunk("/a.md", 0) is None
    assert store.query([0.1, 0.2], 3) == []


def test_what_is_written_is_read_back(make_store):
    store = make_store("contract")

    _fill(store)

    assert store.count() == 3
    assert store.sources() == {"/a.md": 2, "/b.md": 1}
    documents = store.documents()
    assert len(documents) == 3
    for point in POINTS:
        [hit] = [hit for hit in documents if hit["document"] == point["document"]]
        assert _carries(hit, point)
        assert hit["distance"] == 1.0
    chunk = store.get_source_chunk("/a.md", 1)
    assert _carries(chunk, POINTS[1])
    assert chunk["distance"] == 1.0
    assert store.get_source_chunk("/a.md", 7) is None


def test_writing_an_id_again_replaces_it(make_store):
    store = make_store("contract")
    _fill(store)

    _fill(store, [{**POINTS[1], "document": "a one, edited"}])

    assert store.count() == 3
    assert store.get_source_chunk("/a.md", 1)["document"] == "a one, edited"


def test_delete_source_drops_every_chunk_of_that_source_and_nothing_else(make_store):
    store = make_store("contract")
    _fill(store)
    assert store.sources() == {"/a.md": 2, "/b.md": 1}

    store.delete_source("/a.md")

    assert store.sources() == {"/b.md": 1}
    assert store.get_source_chunk("/a.md", 0) is None


def test_delete_source_matches_metadata_source_not_path(make_store):
    """The rule the identity tests' old fake broke: it deleted on `path`."""
    store = make_store("contract")
    moved = {"id": "/x.md#0", "embedding": [0.5, 0.5], "document": "x", "metadata": {"source": "/x.md", "path": "/y.md", "chunk": 0}}
    _fill(store, [moved])
    assert store.count() == 1

    store.delete_source("/y.md")
    assert store.count() == 1

    store.delete_source("/x.md")
    assert store.count() == 0


def test_deleting_a_source_the_store_does_not_hold_changes_nothing(make_store):
    store = make_store("contract")
    _fill(store)

    store.delete_source("/not-there.md")

    assert store.count() == 3


def test_reset_empties_the_store_and_leaves_it_usable(make_store):
    store = make_store("contract")
    _fill(store)
    assert store.count() == 3

    store.reset()

    assert store.count() == 0
    assert store.documents() == []
    assert store.sources() == {}
    assert store.collection_name == "contract"
    _fill(store)
    assert store.count() == 3


def test_query_answers_with_stored_chunks_nearest_first_and_at_most_k(make_store):
    store = make_store("contract", hits=NEAREST_TWO)
    _fill(store)

    hits = store.query([0.1, 0.2], 2)

    assert [hit["document"] for hit in hits] == ["a zero", "a one"]
    for hit, point in zip(hits, POINTS[:2], strict=True):
        assert _carries(hit, point)
    distances = [hit["distance"] for hit in hits]
    assert all(isinstance(distance, float) for distance in distances)
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0.0, abs=1e-6)


@pytest.fixture
def store(make_store, tmp_path, monkeypatch):
    """A store of the implementation under test, written to by `ingest_paths`."""
    from secondbrain import ingest as ingest_mod
    from secondbrain.keyword_index import KeywordIndex

    opened = make_store("identity_test")
    monkeypatch.setattr(ingest_mod, "Store", lambda *a, **kw: opened)
    monkeypatch.setattr(ingest_mod, "embed", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(ingest_mod, "_index", KeywordIndex(tmp_path / "keyword-index.sqlite3"))
    return opened


def test_deleting_one_copy_leaves_the_other_intact(tmp_path, store):
    """The whole point: a file still on disk keeps its content in the store.

    Moved from test_ingest_identity.py (SB-F-42), where it ran only against
    that file's own fake, so it proved the fake's delete rather than a store's.
    """
    from secondbrain.ingest import ingest_paths

    survivor = tmp_path / "a" / "CLAUDE.md"
    doomed = tmp_path / "b" / "CLAUDE.md"
    for path in (survivor, doomed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("@AGENTS.md\n")
    list(ingest_paths(survivor))
    list(ingest_paths(doomed))

    store.delete_source(str(doomed))

    remaining = [p["metadata"]["path"] for p in store.documents()]
    assert remaining == [str(survivor)]


def _public_methods(cls: type) -> set[str]:
    return {name for name, value in vars(cls).items() if not name.startswith("_") and inspect.isfunction(value)}


def test_the_fake_has_exactly_the_protocol_s_methods_with_the_same_parameters():
    assert _public_methods(FakeStore) == _public_methods(VectorStore)
    for name in _public_methods(VectorStore):
        fake = inspect.signature(getattr(FakeStore, name)).parameters
        declared = inspect.signature(getattr(VectorStore, name)).parameters
        assert list(fake) == list(declared), name
    # Beyond the methods: the Protocol's one property, and the fake's record of deletes.
    assert {name for name, value in vars(FakeStore).items() if isinstance(value, property)} == {
        "collection_name",
        "deleted",
    }


@pytest.mark.parametrize("real", [ChromaStore, QdrantStore])
def test_both_real_stores_have_everything_the_protocol_declares(real):
    declared = _public_methods(VectorStore) | {"collection_name"}

    assert declared <= {name for name in vars(real) if not name.startswith("_")}
