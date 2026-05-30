"""Regression tests for the orphaned-chunk bug.

Re-ingesting a file that shrank used to leave its old higher-index chunks behind,
because upsert only overwrites ids it is given. delete_source fixes that, and
ingest_paths must call it *before* upsert on every (non-reset) ingest.
"""
from secondbrain.store import ChromaStore


def _meta(n):
    return [{"source": "a.md", "name": "a.md", "chunk": i} for i in range(n)]


def test_delete_source_removes_every_chunk(tmp_path):
    store = ChromaStore(persist_dir=tmp_path, collection="reingest_test")
    store.upsert(
        ids=["a.md#0", "a.md#1", "a.md#2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        documents=["one", "two", "three"],
        metadatas=_meta(3),
    )
    assert store.count() == 3

    store.delete_source("a.md")
    assert store.count() == 0


def test_reingesting_a_shrunk_file_leaves_no_orphans(tmp_path):
    store = ChromaStore(persist_dir=tmp_path, collection="reingest_test")
    # First ingest: 3 chunks.
    store.upsert(
        ids=["a.md#0", "a.md#1", "a.md#2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        documents=["one", "two", "three"],
        metadatas=_meta(3),
    )
    # Re-ingest after the file shrank to a single chunk.
    store.delete_source("a.md")
    store.upsert(
        ids=["a.md#0"],
        embeddings=[[1.0, 0.0, 0.0]],
        documents=["only"],
        metadatas=_meta(1),
    )
    assert store.count() == 1  # not 3 — the orphans are gone


def test_ingest_paths_deletes_source_before_upsert(tmp_path, monkeypatch):
    """The pipeline must clear a file's prior chunks before writing new ones."""
    import secondbrain.ingest as ingest

    calls: list[tuple] = []

    class FakeStore:
        def reset(self):
            calls.append(("reset",))

        def delete_source(self, source):
            calls.append(("delete_source", source))

        def upsert(self, ids, embeddings, documents, metadatas):
            calls.append(("upsert", len(ids)))

        def count(self):
            return 0

    monkeypatch.setattr(ingest, "Store", lambda *a, **k: FakeStore())
    monkeypatch.setattr(ingest, "embed", lambda texts: [[0.0] for _ in texts])

    note = tmp_path / "note.md"
    note.write_text("hello world\n" * 40)

    list(ingest.ingest_paths(note))

    kinds = [c[0] for c in calls]
    assert "delete_source" in kinds and "upsert" in kinds
    assert kinds.index("delete_source") < kinds.index("upsert")
