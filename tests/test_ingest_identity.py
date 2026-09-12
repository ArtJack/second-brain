"""Chunk identity was the filesystem path captured at ingest time.

`f"{src}#{i}"` as the point id, and `delete_source(src)` as the only
de-duplication, made location and identity the same thing. Two confirmed
consequences.

`POST /ingest` wrote the request body into a `TemporaryDirectory` and ingested
that file, so every request produced a fresh `/var/folders/...` source. The same
document posted twice never matched `delete_source`, so duplicates accumulated
without limit, and the citation the reader saw pointed at a path that was
deleted the moment the request ended. `sb gc` makes that worse rather than
better: those paths genuinely do not exist, so the sweep now treats web-ingested
material as deleted.

And moving or copying a file re-ingested all of its chunks as new points, which
is how 31% of the live collection came to cite files that cannot be opened.

Identity is now the normalised chunk text. Location moves into the payload,
where it belongs and where it can change without creating a second copy.
"""
from __future__ import annotations

import pytest


class FakeStore:
    collection_name = "identity_test"

    def __init__(self):
        self.points: dict[str, dict] = {}
        self.deleted: list[str] = []

    def upsert(self, ids, embeddings, documents, metadatas):
        for point_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.points[point_id] = {"document": document, "metadata": metadata}

    def delete_source(self, source):
        self.deleted.append(source)
        for point_id in [
            pid for pid, point in self.points.items() if point["metadata"].get("path") == source
        ]:
            del self.points[point_id]

    def count(self):
        return len(self.points)

    def reset(self):
        self.points.clear()


@pytest.fixture
def store(monkeypatch):
    from secondbrain import ingest as ingest_mod

    fake = FakeStore()
    monkeypatch.setattr(ingest_mod, "Store", lambda *a, **kw: fake)
    monkeypatch.setattr(ingest_mod, "embed", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(ingest_mod._index, "upsert_chunks", lambda *a, **kw: None)
    monkeypatch.setattr(ingest_mod._index, "delete_source", lambda *a, **kw: None)
    monkeypatch.setattr(ingest_mod._index, "reset", lambda *a, **kw: None)
    return fake


class TestIdentityIsContent:
    def test_the_same_file_ingested_twice_is_one_point(self, tmp_path, store):
        """Identity within a source. Across sources is deliberately not collapsed.

        The first version of this test asserted that a *copy under a second
        path* was one point too. That was the requested behaviour and it turned
        out to be unsafe: see TestIdentityIsScopedToItsSource below, and Verdict
        SB-F-42.
        """
        from secondbrain.ingest import ingest_paths

        note = tmp_path / "a" / "note.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text("# Note\n\nThe gateway binds loopback only.\n")

        list(ingest_paths(note))
        list(ingest_paths(note))

        assert store.count() == 1, "re-ingesting a file must not double the corpus"

    def test_whitespace_differences_do_not_make_a_second_point(self, tmp_path, store):
        """Normalising before hashing is what makes the id stable across reflows.

        An editor that rewraps a paragraph must not turn one chunk into two.
        Asserted within one source, which is the scope identity now has.
        """
        from secondbrain.ingest import ingest_paths

        note = tmp_path / "one.md"
        note.write_text("The gateway binds loopback only.")
        list(ingest_paths(note))

        note.write_text("The   gateway\n  binds loopback   only.   ")
        list(ingest_paths(note))

        assert store.count() == 1

    def test_different_text_is_a_different_point(self, tmp_path, store):
        from secondbrain.ingest import ingest_paths

        first = tmp_path / "one.md"
        second = tmp_path / "two.md"
        first.write_text("The gateway binds loopback only.")
        second.write_text("The gateway is reachable from the LAN.")

        list(ingest_paths(first))
        list(ingest_paths(second))

        assert store.count() == 2

    def test_an_id_is_a_readable_content_hash(self, tmp_path, store):
        from secondbrain.ingest import ingest_paths

        note = tmp_path / "note.md"
        note.write_text("The gateway binds loopback only.")

        list(ingest_paths(note))

        point_id = next(iter(store.points))
        assert point_id.startswith("c:")
        assert len(point_id) == len("c:") + 32
        assert str(tmp_path) not in point_id, "the id must not carry a location"

    def test_editing_a_file_to_produce_fewer_chunks_leaves_no_orphans(self, tmp_path, store):
        from secondbrain.ingest import ingest_paths

        note = tmp_path / "note.md"
        note.write_text("alpha. " * 400)
        list(ingest_paths(note))
        assert store.count() > 1

        note.write_text("alpha.")
        list(ingest_paths(note))

        assert store.count() == 1


class TestPayload:
    def test_location_and_provenance_move_into_the_payload(self, tmp_path, store):
        from secondbrain.ingest import ingest_paths

        note = tmp_path / "runbook.md"
        note.write_text("# Gateway runbook\n\nIt binds loopback only.\n")

        list(ingest_paths(note))

        meta = next(iter(store.points.values()))["metadata"]
        assert meta["path"] == str(note)
        assert meta["source"] == str(note), "kept as an alias so every read site keeps working"
        assert meta["doc_type"] == "md"
        assert meta["title"] == "Gateway runbook"
        assert isinstance(meta["mtime"], float)
        assert meta["ingested_at"].endswith("Z") or "+00:00" in meta["ingested_at"]
        assert meta["chunk"] == 0

    def test_a_document_with_no_heading_titles_itself_by_file_stem(self, tmp_path, store):
        """The stem, not the full file name, and deliberately only one title.

        The queue asks for two slightly different fallbacks in two tasks: "the
        file stem" for the contextual header, and "the file name" for this
        payload. Two titles for one document would be a bug waiting to be
        found, and the stem is the one already shipped and read by the header,
        where "scratch" orients a reader better than "scratch.txt". The
        extension is not lost — it is `doc_type`.
        """
        from secondbrain.ingest import ingest_paths

        note = tmp_path / "scratch.txt"
        note.write_text("no headings at all here")

        list(ingest_paths(note))

        meta = next(iter(store.points.values()))["metadata"]
        assert meta["title"] == "scratch"
        assert meta["name"] == "scratch.txt"
        assert meta["doc_type"] == "txt"


class TestIngestText:
    def test_text_is_ingested_under_the_caller_s_logical_source(self, store):
        from secondbrain.ingest import ingest_text

        chunks = ingest_text(
            "The gateway binds loopback only.",
            source="https://example.com/runbook",
            doc_type="url",
        )

        assert chunks == 1
        meta = next(iter(store.points.values()))["metadata"]
        assert meta["path"] == "https://example.com/runbook"
        assert meta["source"] == "https://example.com/runbook"
        assert meta["doc_type"] == "url"
        assert "/var/folders" not in meta["path"]

    def test_posting_the_same_document_twice_does_not_grow_the_collection(self, store):
        from secondbrain.ingest import ingest_text

        for _ in range(2):
            ingest_text("The gateway binds loopback only.", source="https://example.com/runbook")

        assert store.count() == 1

    def test_re_posting_a_shortened_document_drops_its_old_chunks(self, store):
        from secondbrain.ingest import ingest_text

        ingest_text("alpha. " * 400, source="doc://one")
        assert store.count() > 1

        ingest_text("alpha.", source="doc://one")

        assert store.count() == 1

    def test_empty_text_ingests_nothing_rather_than_raising(self, store):
        from secondbrain.ingest import ingest_text

        assert ingest_text("   \n  ", source="doc://empty") == 0
        assert store.count() == 0


class TestIdentityIsScopedToItsSource:
    """SB-F-42: identity keyed on content, deletion keyed on path.

    Task 3 asked for identical content from two paths to land on one point, and
    it did. What it did not ask for, and what fell out, is that identity and
    deletion then key on different things. The second file to be ingested
    overwrites the point's `path`, so the first file's claim on that content is
    gone. Delete the second file and `delete_source` removes the shared point —
    and the first file, still sitting on disk, silently loses its content from
    the store while the keyword index keeps a row for it. Verdict proved it with
    two projects' `CLAUDE.md`, which really are byte-identical here.

    It does not self-heal. The nightly only re-ingests files that changed, and
    the surviving file did not change.

    So identity is content *within a source*. That keeps everything the P0
    needed — a document posted twice to the web endpoint is one point, because
    the logical source is stable, and re-ingesting a file still replaces rather
    than duplicates — and gives up cross-path de-duplication, which is the part
    that conflicted. True cross-path sharing needs a point to record every path
    that claims it, and that is a different change.
    """

    def test_the_same_text_from_two_paths_is_two_points(self, tmp_path, store):
        from secondbrain.ingest import ingest_paths

        first = tmp_path / "a" / "CLAUDE.md"
        second = tmp_path / "b" / "CLAUDE.md"
        for path in (first, second):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("@AGENTS.md\n")

        list(ingest_paths(first))
        list(ingest_paths(second))

        assert store.count() == 2
        assert {p["metadata"]["path"] for p in store.points.values()} == {str(first), str(second)}

    def test_deleting_one_copy_leaves_the_other_intact(self, tmp_path, store):
        """The whole point: a file still on disk keeps its content in the store."""
        from secondbrain.ingest import ingest_paths

        survivor = tmp_path / "a" / "CLAUDE.md"
        doomed = tmp_path / "b" / "CLAUDE.md"
        for path in (survivor, doomed):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("@AGENTS.md\n")
        list(ingest_paths(survivor))
        list(ingest_paths(doomed))

        store.delete_source(str(doomed))

        remaining = [p["metadata"]["path"] for p in store.points.values()]
        assert remaining == [str(survivor)]

    def test_re_ingesting_one_file_still_replaces_rather_than_duplicates(self, tmp_path, store):
        from secondbrain.ingest import ingest_paths

        note = tmp_path / "note.md"
        note.write_text("the gateway binds loopback only")
        list(ingest_paths(note))
        list(ingest_paths(note))

        assert store.count() == 1

    def test_the_same_logical_source_posted_twice_is_still_one_point(self, store):
        """The P0 this feature existed for, unchanged."""
        from secondbrain.ingest import ingest_text

        for _ in range(2):
            ingest_text("posted text", source="https://example.com/doc")

        assert store.count() == 1

    def test_an_id_still_changes_when_the_content_changes(self, tmp_path, store):
        from secondbrain.ingest import chunk_id

        assert chunk_id("same words", "/a.md") != chunk_id("other words", "/a.md")
        assert chunk_id("same words", "/a.md") != chunk_id("same words", "/b.md")
        assert chunk_id("same  words", "/a.md") == chunk_id("same words", "/a.md")
