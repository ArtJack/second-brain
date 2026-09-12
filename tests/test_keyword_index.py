"""A persisted keyword index, and the two defects that came with not having one.

Keyword search used to pull every chunk out of the store on every query and score
them in Python: 7.2 MB over the network and about a second of work before the
embedding call had even started, growing with the corpus. It also could not see
half of what the owner writes — the tokenizer was `[a-z0-9]+`, so a Russian
question tokenized to nothing and got no keyword rescue at all.

FTS5 with `unicode61` fixes both at once: the index lives on disk beside the task
database, it is updated incrementally at ingest, and it tokenizes text rather
than ASCII. The fallback matters as much as the feature — a collection indexed
before this existed has no rows here, and must keep working.
"""
from __future__ import annotations

import pytest

from secondbrain.keyword_index import KeywordIndex


@pytest.fixture()
def index(tmp_path) -> KeywordIndex:
    return KeywordIndex(tmp_path / "keyword-index.sqlite3")


def _rows(*items):
    return [
        {"source": source, "chunk": chunk, "document": document, "name": source.rsplit("/", 1)[-1]}
        for source, chunk, document in items
    ]


def test_an_exact_phrase_outranks_a_passing_mention(index):
    index.upsert_chunks(
        "c",
        _rows(
            ("/notes/gateway.md", 0, "The LiteLLM gateway routes every model call through one endpoint."),
            ("/notes/misc.md", 0, "We mention a gateway once here, in passing, among other things."),
        ),
    )

    hits = index.query("c", "LiteLLM gateway routes", limit=2)

    assert hits[0]["metadata"]["source"] == "/notes/gateway.md"
    assert hits[0]["metadata"]["chunk"] == 0
    assert hits[0]["metadata"]["name"] == "gateway.md"
    assert 0 < hits[0]["distance"] <= 1


def test_a_cyrillic_question_finds_a_cyrillic_document(index):
    """The old tokenizer produced an empty token list for this and returned nothing."""
    index.upsert_chunks(
        "c",
        _rows(
            ("/qa/course.md", 0, "Цель курса — сделать из студентов QA-автоматизаторов."),
            ("/qa/other.md", 0, "Unrelated English text about deployment pipelines."),
        ),
    )

    hits = index.query("c", "какая цель курса", limit=2)

    assert hits, "a Cyrillic query must return Cyrillic matches"
    assert hits[0]["metadata"]["source"] == "/qa/course.md"


def test_deleting_a_source_removes_exactly_its_chunks(index):
    index.upsert_chunks(
        "c",
        _rows(
            ("/a.md", 0, "alpha gateway"),
            ("/a.md", 1, "alpha gateway again"),
            ("/b.md", 0, "beta gateway"),
        ),
    )

    index.delete_source("c", "/a.md")
    hits = index.query("c", "gateway", limit=10)

    assert [hit["metadata"]["source"] for hit in hits] == ["/b.md"]


def test_re_upserting_a_chunk_replaces_it_rather_than_duplicating(index):
    index.upsert_chunks("c", _rows(("/a.md", 0, "first version mentions qdrant")))
    index.upsert_chunks("c", _rows(("/a.md", 0, "second version mentions qdrant")))

    hits = index.query("c", "qdrant", limit=10)

    assert len(hits) == 1
    assert "second version" in hits[0]["document"]


def test_collections_do_not_see_each_other(index):
    index.upsert_chunks("personal", _rows(("/p.md", 0, "gateway routing decision")))
    index.upsert_chunks("reference", _rows(("/r.md", 0, "gateway routing chapter")))

    assert [h["metadata"]["source"] for h in index.query("personal", "gateway", limit=5)] == ["/p.md"]
    assert [h["metadata"]["source"] for h in index.query("reference", "gateway", limit=5)] == ["/r.md"]


def test_an_unknown_collection_is_empty_not_an_error(index):
    assert index.query("never-indexed", "anything", limit=5) == []
    assert index.count("never-indexed") == 0


def test_reset_empties_one_collection_only(index):
    index.upsert_chunks("a", _rows(("/a.md", 0, "gateway")))
    index.upsert_chunks("b", _rows(("/b.md", 0, "gateway")))

    index.reset("a")

    assert index.count("a") == 0
    assert index.count("b") == 1


def test_a_query_of_only_punctuation_returns_nothing_rather_than_raising(index):
    """FTS5 raises on malformed MATCH expressions; user text must never reach it raw."""
    index.upsert_chunks("c", _rows(("/a.md", 0, "gateway routing")))

    assert index.query("c", '"', limit=5) == []
    assert index.query("c", "AND OR NOT", limit=5) == []
    assert index.query("c", "", limit=5) == []


def test_keyword_query_falls_back_when_the_collection_was_never_indexed(tmp_path, monkeypatch):
    """Legacy collections predate the index and must keep working unchanged."""
    from secondbrain import hybrid

    monkeypatch.setattr(hybrid, "_index", KeywordIndex(tmp_path / "empty.sqlite3"))

    class LegacyStore:
        collection_name = "legacy"
        scanned = 0

        def documents(self):
            LegacyStore.scanned += 1
            return [
                {"document": "the gateway routes model calls", "metadata": {"source": "/a.md", "chunk": 0}, "distance": 1.0}
            ]

    hits = hybrid.keyword_query(LegacyStore(), "gateway routes", limit=3)

    assert LegacyStore.scanned == 1, "an unindexed collection must still be scanned"
    assert hits and hits[0]["metadata"]["source"] == "/a.md"


def test_keyword_query_prefers_the_index_over_scanning(tmp_path, monkeypatch):
    from secondbrain import hybrid

    index = KeywordIndex(tmp_path / "populated.sqlite3")
    index.upsert_chunks("indexed", _rows(("/indexed.md", 0, "the gateway routes model calls")))
    monkeypatch.setattr(hybrid, "_index", index)

    class IndexedStore:
        collection_name = "indexed"
        scanned = 0

        def documents(self):
            IndexedStore.scanned += 1
            return []

    hits = hybrid.keyword_query(IndexedStore(), "gateway routes", limit=3)

    assert IndexedStore.scanned == 0, "the whole point is not to scan"
    assert hits[0]["metadata"]["source"] == "/indexed.md"


def test_rebuilding_backfills_a_collection_ingested_before_the_index_existed(tmp_path):
    from secondbrain.keyword_index import rebuild_from_store

    class LegacyStore:
        collection_name = "legacy"

        def documents(self):
            return [
                {"document": "the gateway routes model calls", "metadata": {"source": "/a.md", "chunk": 0, "name": "a.md"}},
                {"document": "unrelated text", "metadata": {"source": "/b.md", "chunk": 0, "name": "b.md"}},
                # A chunk with no source cannot be cited, so it is not indexed.
                {"document": "orphan", "metadata": {}},
            ]

    index = KeywordIndex(tmp_path / "backfill.sqlite3")
    written = rebuild_from_store(LegacyStore(), index=index)

    assert written == 2
    assert index.count("legacy") == 2
    assert index.query("legacy", "gateway routes", limit=3)[0]["metadata"]["source"] == "/a.md"


def test_rebuilding_replaces_rather_than_accumulates(tmp_path):
    from secondbrain.keyword_index import rebuild_from_store

    class Store:
        collection_name = "c"

        def documents(self):
            return [{"document": "gateway", "metadata": {"source": "/a.md", "chunk": 0, "name": "a.md"}}]

    index = KeywordIndex(tmp_path / "twice.sqlite3")
    rebuild_from_store(Store(), index=index)
    rebuild_from_store(Store(), index=index)

    assert index.count("c") == 1


def test_stopwords_are_dropped_so_common_words_do_not_dilute_the_ranking(index):
    """Measured regression: keeping them moved a correct answer from rank 1 to rank 3.

    The match expression joins terms with OR, so a question's "what", "is" and
    "the" otherwise make every document in the corpus a candidate and spread BM25
    across all of them.
    """
    index.upsert_chunks(
        "c",
        _rows(
            ("/right.md", 0, "Invoice format: PDF with line items, one per delivery."),
            ("/noise.md", 0, "What is the thing that is in the place with the other things?"),
        ),
    )

    hits = index.query("c", "what is my invoice format", limit=2)

    assert hits[0]["metadata"]["source"] == "/right.md"


def test_a_question_made_only_of_stopwords_still_searches(index):
    """Filtering must not turn a real question into an empty one."""
    index.upsert_chunks("c", _rows(("/a.md", 0, "what is the what")))

    assert index.query("c", "what is the", limit=2)


def test_cyrillic_terms_survive_the_english_stopword_filter(index):
    from secondbrain.keyword_index import query_terms

    assert query_terms("какая цель курса") == ["какая", "цель", "курса"]


def test_the_index_uses_wal_so_a_long_write_does_not_lock_out_readers(tmp_path):
    """Verdict SB-F-29: making the MCP tools async made these genuinely concurrent.

    In rollback-journal mode a write takes an EXCLUSIVE lock for the whole
    transaction and readers wait only busy_timeout before raising. A 20,000-chunk
    ingest held it for 66 s, and a concurrent `ask` failed with "database is
    locked" — as a hard tool error on the user-facing path.
    """
    import sqlite3

    index = KeywordIndex(tmp_path / "wal.sqlite3")
    index.upsert_chunks("c", _rows(("/a.md", 0, "gateway")))

    with sqlite3.connect(index.path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal", f"journal_mode is {mode}: one writer still blocks every reader"


def test_a_reader_works_while_a_writer_holds_a_transaction(tmp_path):
    """The property, not the pragma."""
    import sqlite3
    import threading

    index = KeywordIndex(tmp_path / "concurrent.sqlite3")
    index.upsert_chunks("c", _rows(("/seed.md", 0, "gateway routing")))

    holding = threading.Event()
    release = threading.Event()
    errors: list[Exception] = []

    def writer():
        conn = sqlite3.connect(index.path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO kw_c (source, chunk, name, document) VALUES ('/w.md', 0, 'w.md', 'gateway')")
            holding.set()
            release.wait(timeout=5)
            conn.commit()
        except Exception as exc:
            errors.append(exc)
        finally:
            conn.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert holding.wait(timeout=5), "writer never acquired its transaction"
        hits = index.query("c", "gateway routing", limit=5)
    finally:
        release.set()
        thread.join(timeout=10)

    assert not errors, errors
    assert hits, "a reader must not be locked out by an in-flight write"


def test_replacing_a_file_does_not_scan_once_per_chunk(tmp_path):
    """The write cost was quadratic: one full-table DELETE per chunk, because
    source and chunk are UNINDEXED in FTS5 and have no b-tree behind them."""
    import time

    index = KeywordIndex(tmp_path / "scale.sqlite3")
    for batch in range(8):
        index.upsert_chunks(
            "c",
            [
                {"source": f"/f{batch}.md", "chunk": i, "name": "f.md", "document": f"chunk {batch}-{i} gateway"}
                for i in range(500)
            ],
        )

    start = time.perf_counter()
    index.upsert_chunks(
        "c",
        [{"source": "/f0.md", "chunk": i, "name": "f.md", "document": f"rewritten {i}"} for i in range(500)],
    )
    elapsed = time.perf_counter() - start

    assert index.count("c") == 4000, "replacing a file must not duplicate or drop rows"
    assert elapsed < 2.0, f"replacing 500 rows in a 4,000-row table took {elapsed:.2f}s"


def test_an_index_error_degrades_to_vector_search_rather_than_failing_the_answer(tmp_path, monkeypatch):
    """Keyword search is an enhancement over vector search, not a dependency.

    `keyword_query` called the index with no handling, so a SQLite error — the
    `database is locked` that SB-F-29 describes — propagated through
    hybrid_retrieve and ask() to the MCP client as a hard tool error. Losing the
    keyword half of a hybrid answer is a worse answer; losing the answer is an
    outage.
    """
    import sqlite3

    from secondbrain import hybrid

    class BrokenIndex:
        def count(self, collection):
            return 5

        def query(self, collection, query, limit=3):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(hybrid, "_index", BrokenIndex())

    class Store:
        collection_name = "c"

        def query(self, qvec, k):
            return [{"document": "vec", "metadata": {"source": "/v.md", "chunk": 0}, "distance": 0.2}]

        def documents(self):
            return []

    assert hybrid.keyword_query(Store(), "gateway", limit=3) == []

    hits = hybrid.hybrid_retrieve(Store(), "gateway", [0.1], 5)
    assert [h["metadata"]["source"] for h in hits] == ["/v.md"], "the answer must survive a broken index"


def test_rebuilding_keeps_every_chunk_of_a_file_that_spans_a_batch(tmp_path):
    """A large file lost all but its last batch of chunks during backfill.

    `upsert_chunks` replaces a source wholesale — correct for ingest, which hands
    over one file's chunks in a single call. The backfill flushes every N rows
    regardless of which file they came from, so a file bigger than one batch had
    its already-written rows deleted by its own next batch. Measured on the live
    corpus: 79 of 346 files were truncated and the index held 2,695 of 3,934
    chunks, with the largest file down to 10 chunks from 331. Nothing failed —
    keyword search just stopped being able to see two thirds of a long document.
    """
    from secondbrain.keyword_index import rebuild_from_store

    class BigFileStore:
        collection_name = "big"

        def documents(self):
            for chunk in range(25):
                yield {
                    "document": f"gateway routing section {chunk}",
                    "metadata": {"source": "/big.md", "chunk": chunk, "name": "big.md"},
                }

    index = KeywordIndex(tmp_path / "spanning.sqlite3")
    written = rebuild_from_store(BigFileStore(), index=index, batch_size=10)

    assert written == 25
    assert index.count("big") == 25, "a file must not delete its own earlier batches"


def test_rebuilding_still_replaces_a_source_left_over_from_a_previous_index(tmp_path):
    """The fix must not turn the backfill into an append."""
    from secondbrain.keyword_index import rebuild_from_store

    class Store:
        collection_name = "c"

        def documents(self):
            return [{"document": "gateway", "metadata": {"source": "/a.md", "chunk": 0, "name": "a.md"}}]

    index = KeywordIndex(tmp_path / "stale.sqlite3")
    index.upsert_chunks("c", [{"source": "/a.md", "chunk": 0, "name": "a.md", "document": "old text"}])
    index.upsert_chunks("c", [{"source": "/gone.md", "chunk": 0, "name": "gone.md", "document": "deleted file"}])

    rebuild_from_store(Store(), index=index)

    assert index.count("c") == 1
    assert index.query("c", "deleted file", limit=3) == []


def test_an_index_built_on_an_older_schema_is_rebuilt_rather_than_failing(tmp_path):
    """The index is a derived cache, so a schema change may drop it — silently failing is not an option.

    Adding the contextual header as its own searchable column changes the FTS5
    table. `CREATE TABLE IF NOT EXISTS` keeps whatever is already on disk, so
    without this every insert against a pre-existing index raises "table kw_c
    has 4 columns but 5 values were supplied" — during the nightly, on the
    deployed machine, at 03:15.
    """
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE VIRTUAL TABLE kw_c USING fts5("
            "source UNINDEXED, chunk UNINDEXED, name UNINDEXED, document, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        conn.execute("INSERT INTO kw_c VALUES ('/old.md', 0, 'old.md', 'stale text')")

    index = KeywordIndex(path)
    index.upsert_chunks("c", _rows(("/new.md", 0, "gateway routing")))

    assert index.count("c") == 1, "the stale rows go with the old schema; a reindex repopulates"
    assert index.query("c", "gateway routing", limit=3)[0]["metadata"]["source"] == "/new.md"


def test_the_header_is_searchable_but_is_not_what_a_citation_quotes(tmp_path):
    index = KeywordIndex(tmp_path / "headers.sqlite3")
    index.upsert_chunks(
        "c",
        [
            {
                "source": "/runbook.md",
                "chunk": 3,
                "name": "runbook.md",
                "header": "Gateway runbook › Restarting",
                "document": "Stop the launchd job and wait for the port to close.",
            }
        ],
    )

    # The header's words find the chunk even though the body never says them.
    hits = index.query("c", "gateway runbook restarting", limit=3)

    assert hits and hits[0]["metadata"]["source"] == "/runbook.md"
    assert hits[0]["document"] == "Stop the launchd job and wait for the port to close."
    assert "Gateway runbook" not in hits[0]["document"]
    assert hits[0]["metadata"]["header"] == "Gateway runbook › Restarting"
