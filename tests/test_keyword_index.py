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
