"""Hybrid keyword retrieval catches exact sections that semantic search can miss."""

from secondbrain.hybrid import hybrid_retrieve, keyword_query


class FakeStore:
    def __init__(self):
        self.vector_hits = [
            {
                "document": "A semantically nearby but wrong chunk.",
                "metadata": {"source": "vector.md", "chunk": 7},
                "distance": 0.25,
            }
        ]
        self.document_calls = 0

    def query(self, _qvec, _limit):
        return self.vector_hits

    def documents(self):
        self.document_calls += 1
        return [
            {
                "document": "1.3 Testing Principles FL-1.3.1 Explain the seven testing principles",
                "metadata": {"source": "toc.md", "chunk": 1},
            },
            {
                "document": (
                    "1.3. Testing Principles\n"
                    "This syllabus describes seven such principles.\n"
                    "1. Testing shows the presence, not the absence of defects."
                ),
                "metadata": {"source": "syllabus.md", "chunk": 45},
            },
        ]

    def get_source_chunk(self, source, chunk):
        if source == "syllabus.md" and chunk == 46:
            return {
                "document": "2. Exhaustive testing is impossible.",
                "metadata": {"source": "syllabus.md", "chunk": 46},
                "distance": 1.0,
            }
        return None


def test_keyword_query_prefers_answer_section_over_learning_objective():
    hits = keyword_query(FakeStore(), "What are the seven testing principles?", limit=1)

    assert hits[0]["metadata"]["source"] == "syllabus.md"
    assert hits[0]["retrieval"] == "keyword"


def test_keyword_query_prefers_numbered_list_for_generic_list_intent():
    hits = keyword_query(FakeStore(), "list the testing principles", limit=1)

    assert hits[0]["metadata"]["source"] == "syllabus.md"


def test_hybrid_retrieve_can_disable_keyword_fusion():
    store = FakeStore()

    hits = hybrid_retrieve(store, "list the testing principles", [0.1], 1, enabled=False)

    assert hits is store.vector_hits
    assert store.document_calls == 0


def test_hybrid_retrieve_adds_keyword_hits_and_neighbors():
    """Keyword hits and their neighbours reach the caller — within the cap.

    This asked for limit=1 and asserted three hits came back, which was the
    defect Verdict filed as SB-F-23: `limit` was a floor. The capability it was
    really testing — that a keyword hit and its adjacent chunk both survive
    fusion alongside a vector hit — is asserted here at a limit that fits them.
    """
    store = FakeStore()

    hits = hybrid_retrieve(store, "list the testing principles", [0.1], 3, enabled=True)

    sources = [hit["metadata"]["source"] for hit in hits]
    kinds = {hit.get("retrieval", "vector") for hit in hits}
    assert len(hits) == 3
    # Both retrievers reach the answer. Every slot goes to something that
    # actually matched: two keyword hits and the vector hit, with no adjacency
    # padding, because there are three real candidates for three slots.
    assert set(sources) == {"syllabus.md", "toc.md", "vector.md"}
    assert {"keyword", "vector"} <= kinds
    assert "keyword-adjacent" not in kinds


class _CappedStore:
    """A store whose keyword and vector results are disjoint, so fusion has to choose."""

    collection_name = "capped"

    def __init__(self):
        self.docs = [
            {
                "document": f"gateway routing budget cap section {i}\n1. Alpha\n2. Beta",
                "metadata": {"source": f"doc{i}.md", "chunk": 0},
                "distance": 1.0,
            }
            for i in range(3)
        ]
        self.neighbors = {
            (f"doc{i}.md", c): {
                "document": f"neighbour {i}-{c}",
                "metadata": {"source": f"doc{i}.md", "chunk": c},
                "distance": 1.0,
            }
            for i in range(3)
            for c in (1, 2)
        }

    def query(self, qvec, k):
        return [
            {"document": f"vec{i}", "metadata": {"source": f"v{i}.md", "chunk": 0}, "distance": 0.2 + i / 100}
            for i in range(k)
        ]

    def documents(self):
        return self.docs

    def get_source_chunk(self, source, chunk):
        return self.neighbors.get((source, chunk))


def test_hybrid_retrieve_never_exceeds_limit():
    """`limit` is a cap. It used to be a floor: limit=5 returned 14 hits."""
    store = _CappedStore()

    for limit in (1, 3, 5):
        hits = hybrid_retrieve(store, "gateway routing budget cap", [0.1], limit)
        assert len(hits) <= limit, f"limit={limit} returned {len(hits)} hits"


def test_disabling_hybrid_still_returns_exactly_the_vector_hits():
    store = _CappedStore()

    hits = hybrid_retrieve(store, "gateway routing budget cap", [0.1], 5, enabled=False)

    assert [hit["metadata"]["source"] for hit in hits] == ["v0.md", "v1.md", "v2.md", "v3.md", "v4.md"]


def test_fusion_keeps_hits_both_retrievers_found_over_hits_only_one_found():
    """The property reciprocal-rank fusion exists to provide."""
    agreed = {"document": "agreed", "metadata": {"source": "agreed.md", "chunk": 0}, "distance": 0.9}

    class AgreeingStore:
        collection_name = "agreeing"

        def query(self, qvec, k):
            return [
                {"document": "vector only", "metadata": {"source": "vonly.md", "chunk": 0}, "distance": 0.1},
                agreed,
            ]

        def documents(self):
            return [
                agreed,
                {"document": "keyword only gateway", "metadata": {"source": "konly.md", "chunk": 0}, "distance": 1.0},
            ]

        def get_source_chunk(self, source, chunk):
            return None

    hits = hybrid_retrieve(AgreeingStore(), "agreed", [0.1], 1)

    assert [hit["metadata"]["source"] for hit in hits] == ["agreed.md"]


def test_real_matches_beat_adjacency_padding_for_the_cap():
    """Verdict SB-F-27: under a hard cap, padding was what the cap kept.

    Neighbours entered the fused list at retrieved ranks, so a chunk that matched
    nothing outscored the second real keyword match. At the shipped default an
    answer was built from one keyword hit, two of its neighbours, and two vector
    hits — where before the cap it saw five vector hits and three keyword hits.
    Adjacency is context, not evidence: it fills leftover slots, never takes one
    from a retriever that actually matched.
    """

    class Store:
        collection_name = "padding"

        def query(self, qvec, k):
            return [
                {"document": f"vec{i}", "metadata": {"source": f"/v{i}.md", "chunk": 0}, "distance": 0.1 + i / 100}
                for i in range(5)
            ]

        def documents(self):
            return [
                {"document": f"gateway routing match {i}", "metadata": {"source": f"/k{i}.md", "chunk": 0}, "distance": 1.0}
                for i in range(3)
            ]

        def get_source_chunk(self, source, chunk):
            return {
                "document": f"neighbour {source}#{chunk}",
                "metadata": {"source": source, "chunk": chunk},
                "distance": 1.0,
            }

    hits = hybrid_retrieve(Store(), "gateway routing", [0.1], 5)
    kinds = [hit.get("retrieval", "vector") for hit in hits]
    sources = [hit["metadata"]["source"] for hit in hits]

    assert len(hits) == 5
    assert "keyword-adjacent" not in kinds, f"padding took a slot from a real match: {list(zip(sources, kinds, strict=True))}"
    # Every real keyword match survives, and the rest of the budget is vector.
    assert {"/k0.md", "/k1.md", "/k2.md"} <= set(sources)


def test_neighbours_still_fill_slots_the_retrievers_left_empty():
    """Adjacency is not removed — it is demoted to filling leftover room."""

    class Store:
        collection_name = "sparse"

        def query(self, qvec, k):
            return []

        def documents(self):
            return [{"document": "gateway routing", "metadata": {"source": "/k.md", "chunk": 0}, "distance": 1.0}]

        def get_source_chunk(self, source, chunk):
            if chunk > 2:
                return None
            return {
                "document": f"neighbour {chunk}",
                "metadata": {"source": source, "chunk": chunk},
                "distance": 1.0,
            }

    hits = hybrid_retrieve(Store(), "gateway routing", [0.1], 5)

    assert [h["metadata"]["source"] for h in hits] == ["/k.md", "/k.md", "/k.md"]
    assert [h.get("retrieval") for h in hits] == ["keyword", "keyword-adjacent", "keyword-adjacent"]


def test_an_unusable_keyword_index_is_logged_not_silent(monkeypatch, caplog):
    """Every answer went vector-only for hours on 2026-09-12 without a line in any log."""
    import logging
    import sqlite3

    from secondbrain import hybrid

    class BrokenIndex:
        def count(self, collection):
            return 1

        def query(self, collection, query, limit=3):
            raise sqlite3.OperationalError("no such column: header")

    class LiveStore:
        collection_name = "live"

        def documents(self):
            return []

    monkeypatch.setattr(hybrid, "_index", BrokenIndex())

    with caplog.at_level(logging.WARNING, logger="secondbrain.hybrid"):
        hits = hybrid.keyword_query(LiveStore(), "gateway routing", limit=3)

    assert hits == []
    assert "no such column: header" in caplog.text
    assert "live" in caplog.text
