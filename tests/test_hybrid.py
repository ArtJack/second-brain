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
    store = FakeStore()

    hits = hybrid_retrieve(store, "list the testing principles", [0.1], 1, enabled=True)

    sources = [hit["metadata"]["source"] for hit in hits]
    assert sources == ["syllabus.md", "syllabus.md", "vector.md"]
    assert hits[0]["retrieval"] == "keyword"
    assert hits[1]["retrieval"] == "keyword-adjacent"
