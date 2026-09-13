"""`recall` retrieves the way `ask` does: vector search fused with keyword search.

The MCP server tells every agent that `recall` is the cheapest way to ground
itself, and `ask` builds its answers from `hybrid_retrieve`. But `recall` called
`store.query` directly, so an agent grounding itself through it never got the
keyword half of retrieval at all. On 2026-09-12, against the live collection,
`recall` returned only vector-range distances for a question where hybrid
retrieval surfaced keyword matches.

Hermetic in the style of tests/test_hybrid.py: a vector store that only ever
finds nearby-but-wrong chunks, and a real keyword index in tmp_path. Nothing here
reaches a model, a gateway or a live store.
"""
from __future__ import annotations

import asyncio

import pytest

from secondbrain.config import cfg
from secondbrain.keyword_index import KeywordIndex

# Obviously synthetic, with no semantic neighbours, so only keyword search finds
# it. Both of its parts survive query tokenization. An id built from very short
# parts ranks differently, and these tests do not cover it.
IDENTIFIER = "XQZ-40417"


class NearbyButWrongStore:
    """Opened the way `Store()` is; its vector search never finds the identifier."""

    def __init__(self, persist_dir=None, collection=None):
        self.collection_name = collection or cfg.collection

    def count(self):
        return 1

    def query(self, _qvec, k):
        return [
            {
                "document": f"A semantically nearby but wrong chunk {i}.",
                "metadata": {"source": f"/notes/nearby-{i}.md", "chunk": 0},
                "distance": round(0.43 + i / 100, 4),
            }
            for i in range(k)
        ]

    def documents(self):
        return []

    def get_source_chunk(self, _source, _chunk):
        return None


@pytest.fixture(autouse=True)
def _offline_engine(monkeypatch):
    from secondbrain import ask as ask_module

    monkeypatch.setattr(ask_module, "Store", NearbyButWrongStore)
    monkeypatch.setattr(ask_module, "embed", lambda texts: [[0.1, 0.2] for _ in texts])


@pytest.fixture()
def index(tmp_path, monkeypatch) -> KeywordIndex:
    from secondbrain import hybrid

    index = KeywordIndex(tmp_path / "keyword-index.sqlite3")
    monkeypatch.setattr(hybrid, "_index", index)
    return index


def _row(source: str, document: str) -> dict:
    return {"source": source, "chunk": 0, "name": source.rsplit("/", 1)[-1], "document": document}


def test_the_mcp_recall_tool_returns_a_match_only_keyword_search_finds(index):
    """The defect, through the tool an agent actually calls rather than the engine function."""
    from secondbrain import mcp_server

    index.upsert_chunks(cfg.collection, [_row("/notes/parser.md", f"{IDENTIFIER} appears in the parser notes.")])

    out = asyncio.run(mcp_server.recall(IDENTIFIER))

    sources = [hit["source"] for hit in out["hits"]]
    assert "/notes/parser.md" in sources, f"recall never ran keyword search: {sources}"


def test_recall_keeps_its_shape_and_its_cap_with_keyword_hits_in_the_mix(index):
    """MCP clients read `count` and `hits` of exactly source, distance and text.

    A keyword hit carries more than a vector hit does — a `retrieval` label, the
    section header, the chunk name — and none of it may reach the tool's output.
    """
    from secondbrain import ask as ask_module

    index.upsert_chunks(
        cfg.collection,
        [_row(f"/notes/match-{i}.md", f"{IDENTIFIER} appears in note {i}.") for i in range(4)],
    )

    out = ask_module.recall(IDENTIFIER, top_k=2)

    assert set(out) == {"count", "hits"}
    assert out["count"] == len(out["hits"]) == 2, "top_k is a cap on the fused result, not a floor"
    assert any(hit["source"].startswith("/notes/match-") for hit in out["hits"]), "no keyword hit came back"
    for hit in out["hits"]:
        assert set(hit) == {"source", "distance", "text"}
        assert hit["distance"] == round(hit["distance"], 4)


def test_recall_is_vector_only_when_hybrid_is_switched_off(index, monkeypatch):
    """`SB_HYBRID=0` is the owner's switch, and recall obeys it the way ask does."""
    from secondbrain import ask as ask_module
    from secondbrain import hybrid

    index.upsert_chunks(cfg.collection, [_row("/notes/parser.md", f"{IDENTIFIER} appears in the parser notes.")])
    monkeypatch.setattr(cfg, "hybrid_enabled", False)
    monkeypatch.setattr(hybrid, "keyword_query", lambda *a, **k: pytest.fail("keyword search ran with SB_HYBRID off"))

    out = ask_module.recall(IDENTIFIER)

    assert [hit["source"] for hit in out["hits"]] == [f"/notes/nearby-{i}.md" for i in range(cfg.top_k)]


def test_public_recall_never_surfaces_a_keyword_match_from_the_owners_collection(index):
    """The corpus boundary, on the read path this change adds.

    One keyword index file holds every collection's rows, and the web API's
    `/recall` route serves anonymous traffic on the public corpus. Checked
    through the route itself, so the collection routing is the real one.
    """
    from fastapi.testclient import TestClient

    from secondbrain import server

    index.upsert_chunks(cfg.collection, [_row("/owner/private-note.md", f"{IDENTIFIER} in the owner's notes.")])
    index.upsert_chunks("second_brain_public", [_row("/public/demo.md", f"{IDENTIFIER} in the public demo.")])

    response = TestClient(server.app).post("/recall", json={"query": IDENTIFIER, "top_k": 5, "corpus": "public"})

    assert response.status_code == 200
    sources = [hit["source"] for hit in response.json()["hits"]]
    assert "/public/demo.md" in sources, "the public corpus's own keyword match is missing, so this proves nothing"
    assert "/owner/private-note.md" not in sources
