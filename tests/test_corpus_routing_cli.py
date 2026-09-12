"""Reference material and personal notes are different corpora, and the local
surfaces should be able to say which one they mean.

The web API has had this since the public demo: a request names a corpus and
`_collection_for` maps it to a collection. The CLI and the MCP server never got
the equivalent, so everything the owner ingests lands in one collection and
competes in one ranking. That is not a hypothetical cost: after the 2026-09-12
sweep, 2,935 of 4,483 chunks were ISTQB syllabus PDFs, and a question about this
system's own architecture spent two of its five slots on them.

The default must not move. An `sb ask` with no `--corpus` has to hit exactly the
collection it hit before this existed, because that is the command the owner
types every day and the one the MCP server calls on their behalf.
"""
from __future__ import annotations

import pytest

from secondbrain.config import cfg, collections_for


@pytest.fixture(autouse=True)
def _no_ambient_collection_flag(monkeypatch):
    """`sb --collection X` sets a module global that outlives the command.

    Another test in this suite invokes the CLI with that flag, and the global it
    leaves behind would make these assertions pass or fail by collection order.
    """
    import secondbrain.cli as cli

    monkeypatch.setattr(cli, "_COLLECTION", None)


def test_personal_is_the_default_and_resolves_to_the_configured_collection():
    assert collections_for("personal") == [cfg.collection]
    assert collections_for(None) == [cfg.collection]


def test_reference_resolves_to_the_reference_collection():
    assert collections_for("reference") == [cfg.reference_collection]


def test_all_resolves_to_both_with_personal_first():
    assert collections_for("all") == [cfg.collection, cfg.reference_collection]


def test_an_unknown_corpus_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        collections_for("everything")

    assert "everything" in str(excinfo.value)
    assert "personal" in str(excinfo.value)


def test_ask_without_a_corpus_queries_exactly_what_it_queried_before(monkeypatch):
    """The regression this whole change must not cause."""
    import secondbrain.cli as cli

    seen: list[object] = []
    monkeypatch.setattr(
        cli,
        "ask_fn",
        lambda question, k=None, collection=None, **kw: seen.append(collection)
        or {"answer": "x", "sources": [], "invalid_citations": [], "grounding": {}},
    )

    cli.ask(question="anything", k=None, corpus="personal")

    assert seen == [[cfg.collection]]


def test_ask_with_reference_queries_the_reference_collection(monkeypatch):
    import secondbrain.cli as cli

    seen: list[object] = []
    monkeypatch.setattr(
        cli,
        "ask_fn",
        lambda question, k=None, collection=None, **kw: seen.append(collection)
        or {"answer": "x", "sources": [], "invalid_citations": [], "grounding": {}},
    )

    cli.ask(question="anything", k=None, corpus="reference")

    assert seen == [[cfg.reference_collection]]


def test_recall_forwards_the_corpus_too(monkeypatch):
    import secondbrain.cli as cli

    seen: list[object] = []
    monkeypatch.setattr(
        cli,
        "recall_fn",
        lambda query, top_k=0, collection=None, **kw: seen.append(collection) or {"count": 0, "hits": []},
    )

    cli.recall(query="anything", top_k=0, corpus="reference")

    assert seen == [[cfg.reference_collection]]


def test_the_mcp_ask_tool_forwards_its_corpus(monkeypatch):
    import secondbrain.mcp_server as mcp_server

    seen: list[object] = []

    def fake_ask(question, k=None, collection=None, **kw):
        seen.append(collection)
        return {
            "answer": "x",
            "sources": [],
            "invalid_citations": [],
            "grounding": {"citations_present": False, "sources_retrieved": 0},
        }

    monkeypatch.setattr(mcp_server, "ask_fn", fake_ask)

    mcp_server.ask("anything")
    mcp_server.ask("anything", corpus="reference")

    assert seen == [[cfg.collection], [cfg.reference_collection]]


def test_asking_across_both_corpora_retrieves_from_each(monkeypatch):
    """`all` fans out and fuses, rather than silently picking one."""
    import secondbrain.ask as ask_module

    queried: list[str | None] = []

    class FakeStore:
        def __init__(self, collection=None, **kw):
            self.collection = collection
            queried.append(collection)

        def count(self):
            return 3

        def query(self, qvec, k):
            return [
                {
                    "document": f"doc from {self.collection}",
                    "metadata": {"source": f"/{self.collection}/a.md", "chunk": 0},
                    "distance": 0.5 if self.collection == cfg.collection else 0.4,
                }
            ]

        def documents(self):
            return []

    monkeypatch.setattr(ask_module, "Store", FakeStore)
    monkeypatch.setattr(ask_module, "embed", lambda texts: [[0.1, 0.2]])

    result = ask_module.recall("anything", top_k=5, collection=collections_for("all"))

    assert queried == [cfg.collection, cfg.reference_collection]
    assert result["count"] == 2
    # Nearer hit first, whichever corpus it came from.
    assert result["hits"][0]["source"] == f"/{cfg.reference_collection}/a.md"
