"""Offline unit tests for the MCP server layer.

These fake the engine dependencies (so no gateway / Qdrant / disk is touched) and
assert that each tool calls the engine correctly and shapes its output. The live
end-to-end behaviour of `ask`/`recall`/`ingest`/`learn` is covered by the engine's
own tests and by the MCP Inspector.
"""
from __future__ import annotations

import pytest


def test_ask_shapes_citations(monkeypatch):
    from secondbrain import mcp_server as m

    def fake_ask(question, k=None):
        assert question == "q"
        assert k is None  # top_k=0 -> None (use configured default)
        return {
            "answer": "A",
            "sources": [{"n": 1, "source": "/x.md", "distance": 0.1234}],
            "invalid_citations": [],
        }

    monkeypatch.setattr(m, "ask_fn", fake_ask)
    out = m.ask("q")
    assert out["answer"] == "A"
    assert out["citations"] == [{"n": 1, "source": "/x.md", "distance": 0.1234}]
    assert out["ungrounded_citations"] == []


def test_ask_passes_top_k(monkeypatch):
    from secondbrain import mcp_server as m

    seen = {}
    monkeypatch.setattr(m, "ask_fn", lambda q, k=None: seen.update(k=k) or {"answer": "", "sources": []})
    m.ask("q", top_k=7)
    assert seen["k"] == 7


def test_list_tasks_passes_status(monkeypatch):
    from secondbrain import mcp_server as m

    captured = {}

    class FakeTS:
        def __init__(self, *a, **k):
            pass

        def list(self, status="open", limit=50):
            captured["status"] = status
            return [{"id": 1, "title": "t", "status": status, "created_at": "now", "notes": "", "completed_at": None}]

    monkeypatch.setattr(m, "TaskStore", FakeTS)
    out = m.list_tasks(status="all")
    assert captured["status"] == "all"
    assert out["count"] == 1
    assert out["tasks"][0] == {"id": 1, "title": "t", "status": "all", "created_at": "now"}


def test_add_task_shape(monkeypatch):
    from secondbrain import mcp_server as m

    class FakeTS:
        def __init__(self, *a, **k):
            pass

        def add(self, title, notes=""):
            return {"id": 5, "title": title, "status": "open", "notes": notes, "created_at": "now", "completed_at": None}

    monkeypatch.setattr(m, "TaskStore", FakeTS)
    out = m.add_task("write docs")
    assert out == {"id": 5, "title": "write docs", "status": "open", "created_at": "now"}


def test_complete_missing_task_is_false(monkeypatch):
    from secondbrain import mcp_server as m

    class FakeTS:
        def __init__(self, *a, **k):
            pass

        def complete(self, task_id):
            raise KeyError("nope")

    monkeypatch.setattr(m, "TaskStore", FakeTS)
    out = m.complete_task(999)
    assert out["completed"] is False
    assert out["task_id"] == 999


def test_status_shape_and_no_secrets(monkeypatch):
    from secondbrain import mcp_server as m

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def count(self):
            return 7

    monkeypatch.setattr(m, "Store", FakeStore)
    s = m.status()
    for key in ("store", "backend", "embed_model", "chat_model", "memory_dir", "state_db", "chunks"):
        assert key in s, f"status() missing {key}"
    assert s["chunks"] == 7
    assert "api_key" not in s  # never expose the key
    assert "qdrant_api_key" not in s


def test_recall_empty_store_short_circuits(monkeypatch):
    from secondbrain import mcp_server as m

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def count(self):
            return 0

    monkeypatch.setattr(m, "Store", FakeStore)
    # embed must NOT be called when the store is empty
    monkeypatch.setattr(m, "embed", lambda *a, **k: pytest.fail("embed should not run on empty store"))
    out = m.recall("anything")
    assert out == {"count": 0, "hits": []}


def test_expected_tools_are_registered():
    from secondbrain import mcp_server as m

    for name in ("ask", "recall", "ingest", "learn", "list_tasks", "add_task", "complete_task", "status"):
        assert callable(getattr(m, name)), f"missing tool: {name}"
    assert callable(m.main)
