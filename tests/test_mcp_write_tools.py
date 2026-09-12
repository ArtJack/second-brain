"""Direct tests for the two MCP tools that can change durable state.

`ingest` and `learn` are the highest-risk surface this server exposes: one adds
files to the brain, the other writes a memory and ingests it. Both were covered
only indirectly — `tests/test_mcp_server.py` exercises `ask`, `recall` and the
task tools, and `memory.learn` is touched once at the memory layer, which is not
the same thing as the MCP tool. So the accounting these tools report back to a
caller, and `ingest`'s refusal path, were unasserted.

Everything here fakes the layer below the tool. The point is the tool's own
contract — what it forwards, what it returns, and what it refuses — not the
ingestion pipeline, which has its own tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_ingest_reports_file_and_chunk_counts(monkeypatch, tmp_path: Path) -> None:
    """The counts a caller sees are summed across every file, not per file."""
    from secondbrain import mcp_server as m

    # The allow-list is real now; a caller ingesting from elsewhere
    # configures SB_INGEST_ROOTS, and so does this test.
    monkeypatch.setattr(m.cfg, "ingest_roots", [str(tmp_path)])

    target = tmp_path / "notes"
    target.mkdir()
    seen = {}

    def fake_ingest_paths(path, collection=None):
        seen["path"] = path
        yield ("a.md", 3)
        yield ("b.md", 4)

    class FakeStore:
        def count(self):
            return 99

    monkeypatch.setattr(m, "ingest_paths", fake_ingest_paths)
    monkeypatch.setattr(m, "Store", FakeStore)

    result = asyncio.run(m.ingest(str(target)))

    assert seen["path"] == str(target)
    assert result["files"] == 2
    assert result["chunks"] == 7, "chunks must be the sum across files, not the last file's count"
    assert result["total"] == 99
    assert result["path"] == str(target)


def test_ingest_expands_a_user_relative_path(monkeypatch, tmp_path: Path) -> None:
    """`~` has to resolve, or every caller passing a home-relative path fails."""
    from secondbrain import mcp_server as m

    # The allow-list is real now; a caller ingesting from elsewhere
    # configures SB_INGEST_ROOTS, and so does this test.
    monkeypatch.setattr(m.cfg, "ingest_roots", [str(tmp_path)])

    seen = {}
    monkeypatch.setattr(
        m, "ingest_paths", lambda path, collection=None: seen.update(path=path) or iter([("x", 1)])
    )
    monkeypatch.setattr(m, "Store", type("S", (), {"count": lambda self: 0}))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "note.md").write_text("hello")

    result = asyncio.run(m.ingest("~/note.md"))

    assert seen["path"] == str(tmp_path / "note.md")
    assert result["path"] == str(tmp_path / "note.md")


def test_ingest_refuses_a_missing_path_before_touching_the_store(monkeypatch, tmp_path: Path) -> None:
    """The refusal must happen first — a bad path should not reach ingestion."""
    from secondbrain import mcp_server as m

    # The allow-list is real now; a caller ingesting from elsewhere
    # configures SB_INGEST_ROOTS, and so does this test.
    monkeypatch.setattr(m.cfg, "ingest_roots", [str(tmp_path)])

    called = []
    monkeypatch.setattr(m, "ingest_paths", lambda *a, **k: called.append(1) or iter(()))
    monkeypatch.setattr(m, "Store", type("S", (), {"count": lambda self: 0}))

    missing = tmp_path / "does-not-exist.md"
    with pytest.raises(ValueError, match="Path not found"):
        asyncio.run(m.ingest(str(missing)))

    assert called == [], "ingestion ran despite the path not existing"


def test_learn_forwards_the_fact_and_returns_the_memory_path(monkeypatch) -> None:
    """`learn` is a thin pass-through; the contract is what it hands back."""
    from secondbrain import mcp_server as m

    seen = {}

    def fake_learn_memory(fact, source="user"):
        seen["fact"] = fact
        return {"path": "/tmp/memories/2026-09-06-a-fact.md", "chunks": 2}

    monkeypatch.setattr(m, "learn_memory", fake_learn_memory)

    result = asyncio.run(m.learn("The lab gateway is the only front door for models."))

    assert seen["fact"] == "The lab gateway is the only front door for models."
    assert result == {"memory_file": "/tmp/memories/2026-09-06-a-fact.md", "chunks": 2}


def test_learn_stringifies_a_path_object_from_the_memory_layer(monkeypatch) -> None:
    """`memory.learn` returns a real path; the tool must not leak a Path over MCP.

    MCP results are serialised to JSON, and a `Path` is not JSON-serialisable —
    so the `str()` in the tool is load-bearing, not cosmetic.
    """
    from secondbrain import mcp_server as m

    monkeypatch.setattr(
        m, "learn_memory", lambda fact, source="user": {"path": Path("/tmp/memories/x.md"), "chunks": 1}
    )

    result = asyncio.run(m.learn("anything"))

    assert isinstance(result["memory_file"], str)
    assert result["memory_file"] == "/tmp/memories/x.md"


def test_learn_does_not_swallow_a_failure_from_the_memory_layer(monkeypatch) -> None:
    """A failed write must surface, not return a success-shaped dict."""
    from secondbrain import mcp_server as m

    def boom(fact, source="user"):
        raise OSError("disk full")

    monkeypatch.setattr(m, "learn_memory", boom)

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(m.learn("a fact that cannot be written"))
