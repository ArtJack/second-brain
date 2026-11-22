"""Four ways learned memory and MCP writes were unbounded.

The brain's write surface is reachable over Tailscale by anything holding
`SB_MCP_TOKEN`, and it writes to a directory the nightly scan then ingests. That
combination turns three small gaps into real ones:

* a memory learned against a sandbox corpus landed in the same directory as the
  owner's real ones, and the nightly walked it into the real collection;
* `ingest` accepted any path on the host, so with `recall` a leaked token was an
  arbitrary-file-read primitive;
* a fact written by a tool call was recorded as `source: user`, indistinguishable
  from one the owner typed;
* and nothing could forget — deleting a memory file left its chunks retrievable
  forever, which is the wrong default for a system whose promise is that the
  owner controls what it knows.
"""
from __future__ import annotations

import asyncio

import pytest


class FakeStore:
    collection_name = "second_brain"

    def __init__(self):
        self.deleted: list[str] = []

    def delete_source(self, source):
        self.deleted.append(source)

    def upsert(self, **kw):
        pass

    def count(self):
        return 0


def test_a_sandbox_memory_does_not_land_beside_the_real_ones(tmp_path, monkeypatch):
    from secondbrain import memory

    monkeypatch.setattr(memory.cfg, "memory_dir", tmp_path)
    monkeypatch.setattr(memory, "ingest_paths", lambda *a, **kw: iter([]))

    default = memory.write_memory("a real fact")
    sandbox = memory.write_memory("a visitor's fact", collection="second_brain_sandbox_abc")

    assert default.parent == tmp_path, "the default collection keeps the existing location"
    assert sandbox.parent == tmp_path / "second_brain_sandbox_abc"
    assert sandbox.parent != default.parent


def test_provenance_is_recorded_in_the_memory_file(tmp_path, monkeypatch):
    from secondbrain import memory

    monkeypatch.setattr(memory.cfg, "memory_dir", tmp_path)
    path = memory.write_memory("learned by a tool", source="mcp")

    assert "source: mcp" in path.read_text()


class TestIngestAllowList:
    def test_a_path_outside_the_allowed_roots_is_refused(self, tmp_path, monkeypatch):
        from secondbrain import mcp_server as m

        outside = tmp_path / "secrets.md"
        outside.write_text("not yours to read")
        monkeypatch.setattr(m.cfg, "ingest_roots", [str(tmp_path / "allowed")])
        (tmp_path / "allowed").mkdir()

        called: list = []
        monkeypatch.setattr(m, "ingest_paths", lambda *a, **kw: called.append(a) or iter([]))

        with pytest.raises(ValueError) as excinfo:
            asyncio.run(m.ingest(str(outside)))

        assert called == [], "nothing may be read before the path is judged"
        assert "secrets.md" in str(excinfo.value)

    def test_a_symlink_pointing_outside_is_refused(self, tmp_path, monkeypatch):
        """Resolving before comparing is the whole check."""
        from secondbrain import mcp_server as m

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        secret = tmp_path / "secret.md"
        secret.write_text("not yours")
        link = allowed / "innocent.md"
        link.symlink_to(secret)
        monkeypatch.setattr(m.cfg, "ingest_roots", [str(allowed)])
        monkeypatch.setattr(m, "ingest_paths", lambda *a, **kw: iter([]))

        with pytest.raises(ValueError):
            asyncio.run(m.ingest(str(link)))

    def test_a_path_inside_the_roots_is_allowed(self, tmp_path, monkeypatch):
        from secondbrain import mcp_server as m

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        note = allowed / "note.md"
        note.write_text("mine")
        monkeypatch.setattr(m.cfg, "ingest_roots", [str(allowed)])
        monkeypatch.setattr(m, "ingest_paths", lambda *a, **kw: iter([(note, 1)]))
        monkeypatch.setattr(m, "Store", lambda *a, **kw: FakeStore())

        result = asyncio.run(m.ingest(str(note)))

        assert result["files"] == 1


def test_mcp_learn_records_that_a_tool_wrote_it(monkeypatch):
    from secondbrain import mcp_server as m

    seen = {}
    monkeypatch.setattr(
        m, "learn_memory", lambda fact, **kw: seen.update(kw) or {"path": "/m.md", "chunks": 1}
    )

    asyncio.run(m.learn("a fact from a tool call"))

    assert seen.get("source") == "mcp", "a tool-written fact must not look like one the owner typed"


class TestForget:
    def test_forget_removes_the_file_and_its_points(self, tmp_path, monkeypatch):
        from secondbrain import mcp_server as m

        monkeypatch.setattr(m.cfg, "memory_dir", tmp_path)
        note = tmp_path / "a-memory.md"
        note.write_text("something the owner no longer wants remembered")
        store = FakeStore()
        monkeypatch.setattr(m, "Store", lambda *a, **kw: store)

        result = asyncio.run(m.forget(str(note)))

        assert not note.exists()
        assert store.deleted == [str(note)]
        assert result["memory_file"] == str(note)

    def test_forget_refuses_a_path_outside_the_memory_directory(self, tmp_path, monkeypatch):
        from secondbrain import mcp_server as m

        monkeypatch.setattr(m.cfg, "memory_dir", tmp_path / "memories")
        (tmp_path / "memories").mkdir()
        elsewhere = tmp_path / "important.md"
        elsewhere.write_text("not a memory")
        store = FakeStore()
        monkeypatch.setattr(m, "Store", lambda *a, **kw: store)

        with pytest.raises(ValueError):
            asyncio.run(m.forget(str(elsewhere)))

        assert elsewhere.exists(), "forget must not be a general-purpose delete"
        assert store.deleted == []

    def test_forget_removes_the_memory_from_keyword_search_too(self, tmp_path, monkeypatch):
        """SB-F-52: a forgotten memory stayed findable by keyword.

        Every chunk lives in two copies, the vector store and the keyword index, and
        every other delete writes to both. `forget` wrote to the store alone, so the
        keyword search that `ask` fuses into its context, and that hybrid `recall`
        returns word for word, kept finding the memory the owner had just forgotten.
        """
        from secondbrain import hybrid
        from secondbrain import mcp_server as m
        from secondbrain.keyword_index import KeywordIndex

        memories = tmp_path / "memories"
        memories.mkdir()
        monkeypatch.setattr(m.cfg, "memory_dir", memories)
        forgotten = memories / "a-memory.md"
        forgotten.write_text("zephyrquill: a placeholder the owner asked to forget")
        kept = memories / "another-memory.md"
        kept.write_text("zephyrquill: a placeholder that stays")
        store = FakeStore()
        monkeypatch.setattr(m, "Store", lambda *a, **kw: store)
        index = KeywordIndex(tmp_path / "state" / "keyword-index.sqlite3")
        monkeypatch.setattr(m, "_index", index)
        monkeypatch.setattr(hybrid, "_index", index)
        index.upsert_chunks(
            store.collection_name,
            [{"source": str(p), "chunk": 0, "name": p.name, "document": p.read_text()} for p in (forgotten, kept)],
        )

        asyncio.run(m.forget(str(forgotten)))

        found = [hit["metadata"]["source"] for hit in hybrid.keyword_query(store, "zephyrquill", limit=5)]
        assert found == [str(kept)], "keyword search must lose the forgotten memory, and only that one"

    def test_forget_is_registered_as_a_write_tool(self):
        from secondbrain import mcp_server as m

        tools = {tool.name: tool for tool in asyncio.run(m.mcp.list_tools())}

        assert "forget" in tools
        assert tools["forget"].annotations.readOnlyHint is False


def test_the_nightly_scan_ignores_non_default_memory_subdirectories(tmp_path, monkeypatch):
    """Per-collection directories only help if the scan respects them.

    The overnight worker ingests the memory directory into the default
    collection. Without this, a sandbox memory filed under its own subdirectory
    would still be walked into the real brain the next night — the back door the
    subdirectories exist to close, left open one level down.
    """
    from secondbrain import overnight

    memory_root = tmp_path / "memory"
    (memory_root / "second_brain_sandbox_abc").mkdir(parents=True)
    (memory_root / "mine.md").write_text("the owner's own memory")
    (memory_root / "second_brain_sandbox_abc" / "visitor.md").write_text("a stranger's text")
    monkeypatch.setattr(overnight.cfg, "memory_dir", memory_root)

    result = overnight.scan_targets({"targets": [str(memory_root)], "max_file_mb": 1})

    assert [p.name for p in result.files] == ["mine.md"]
    assert any("sandbox" in item["path"] for item in result.skipped)
