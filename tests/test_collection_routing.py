from __future__ import annotations

from typer.testing import CliRunner


def test_ask_routes_to_named_collection(monkeypatch):
    import secondbrain.ask as ask_module

    captured = {}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            captured["collection"] = kwargs.get("collection")

        def count(self):
            return 0

    monkeypatch.setattr(ask_module, "Store", FakeStore)

    out = ask_module.ask("what is stored?", collection="second_brain_public")

    assert captured["collection"] == "second_brain_public"
    assert out["sources"] == []


def test_recall_routes_to_named_collection(monkeypatch):
    import secondbrain.ask as ask_module

    captured = {}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            captured["collection"] = kwargs.get("collection")

        def count(self):
            return 1

        def query(self, qvec, limit):
            captured["qvec"] = qvec
            captured["limit"] = limit
            return [
                {
                    "document": "chunk text",
                    "metadata": {"source": "source.md"},
                    "distance": 0.12345,
                }
            ]

    monkeypatch.setattr(ask_module, "Store", FakeStore)
    monkeypatch.setattr(ask_module, "embed", lambda texts: [[0.5, 0.6]])

    out = ask_module.recall("query", top_k=3, collection="second_brain_neutral")

    assert captured == {"collection": "second_brain_neutral", "qvec": [0.5, 0.6], "limit": 3}
    assert out == {
        "count": 1,
        "hits": [{"source": "source.md", "distance": 0.1235, "text": "chunk text"}],
    }


def test_ingest_paths_routes_to_named_collection(tmp_path, monkeypatch):
    import secondbrain.ingest as ingest

    captured = {}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            captured["collection"] = kwargs.get("collection")

        def delete_source(self, source):
            captured["deleted"] = source

        def upsert(self, ids, embeddings, documents, metadatas):
            captured["upserted"] = len(ids)

    monkeypatch.setattr(ingest, "Store", FakeStore)
    monkeypatch.setattr(ingest, "embed", lambda texts: [[0.0] for _ in texts])

    note = tmp_path / "note.md"
    note.write_text("hello world\n" * 40)

    out = list(ingest.ingest_paths(note, collection="second_brain_public"))

    assert captured["collection"] == "second_brain_public"
    assert captured["deleted"] == str(note)
    assert captured["upserted"] == 1
    assert out == [(note, 1)]


def test_learn_routes_to_named_collection(tmp_path, monkeypatch):
    import secondbrain.memory as memory

    captured = {}
    path = tmp_path / "memory.md"

    monkeypatch.setattr(memory, "write_memory", lambda text, source="user": path)

    def fake_ingest_paths(ingest_path, collection=None):
        captured["path"] = ingest_path
        captured["collection"] = collection
        yield ingest_path, 2

    monkeypatch.setattr(memory, "ingest_paths", fake_ingest_paths)

    out = memory.learn("remember this", collection="second_brain_public")

    assert captured == {"path": path, "collection": "second_brain_public"}
    assert out == {"path": path, "chunks": 2}


def test_cli_global_collection_flows_to_recall(monkeypatch):
    import secondbrain.cli as cli

    captured = {}
    monkeypatch.setattr(
        cli,
        "recall_fn",
        lambda query, top_k=0, collection=None: captured.update(
            query=query, top_k=top_k, collection=collection
        )
        or {"count": 0, "hits": []},
    )

    result = CliRunner().invoke(cli.app, ["--collection", "second_brain_public", "recall", "hybrid", "--k", "4"])

    assert result.exit_code == 0
    assert captured == {"query": "hybrid", "top_k": 4, "collection": "second_brain_public"}
