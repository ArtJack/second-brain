from __future__ import annotations

from fastapi.testclient import TestClient


def _client():
    import secondbrain.server as server

    server._SESSIONS.clear()
    return TestClient(server.app), server


def test_public_read_endpoints_route_to_public_collection(monkeypatch):
    client, server = _client()
    captured = {}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            captured.setdefault("store_collections", []).append(kwargs.get("collection"))

        def count(self):
            return 7

    def fake_ask(question, k=None, collection=None):
        captured["ask"] = {"question": question, "k": k, "collection": collection}
        return {"answer": "A [1].", "sources": [], "invalid_citations": []}

    def fake_recall(query, top_k=0, collection=None):
        captured["recall"] = {"query": query, "top_k": top_k, "collection": collection}
        return {"count": 0, "hits": []}

    monkeypatch.setattr(server, "Store", FakeStore)
    monkeypatch.setattr(server, "ask_fn", fake_ask)
    monkeypatch.setattr(server, "recall_fn", fake_recall)

    status = client.get("/status", params={"corpus": "public"})
    ask = client.post("/ask", json={"question": "How?", "k": 2, "corpus": "public"})
    recall = client.post("/recall", json={"query": "hybrid", "top_k": 3, "corpus": "public"})

    assert status.status_code == 200
    assert status.json()["collection"] == "second_brain_public"
    assert ask.status_code == 200
    assert recall.status_code == 200
    assert captured["store_collections"] == ["second_brain_public"]
    assert captured["ask"] == {"question": "How?", "k": 2, "collection": "second_brain_public"}
    assert captured["recall"] == {"query": "hybrid", "top_k": 3, "collection": "second_brain_public"}


def test_writes_to_public_and_neutral_are_forbidden(monkeypatch):
    client, server = _client()
    monkeypatch.setattr(server, "learn_memory", lambda *args, **kwargs: {"path": "", "chunks": 0})

    public = client.post("/learn", json={"text": "remember this", "corpus": "public"})
    neutral = client.post("/ingest", json={"text": "doc", "corpus": "neutral"})

    assert public.status_code == 403
    assert neutral.status_code == 403


def test_real_corpus_requires_owner_token(monkeypatch):
    client, server = _client()
    monkeypatch.delenv("SB_WEB_OWNER_TOKEN", raising=False)
    monkeypatch.setattr(server, "ask_fn", lambda *args, **kwargs: {"answer": "", "sources": []})

    resp = client.post("/ask", json={"question": "private?", "corpus": "real"})

    assert resp.status_code == 403


def test_owner_token_can_read_real_corpus(monkeypatch):
    client, server = _client()
    captured = {}
    monkeypatch.setenv("SB_WEB_OWNER_TOKEN", "owner-secret")
    monkeypatch.setattr(
        server,
        "ask_fn",
        lambda question, k=None, collection=None: captured.update(collection=collection)
        or {"answer": "A", "sources": [], "invalid_citations": []},
    )

    resp = client.post(
        "/ask",
        json={"question": "private?", "corpus": "real"},
        headers={"Authorization": "Bearer owner-secret"},
    )

    assert resp.status_code == 200
    assert captured["collection"] == server.cfg.collection


def test_sandbox_routes_are_disabled_by_default(monkeypatch):
    client, _server = _client()
    monkeypatch.delenv("SB_WEB_SANDBOX_ENABLED", raising=False)

    read = client.post("/recall", json={"query": "q", "corpus": "sandbox"})
    write = client.post("/ingest", json={"text": "doc", "corpus": "sandbox"})

    assert read.status_code == 503
    assert write.status_code == 503


def test_sse_stream_yields_deltas_and_routes_collection(monkeypatch):
    client, server = _client()
    captured = {}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            captured["collection"] = kwargs.get("collection")

        def count(self):
            return 1

    monkeypatch.setattr(server, "Store", FakeStore)
    monkeypatch.setattr(server, "embed", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(
        server,
        "hybrid_retrieve",
        lambda store, question, qvec, limit, enabled=True: [
            {
                "document": "Hybrid retrieval combines vector and keyword hits.",
                "metadata": {"source": "docs/design.md", "name": "design.md"},
                "distance": 0.22,
            }
        ],
    )
    monkeypatch.setattr(server, "answer_stream", lambda question, context: iter(["hello ", "world"]))

    resp = client.post("/ask/stream", json={"question": "How?", "k": 2, "corpus": "neutral"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert captured["collection"] == "second_brain_neutral"
    assert "event: sources" in resp.text
    assert "event: token\ndata: hello " in resp.text
    assert "event: token\ndata: world" in resp.text
    assert "event: done" in resp.text


def test_health_checks_store_and_embedding(monkeypatch):
    client, server = _client()
    captured = {}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            captured["collection"] = kwargs.get("collection")

        def count(self):
            return 4

    monkeypatch.setattr(server, "Store", FakeStore)
    monkeypatch.setattr(server, "embed", lambda texts: captured.update(embed_texts=texts) or [[0.0]])

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["chunks"] == 4
    assert captured == {"collection": "second_brain_public", "embed_texts": ["health"]}
