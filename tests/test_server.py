from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from secondbrain import server


@pytest.fixture(autouse=True)
def _hermetic_web_env(monkeypatch):
    # config.py runs load_dotenv() at import, so whatever SB_WEB_* the
    # developer's real .env carries (the deployed demo sets REQUIRE_AUTH=1
    # and a concierge collection) would leak into every test. Strip the lot
    # back to defaults; tests that need a mode set it explicitly.
    for name in (
        "SB_WEB_REQUIRE_AUTH",
        "SB_WEB_READ_TOKEN",
        "SB_WEB_OWNER_TOKEN",
        "SB_WEB_PUBLIC_COLLECTION",
        "SB_WEB_NEUTRAL_COLLECTION",
        "SB_WEB_SANDBOX_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    # Every TestClient request shares one client address, so the sliding
    # window would trip across unrelated tests. 0 disables the limiter;
    # the rate-limit tests set their own budget explicitly.
    monkeypatch.setenv("SB_WEB_RATE_LIMIT_PER_MIN", "0")


def _client():
    from secondbrain import server

    server._SESSIONS.clear()
    server._rate_hits.clear()
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


def test_read_token_can_read_public_corpus(monkeypatch):
    client, _server = _client()
    captured = {}
    monkeypatch.setenv("SB_WEB_READ_TOKEN", "read-secret")
    monkeypatch.setattr(
        _server,
        "ask_fn",
        lambda question, k=None, collection=None: captured.update(
            question=question,
            k=k,
            collection=collection,
        )
        or {"answer": "A [1].", "sources": [], "invalid_citations": []},
    )

    resp = client.post(
        "/ask",
        json={"question": "public?", "k": 2, "corpus": "public"},
        headers={"Authorization": "Bearer read-secret"},
    )

    assert resp.status_code == 200
    assert captured == {"question": "public?", "k": 2, "collection": "second_brain_public"}


def test_read_token_cannot_access_real_corpus(monkeypatch):
    client, _server = _client()
    monkeypatch.setenv("SB_WEB_READ_TOKEN", "read-secret")

    resp = client.post(
        "/ask",
        json={"question": "private?", "corpus": "real"},
        headers={"Authorization": "Bearer read-secret"},
    )

    assert resp.status_code == 403


def test_read_token_cannot_access_tasks(monkeypatch):
    client, _server = _client()
    monkeypatch.setenv("SB_WEB_READ_TOKEN", "read-secret")

    resp = client.get("/tasks", headers={"Authorization": "Bearer read-secret"})

    assert resp.status_code == 403


def test_garbage_bearer_token_is_unauthorized(monkeypatch):
    client, _server = _client()
    monkeypatch.setenv("SB_WEB_READ_TOKEN", "read-secret")

    resp = client.post(
        "/ask",
        json={"question": "public?", "corpus": "public"},
        headers={"Authorization": "Bearer nope"},
    )

    assert resp.status_code == 401


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


def test_require_auth_blocks_anonymous_everywhere_but_health(monkeypatch):
    client, server = _client()
    monkeypatch.setenv("SB_WEB_REQUIRE_AUTH", "1")
    monkeypatch.delenv("SB_WEB_OWNER_TOKEN", raising=False)

    # /health's own dependencies are faked because the subject of this test is
    # the auth exemption, not the store or the model. Unfaked, the probe makes a
    # real embedding call; the suite points that at a closed port, so /health
    # answers 503 and the last assertion below fails for a reason that has
    # nothing to do with auth. `test_health_checks_store_and_embedding` is where
    # the probe's own behaviour is covered.
    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def count(self):
            return 0

    monkeypatch.setattr(server, "Store", FakeStore)
    monkeypatch.setattr(server, "embed", lambda texts: [[0.0]])

    # Anonymous reads are refused outright — the tunnel makes this port
    # world-reachable, so "anonymous" must mean "turned away", not "public".
    assert client.post("/ask", json={"corpus": "public", "question": "hi"}).status_code == 401
    assert client.post("/recall", json={"corpus": "public", "query": "hi"}).status_code == 401
    assert client.get("/status", params={"corpus": "public"}).status_code == 401
    # Session minting is anonymous by definition; in this mode nobody mints.
    assert client.post("/session").status_code == 401
    # The pulse stays reachable for the tunnel and the uptime monitor.
    assert client.get("/health").status_code == 200


def test_require_auth_admits_service_read_token(monkeypatch):
    client, server = _client()
    monkeypatch.setenv("SB_WEB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("SB_WEB_READ_TOKEN", "svc-token")

    class FakeStore:
        def __init__(self, collection=None):
            self.collection = collection

        def count(self):
            return 3

    monkeypatch.setattr(server, "Store", FakeStore)
    response = client.get(
        "/status",
        params={"corpus": "public"},
        headers={"Authorization": "Bearer svc-token"},
    )
    assert response.status_code == 200


def test_rate_limit_blocks_after_budget_and_exempts_health(monkeypatch):
    client, server = _client()
    monkeypatch.setenv("SB_WEB_RATE_LIMIT_PER_MIN", "3")

    class FakeStore:
        def __init__(self, collection=None):
            self.collection = collection

        def count(self):
            return 3

    monkeypatch.setattr(server, "Store", FakeStore)
    monkeypatch.setattr(server, "embed", lambda texts: [[0.0]])

    for _ in range(3):
        assert client.get("/status", params={"corpus": "public"}).status_code == 200
    over = client.get("/status", params={"corpus": "public"})
    assert over.status_code == 429
    assert over.headers["retry-after"] == "60"
    # The pulse must survive a limited caller: the uptime monitor and the
    # tunnel keep probing /health no matter what an abuser is doing.
    assert client.get("/health").status_code == 200


def test_rate_limit_keys_on_cf_connecting_ip(monkeypatch):
    client, server = _client()
    monkeypatch.setenv("SB_WEB_RATE_LIMIT_PER_MIN", "1")

    class FakeStore:
        def __init__(self, collection=None):
            self.collection = collection

        def count(self):
            return 3

    monkeypatch.setattr(server, "Store", FakeStore)

    first = client.get(
        "/status", params={"corpus": "public"}, headers={"CF-Connecting-IP": "203.0.113.7"}
    )
    blocked = client.get(
        "/status", params={"corpus": "public"}, headers={"CF-Connecting-IP": "203.0.113.7"}
    )
    other = client.get(
        "/status", params={"corpus": "public"}, headers={"CF-Connecting-IP": "203.0.113.8"}
    )
    assert first.status_code == 200
    assert blocked.status_code == 429
    assert other.status_code == 200


def test_rate_limit_trusts_visitor_ip_only_from_service_token(monkeypatch):
    client, server = _client()
    monkeypatch.setenv("SB_WEB_RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv("SB_WEB_READ_TOKEN", "svc-token")

    class FakeStore:
        def __init__(self, collection=None):
            self.collection = collection

        def count(self):
            return 3

    monkeypatch.setattr(server, "Store", FakeStore)
    svc = {"Authorization": "Bearer svc-token"}

    # The proxy declares distinct visitors: each gets their own bucket.
    a1 = client.get(
        "/status", params={"corpus": "public"}, headers={**svc, "X-Visitor-IP": "198.51.100.1"}
    )
    a2 = client.get(
        "/status", params={"corpus": "public"}, headers={**svc, "X-Visitor-IP": "198.51.100.1"}
    )
    b1 = client.get(
        "/status", params={"corpus": "public"}, headers={**svc, "X-Visitor-IP": "198.51.100.2"}
    )
    assert a1.status_code == 200
    assert a2.status_code == 429
    assert b1.status_code == 200

    # An anonymous caller does not get to pick its own bucket.
    server._rate_hits.clear()
    first = client.get(
        "/status", params={"corpus": "public"}, headers={"X-Visitor-IP": "198.51.100.3"}
    )
    spoofed = client.get(
        "/status", params={"corpus": "public"}, headers={"X-Visitor-IP": "198.51.100.4"}
    )
    assert first.status_code == 200
    assert spoofed.status_code == 429


def test_require_auth_off_keeps_anonymous_reads_working(monkeypatch):
    client, server = _client()
    monkeypatch.delenv("SB_WEB_REQUIRE_AUTH", raising=False)

    class FakeStore:
        def __init__(self, collection=None):
            self.collection = collection

        def count(self):
            return 3

    monkeypatch.setattr(server, "Store", FakeStore)
    assert client.get("/status", params={"corpus": "public"}).status_code == 200


# --- Task 11: session lifetime, health probe cost, and a limiter that really evicts ---


class TestSessionLifetime:
    """Anonymous session tokens are minted by anyone who can reach /session.

    The store was a plain module-level dict with no cap and no expiry, so every
    token ever issued stayed valid and resident for the life of the process. On a
    port exposed through a tunnel that is a memory leak with an authentication
    bonus.
    """

    def test_minting_past_the_cap_evicts_the_oldest(self, monkeypatch):
        monkeypatch.setenv("SB_WEB_SESSION_MAX", "3")
        client = TestClient(server.app)
        server._SESSIONS.clear()

        tokens = [client.post("/session").json()["token"] for _ in range(4)]

        assert server._SESSIONS.lookup(tokens[0]) is None, "the oldest token should be gone"
        assert server._SESSIONS.lookup(tokens[-1]) is not None

    def test_an_expired_token_is_not_a_session(self, monkeypatch):
        monkeypatch.setenv("SB_WEB_SESSION_TTL_S", "60")
        client = TestClient(server.app)
        server._SESSIONS.clear()
        token = client.post("/session").json()["token"]

        assert server._SESSIONS.lookup(token) is not None

        now = time.time()
        monkeypatch.setattr(server.time, "time", lambda: now + 61)

        assert server._SESSIONS.lookup(token) is None

    def test_expiry_is_enforced_at_the_auth_boundary(self, monkeypatch):
        """A dead token must not authenticate, not merely fail a lookup."""
        monkeypatch.setenv("SB_WEB_SESSION_TTL_S", "60")
        server._SESSIONS.clear()
        client = TestClient(server.app)
        token = client.post("/session").json()["token"]

        now = time.time()
        monkeypatch.setattr(server.time, "time", lambda: now + 61)

        with pytest.raises(HTTPException) as excinfo:
            server._auth_from_header(f"Bearer {token}")
        assert excinfo.value.status_code == 401


class TestHealthProbeCost:
    def test_a_burst_of_health_checks_costs_one_embedding(self, monkeypatch):
        """/health is anonymous and exempt from the limiter: it must not be a GPU tap."""
        calls = {"n": 0}

        def counted(texts):
            calls["n"] += 1
            return [[0.1, 0.2]]

        monkeypatch.setattr(server, "embed", counted)
        monkeypatch.setattr(server, "Store", lambda **kw: type("S", (), {"count": lambda self: 3})())
        server._reset_health_cache()
        client = TestClient(server.app)

        first = client.get("/health")
        second = client.get("/health")

        assert first.status_code == 200
        assert second.json() == first.json()
        assert calls["n"] == 1, f"{calls['n']} embeddings for two health checks"

    def test_the_cache_expires_so_an_outage_is_still_noticed(self, monkeypatch):
        calls = {"n": 0}

        def counted(texts):
            calls["n"] += 1
            return [[0.1, 0.2]]

        monkeypatch.setattr(server, "embed", counted)
        monkeypatch.setattr(server, "Store", lambda **kw: type("S", (), {"count": lambda self: 3})())
        server._reset_health_cache()
        client = TestClient(server.app)
        client.get("/health")

        now = time.monotonic()
        monkeypatch.setattr(server.time, "monotonic", lambda: now + 120)
        client.get("/health")

        assert calls["n"] == 2


class TestRateLimitEviction:
    def test_idle_keys_are_dropped_not_just_empty_ones(self, monkeypatch):
        """The old sweep deleted entries whose deque was empty — which, inside a
        window, is none of them. A scan across source addresses grew the table
        without bound."""
        monkeypatch.setenv("SB_WEB_RATE_LIMIT_PER_MIN", "20")
        server._rate_hits.clear()

        now = [1000.0]
        monkeypatch.setattr(server.time, "monotonic", lambda: now[0])

        for i in range(12_000):
            request = SimpleNamespace(
                headers={"cf-connecting-ip": f"10.0.{i // 256}.{i % 256}"},
                state=SimpleNamespace(auth=None),
                client=None,
            )
            server._rate_limited(request)

        assert len(server._rate_hits) == 12_000

        now[0] += 3600  # everything is now far outside the window
        request = SimpleNamespace(
            headers={"cf-connecting-ip": "10.9.9.9"},
            state=SimpleNamespace(auth=None),
            client=None,
        )
        server._rate_limited(request)

        assert len(server._rate_hits) < 100, f"{len(server._rate_hits)} stale keys retained"
