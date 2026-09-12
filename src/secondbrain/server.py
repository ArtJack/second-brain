"""FastAPI serving layer for the second-brain web client."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .ask import ask as ask_fn
from .ask import recall as recall_fn
from .citations import grounding, invalid_citations
from .config import cfg
from .hybrid import hybrid_retrieve
from .ingest import ingest_paths
from .llm import answer_stream, embed
from .memory import learn as learn_memory
from .store import Store
from .tasks import TaskStore


class Corpus(str, Enum):
    public = "public"
    neutral = "neutral"
    sandbox = "sandbox"
    real = "real"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=25)
    corpus: Corpus = Corpus.public


class RecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=0, ge=0, le=25)
    corpus: Corpus = Corpus.public


class TextIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=100_000)
    source: str = Field(default="web-ingest.md", min_length=1, max_length=120)
    corpus: Corpus = Corpus.sandbox


class LearnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=20_000)
    source: str = Field(default="web", min_length=1, max_length=120)
    corpus: Corpus = Corpus.sandbox


class AddTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    notes: str = Field(default="", max_length=5000)


@dataclass(frozen=True)
class AuthContext:
    kind: Literal["anonymous", "anon_session", "owner", "service_read"]
    session_id: str | None = None

    @property
    def is_owner(self) -> bool:
        return self.kind == "owner"


class _SessionStore:
    """Anonymous session tokens, bounded in both count and lifetime.

    This was a plain dict: anyone who could reach POST /session minted an entry
    that stayed valid and resident for the life of the process. On a port exposed
    through a tunnel that is a memory leak with an authentication bonus. Both
    bounds are enforced lazily on access, so there is no background thread and no
    change to how a request is authenticated.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, float | str]] = {}

    def _max(self) -> int:
        try:
            return max(1, int(_env("SB_WEB_SESSION_MAX", "500")))
        except ValueError:
            return 500

    def _ttl(self) -> float:
        try:
            return float(_env("SB_WEB_SESSION_TTL_S", "86400"))
        except ValueError:
            return 86400.0

    def mint(self, token: str, session_id: str) -> None:
        self._entries[token] = {"session_id": session_id, "created_at": time.time()}
        # Oldest out first. dicts preserve insertion order, and a re-minted token
        # is re-inserted, so this is genuinely least-recently-created.
        while len(self._entries) > self._max():
            self._entries.pop(next(iter(self._entries)))

    def lookup(self, token: str) -> str | None:
        entry = self._entries.get(token)
        if entry is None:
            return None
        if time.time() - float(entry["created_at"]) > self._ttl():
            self._entries.pop(token, None)
            return None
        return str(entry["session_id"])

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


_SESSIONS = _SessionStore()

# The health probe's last verdict, and when it was taken. /health is anonymous
# and exempt from the limiter by design (the tunnel and the uptime monitor need a
# pulse), so without this every caller — and every 30-second poll from every open
# browser tab — bought an embedding on the lab GPU.
_HEALTH_TTL_S = 60.0
_health_cache: dict[str, object] = {"at": None, "result": None, "error": None}


def _reset_health_cache() -> None:
    _health_cache.update({"at": None, "result": None, "error": None})


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _enabled(name: str, default: str = "0") -> bool:
    return _env(name, default).lower() in {"1", "true", "yes", "on"}


def _allowed_origins() -> list[str]:
    raw = _env("SB_WEB_ALLOWED_ORIGINS")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError("SB_WEB_ALLOWED_ORIGINS must not contain '*'")
    return origins


app = FastAPI(title="second-brain web API")

_origins = _allowed_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["authorization", "content-type"],
    )


def _auth_from_header(header: str | None) -> AuthContext:
    if not header:
        return AuthContext("anonymous")
    prefix = "Bearer "
    if not header.startswith(prefix):
        raise HTTPException(status_code=401, detail="unauthorized")
    token = header[len(prefix) :]
    owner_token = _env("SB_WEB_OWNER_TOKEN")
    if owner_token and secrets.compare_digest(token, owner_token):
        return AuthContext("owner")
    read_token = _env("SB_WEB_READ_TOKEN")
    if read_token and secrets.compare_digest(token, read_token):
        return AuthContext("service_read")
    session_id = _SESSIONS.lookup(token)
    if session_id:
        return AuthContext("anon_session", session_id=session_id)
    raise HTTPException(status_code=401, detail="unauthorized")


# Paths an unauthenticated caller may reach even in locked-down mode: the
# tunnel and the uptime monitor need a pulse, and OPTIONS is CORS preflight.
_ANON_EXEMPT_PATHS = {"/health"}

# Origin-side rate limit. This process is single-instance, so an in-memory
# sliding window here is globally enforceable — unlike the Next proxy's
# per-instance advisory counter. Keyed on CF-Connecting-IP because public
# traffic only reaches this port through the Cloudflare tunnel; the header
# is spoofable only from inside the tailnet, which is our own machines.
_RATE_WINDOW_SECONDS = 60.0
_rate_hits: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(request: Request) -> bool:
    limit_raw = _env("SB_WEB_RATE_LIMIT_PER_MIN", "20")
    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 20
    if limit <= 0:  # 0 disables the limiter (tests, local dev)
        return False
    # The Next proxy reaches us from a handful of Vercel egress addresses;
    # keying on those would pool every visitor into one bucket. The proxy
    # declares the real visitor in X-Visitor-IP, and only the service token
    # earns trust in that declaration — an anonymous caller choosing its own
    # key would be choosing an empty bucket.
    ip = None
    auth = getattr(request.state, "auth", None)
    if auth is not None and auth.kind == "service_read":
        ip = request.headers.get("x-visitor-ip")
    ip = ip or request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    hits = _rate_hits[ip]
    while hits and now - hits[0] > _RATE_WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= limit:
        return True
    hits.append(now)
    # Bound the table. The previous sweep deleted entries whose deque was empty,
    # which inside a window is none of them — every key that had just been used
    # held at least one timestamp — so a scan across source addresses grew the
    # table without limit. Drop any key whose newest hit has fallen out of the
    # window instead: those cannot affect a decision again.
    if len(_rate_hits) > 10_000:
        for key in [k for k, v in _rate_hits.items() if not v or now - v[-1] > _RATE_WINDOW_SECONDS]:
            del _rate_hits[key]
    return False


@app.middleware("http")
async def bearer_auth_context(request: Request, call_next):
    try:
        request.state.auth = _auth_from_header(request.headers.get("authorization"))
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    # Auth first, then the limiter: the rate key depends on who is asking.
    if (
        request.method != "OPTIONS"
        and request.url.path not in _ANON_EXEMPT_PATHS
        and _rate_limited(request)
    ):
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limited"},
            headers={"retry-after": "60"},
        )
    # SB_WEB_REQUIRE_AUTH: refuse anonymous callers outright. The public
    # deployment exposes this port through a Cloudflare tunnel, and without
    # this gate anyone with curl could read the public corpora — and burn
    # GPU time — while skipping the site's Turnstile and rate limits. The
    # Next proxy authenticates with SB_WEB_READ_TOKEN, so real visitors are
    # never anonymous by the time they reach here. Blocking POST /session
    # for anons is deliberate: in this mode sessions are minted by nobody.
    if (
        _enabled("SB_WEB_REQUIRE_AUTH")
        and request.state.auth.kind == "anonymous"
        and request.method != "OPTIONS"
        and request.url.path not in _ANON_EXEMPT_PATHS
    ):
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


def auth_context(request: Request) -> AuthContext:
    return getattr(request.state, "auth", AuthContext("anonymous"))


def _collection_for(corpus: Corpus, auth: AuthContext, *, write: bool) -> str:
    if corpus == Corpus.public:
        if write:
            raise HTTPException(status_code=403, detail="public corpus is read-only")
        return _env("SB_WEB_PUBLIC_COLLECTION", "second_brain_public")
    if corpus == Corpus.neutral:
        if write:
            raise HTTPException(status_code=403, detail="neutral corpus is read-only")
        return _env("SB_WEB_NEUTRAL_COLLECTION", "second_brain_neutral")
    if corpus == Corpus.real:
        if not auth.is_owner:
            raise HTTPException(status_code=403, detail="owner token required")
        return cfg.collection
    if corpus == Corpus.sandbox:
        if not _enabled("SB_WEB_SANDBOX_ENABLED"):
            raise HTTPException(status_code=503, detail="sandbox disabled")
        if auth.is_owner:
            return "second_brain_sandbox_owner"
        if auth.kind != "anon_session":
            raise HTTPException(status_code=401, detail="anon session token required")
        return f"second_brain_sandbox_{auth.session_id}"
    raise HTTPException(status_code=400, detail="unknown corpus")


def _require_owner(auth: AuthContext) -> None:
    if not auth.is_owner:
        raise HTTPException(status_code=403, detail="owner token required")


@app.post("/session")
def create_session() -> dict:
    session_id = secrets.token_urlsafe(16)
    token = secrets.token_urlsafe(32)
    _SESSIONS.mint(token, session_id)
    return {"token": token, "session_id": session_id, "token_type": "bearer"}


@app.get("/health")
def health() -> dict:
    """Liveness, with the model probe cached for a minute.

    The store count is cheap and runs every time; the embedding call is not, and
    this endpoint is anonymous and exempt from the rate limiter, so a burst — or
    one browser tab polling every 30 seconds — used to mean a burst of GPU work.
    The cached verdict includes failures, so an outage is still reported, just not
    re-measured on every request.
    """
    collection = _env("SB_WEB_PUBLIC_COLLECTION", "second_brain_public")
    now = time.monotonic()
    at = _health_cache["at"]
    fresh = at is not None and (now - float(at)) < _HEALTH_TTL_S
    if not fresh:
        try:
            embed(["health"])
            _health_cache.update({"at": now, "result": True, "error": None})
        except Exception as exc:
            _health_cache.update({"at": now, "result": False, "error": str(exc)})
    if not _health_cache["result"]:
        raise HTTPException(status_code=503, detail=f"unavailable: {_health_cache['error']}")
    try:
        chunks = Store(collection=collection).count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"unavailable: {exc}") from exc
    return {"ok": True, "collection": collection, "chunks": chunks}


@app.get("/status")
def status(
    corpus: Corpus = Query(default=Corpus.public),
    auth: AuthContext = Depends(auth_context),
) -> dict:
    collection = _collection_for(corpus, auth, write=False)
    try:
        chunks: int | str = Store(collection=collection).count()
    except Exception as exc:
        chunks = f"unavailable: {exc}"
    return {
        "corpus": corpus.value,
        "collection": collection,
        "store": cfg.store_backend,
        "backend": cfg.base_url,
        "embed_model": cfg.embed_model,
        "chat_model": cfg.chat_model,
        "chunks": chunks,
    }


@app.post("/ask")
def ask(req: AskRequest, auth: AuthContext = Depends(auth_context)) -> dict:
    collection = _collection_for(req.corpus, auth, write=False)
    return ask_fn(req.question, k=req.k, collection=collection)


@app.post("/recall")
def recall(req: RecallRequest, auth: AuthContext = Depends(auth_context)) -> dict:
    collection = _collection_for(req.corpus, auth, write=False)
    return recall_fn(req.query, top_k=req.top_k, collection=collection)


def _context_for_stream(question: str, k: int | None, collection: str) -> dict:
    store = Store(collection=collection)
    if store.count() == 0:
        return {"empty": True, "context": "", "sources": []}
    limit = k or cfg.top_k
    qvec = embed([question])[0]
    hits = hybrid_retrieve(store, question, qvec, limit, enabled=cfg.hybrid_enabled)
    context_parts, sources = [], []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        context_parts.append(f"[{i}] (from {meta.get('name', meta.get('source'))})\n{hit['document']}")
        sources.append(
            {
                "n": i,
                "source": meta.get("source", "?"),
                "distance": hit["distance"],
                "retrieval": hit.get("retrieval", "vector"),
                # The chunk text rides along so the client's citation panel
                # can show what [n] actually cites. Without it the stream's
                # sources overwrite the recall-populated ones and the panel
                # degrades to "source text unavailable".
                "text": hit["document"],
            }
        )
    return {"empty": False, "context": "\n\n".join(context_parts), "sources": sources}


def _sse(data: object, *, event: str | None = None) -> str:
    text = data if isinstance(data, str) else json.dumps(data)
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    payload_lines = text.splitlines() or [""]
    lines.extend(f"data: {line}" for line in payload_lines)
    return "\n".join(lines) + "\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest, auth: AuthContext = Depends(auth_context)) -> StreamingResponse:
    collection = _collection_for(req.corpus, auth, write=False)

    def events():
        evidence = _context_for_stream(req.question, req.k, collection)
        sources = evidence["sources"]
        yield _sse(sources, event="sources")
        if evidence["empty"]:
            answer_text = "Nothing ingested yet - run `sb ingest <path>` first."
            yield _sse(answer_text, event="token")
            yield _sse(
                {"invalid_citations": [], "grounding": grounding("", 0)},
                event="done",
            )
            return
        parts: list[str] = []
        for delta in answer_stream(req.question, evidence["context"]):
            parts.append(delta)
            yield _sse(delta, event="token")
        answer_text = "".join(parts)
        # The streaming path is what the web UI reaches for interactive answers,
        # so it needs the same grounding signal as POST /ask. `invalid_citations`
        # alone is structurally blind to an answer that cites nothing at all.
        yield _sse(
            {
                "invalid_citations": invalid_citations(answer_text, len(sources)),
                "grounding": grounding(answer_text, len(sources)),
            },
            event="done",
        )

    return StreamingResponse(events(), media_type="text/event-stream")


def _temp_source_name(source: str) -> str:
    name = Path(source).name.strip() or "web-ingest.md"
    if "." not in name:
        name += ".md"
    return name


@app.post("/ingest")
def ingest(req: TextIngestRequest, auth: AuthContext = Depends(auth_context)) -> dict:
    collection = _collection_for(req.corpus, auth, write=True)
    files = chunks = 0
    with tempfile.TemporaryDirectory(prefix="secondbrain-web-") as tmp:
        path = Path(tmp) / _temp_source_name(req.source)
        path.write_text(req.text, encoding="utf-8")
        for _file, n in ingest_paths(path, collection=collection):
            files += 1
            chunks += n
    total = Store(collection=collection).count()
    return {"files": files, "chunks": chunks, "total": total, "collection": collection}


@app.post("/learn")
def learn(req: LearnRequest, auth: AuthContext = Depends(auth_context)) -> dict:
    collection = _collection_for(req.corpus, auth, write=True)
    result = learn_memory(req.text, source=req.source, collection=collection)
    return {"memory_file": str(result["path"]), "chunks": result["chunks"], "collection": collection}


@app.get("/tasks")
def list_tasks(
    status: str = Query(default="open", pattern="^(open|done|all)$"),
    auth: AuthContext = Depends(auth_context),
) -> dict:
    _require_owner(auth)
    tasks = TaskStore().list(status=status)
    return {"count": len(tasks), "tasks": tasks}


@app.post("/tasks")
def add_task(req: AddTaskRequest, auth: AuthContext = Depends(auth_context)) -> dict:
    _require_owner(auth)
    return TaskStore().add(req.title, req.notes)


@app.post("/tasks/{task_id}/complete")
def complete_task(task_id: int, auth: AuthContext = Depends(auth_context)) -> dict:
    _require_owner(auth)
    try:
        task = TaskStore().complete(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    return {"task_id": task_id, "completed": True, "status": task["status"]}


def main() -> None:
    import uvicorn

    host = _env("SB_WEB_HOST", "127.0.0.1")
    port = int(_env("SB_WEB_PORT", "8850"))
    uvicorn.run("secondbrain.server:app", host=host, port=port, log_level="info")
