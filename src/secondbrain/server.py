"""FastAPI serving layer for the second-brain web client."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
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
from .citations import invalid_citations
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
    kind: Literal["anonymous", "anon_session", "owner"]
    session_id: str | None = None

    @property
    def is_owner(self) -> bool:
        return self.kind == "owner"


_SESSIONS: dict[str, str] = {}


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
    session_id = _SESSIONS.get(token)
    if session_id:
        return AuthContext("anon_session", session_id=session_id)
    raise HTTPException(status_code=401, detail="unauthorized")


@app.middleware("http")
async def bearer_auth_context(request: Request, call_next):
    try:
        request.state.auth = _auth_from_header(request.headers.get("authorization"))
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
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
    _SESSIONS[token] = session_id
    return {"token": token, "session_id": session_id, "token_type": "bearer"}


@app.get("/health")
def health() -> dict:
    collection = _env("SB_WEB_PUBLIC_COLLECTION", "second_brain_public")
    try:
        chunks = Store(collection=collection).count()
        embed(["health"])
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
            yield _sse({"invalid_citations": []}, event="done")
            return
        parts: list[str] = []
        for delta in answer_stream(req.question, evidence["context"]):
            parts.append(delta)
            yield _sse(delta, event="token")
        yield _sse(
            {"invalid_citations": invalid_citations("".join(parts), len(sources))},
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
