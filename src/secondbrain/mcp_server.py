"""MCP server for second-brain.

Exposes the second-brain RAG engine as Model Context Protocol tools so any MCP
client (Claude Code, Claude Desktop, a native iPad/MacBook client) can query and
teach your brain.

It reuses the existing engine — same `cfg`, same LiteLLM-gateway routing, same
Qdrant/Chroma store — so it inherits the lab's free-local-model defaults at zero
marginal cost. No model is called except through that engine.

Two transports, selected by env:

    SB_MCP_TRANSPORT=stdio   (default)  local clients launch it as a subprocess
    SB_MCP_TRANSPORT=http               Streamable HTTP for networked clients

HTTP env:
    SB_MCP_HOST   bind address (default 127.0.0.1; use the Tailscale IP to share)
    SB_MCP_PORT   port (default 8848)
    SB_MCP_TOKEN  if set, clients must send `Authorization: Bearer <token>`

Run:
    uv run sb-mcp                                  # stdio
    SB_MCP_TRANSPORT=http SB_MCP_TOKEN=… uv run sb-mcp
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from .ask import ask as ask_fn
from .ask import recall as recall_fn
from .config import cfg
from .ingest import ingest_paths
from .memory import learn as learn_memory
from .store import Store
from .tasks import TaskStore

# stdout is the protocol channel for stdio transport — never log there.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("secondbrain.mcp")

def _transport_security() -> TransportSecuritySettings:
    """Allow-list the hosts clients reach us on (DNS-rebinding protection stays ON).

    Without this, the Streamable-HTTP layer rejects any `Host` header other than
    localhost with 421 — which breaks access via the Tailscale IP. We trust exactly
    the configured bind address (+ localhost), extendable via SB_MCP_ALLOWED_HOSTS.
    """
    host = os.getenv("SB_MCP_HOST", "127.0.0.1")
    port = os.getenv("SB_MCP_PORT", "8848")
    hosts = {host, f"{host}:{port}", "localhost", f"localhost:{port}", "127.0.0.1", f"127.0.0.1:{port}"}
    for extra in os.getenv("SB_MCP_ALLOWED_HOSTS", "").split(","):
        if extra.strip():
            hosts.add(extra.strip())
    allowed_hosts = sorted(hosts)
    allowed_origins = sorted({f"http://{h}" for h in allowed_hosts})
    return TransportSecuritySettings(allowed_hosts=allowed_hosts, allowed_origins=allowed_origins)


mcp = FastMCP(
    "second-brain",
    instructions=(
        "Your second brain: a personal RAG assistant over the user's own notes, "
        "docs, code, and learned memories, with answers cited to their sources. "
        "Use `recall` to pull raw source chunks (cheapest, no chat-model call) when "
        "you want to ground your own reasoning; use `ask` for a ready cited answer. "
        "Use `learn` to durably remember a fact, `ingest` to add files, and the task "
        "tools to track to-dos. Retrieval is grounded — no source, no claim."
    ),
    transport_security=_transport_security(),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


@mcp.tool(annotations=_READONLY)
def ask(question: str, top_k: int = 0) -> dict:
    """Answer a question from the user's second brain, cited to their own sources.

    Retrieves the most relevant stored chunks and has the local model answer using
    ONLY that context, returning the answer plus the sources it cited.

    Args:
        question: The natural-language question to answer.
        top_k: How many chunks to retrieve (0 = use the configured default).
    """
    result = ask_fn(question, k=top_k if top_k and top_k > 0 else None)
    return {
        "answer": result["answer"],
        "citations": [
            {"n": s["n"], "source": s["source"], "distance": round(s["distance"], 4)}
            for s in result["sources"]
        ],
        "ungrounded_citations": result.get("invalid_citations", []),
    }


@mcp.tool(annotations=_READONLY)
def recall(query: str, top_k: int = 0) -> dict:
    """Retrieve raw matching chunks from the brain WITHOUT calling the chat model.

    The cheapest grounding primitive: returns the top source passages and their
    similarity distances so the caller can reason over them directly. Use this when
    you want the evidence, not a pre-written answer.

    Args:
        query: What to search for.
        top_k: How many chunks to return (0 = configured default).
    """
    return recall_fn(query, top_k=top_k)


@mcp.tool(annotations=_WRITE)
def ingest(path: str) -> dict:
    """Ingest a file or folder into the second brain (chunk, embed, store).

    Args:
        path: A file or directory path on the host running this server. Supported
            types include .md, .txt, .py/.js/.ts, .json/.yaml/.toml, and .pdf.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"Path not found: {p}")
    files = chunks = 0
    for _f, n in ingest_paths(str(p)):
        files += 1
        chunks += n
    return {"path": str(p), "files": files, "chunks": chunks, "total": Store().count()}


@mcp.tool(annotations=_WRITE)
def learn(fact: str) -> dict:
    """Durably remember a fact: write it as a Markdown memory and ingest it.

    Use for explicit, user-confirmed facts the brain should retain — not for
    unverified guesses.

    Args:
        fact: The fact to remember, as a short self-contained statement.
    """
    result = learn_memory(fact)
    return {"memory_file": str(result["path"]), "chunks": result["chunks"]}


@mcp.tool(annotations=_READONLY)
def list_tasks(status: str = "open") -> dict:
    """List tasks tracked in the second brain.

    Args:
        status: Which tasks to return — "open" (default), "done", or "all".
    """
    tasks = TaskStore().list(status=status)
    return {
        "count": len(tasks),
        "tasks": [
            {"id": t["id"], "title": t["title"], "status": t["status"], "created_at": t["created_at"]}
            for t in tasks
        ],
    }


@mcp.tool(annotations=_WRITE)
def add_task(title: str, notes: str = "") -> dict:
    """Add a task to the second brain.

    Args:
        title: The task description.
        notes: Optional extra detail.
    """
    t = TaskStore().add(title, notes)
    return {"id": t["id"], "title": t["title"], "status": t["status"], "created_at": t["created_at"]}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
def complete_task(task_id: int) -> dict:
    """Mark a task as done.

    Args:
        task_id: The id of the task to complete (from list_tasks).
    """
    try:
        t = TaskStore().complete(task_id)
    except KeyError:
        return {"task_id": task_id, "completed": False, "error": "task not found"}
    return {"task_id": task_id, "completed": True, "status": t["status"]}


@mcp.tool(annotations=_READONLY)
def status() -> dict:
    """Report the active backend, models, store, and how much is stored.

    Returns no secrets — only the gateway base URL and model aliases.
    """
    try:
        chunks = Store().count()
    except Exception as exc:  # store may be unreachable; report rather than crash
        chunks = f"unavailable: {exc}"
    return {
        "store": cfg.store_backend,
        "backend": cfg.base_url,
        "embed_model": cfg.embed_model,
        "chat_model": cfg.chat_model,
        "memory_dir": str(cfg.memory_dir),
        "state_db": str(cfg.state_db),
        "chunks": chunks,
    }


class _BearerAuthASGI:
    """Pure-ASGI bearer-token gate.

    Implemented at the ASGI layer (not Starlette's BaseHTTPMiddleware) so it does
    not buffer the Streamable-HTTP response stream.
    """

    def __init__(self, app, token: str):
        self.app = app
        self._expected = f"Bearer {token}"

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            if headers.get(b"authorization", b"").decode() != self._expected:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


def _run_http() -> None:
    import uvicorn

    host = os.getenv("SB_MCP_HOST", "127.0.0.1")
    port = int(os.getenv("SB_MCP_PORT", "8848"))
    token = os.getenv("SB_MCP_TOKEN")

    app = mcp.streamable_http_app()
    if token:
        app = _BearerAuthASGI(app, token)
        log.info("HTTP transport: bearer-token auth ENABLED")
    else:
        log.warning("HTTP transport: NO token set (SB_MCP_TOKEN) — endpoint is open on %s:%s", host, port)
    log.info("Serving second-brain MCP at http://%s:%s/mcp", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    transport = os.getenv("SB_MCP_TRANSPORT", "stdio").strip().lower()
    if transport in {"http", "streamable-http", "streamable_http"}:
        _run_http()
    else:
        log.info("Serving second-brain MCP over stdio")
        mcp.run()


if __name__ == "__main__":
    main()
