"""MCP server for second-brain.

Exposes the second-brain RAG engine as Model Context Protocol tools so any MCP
client (Claude Code, Claude Desktop, a native mobile or desktop client) can query and
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
    SB_MCP_TOKEN  required: clients must send `Authorization: Bearer <token>`.
                  Without it the HTTP transport refuses to start.
    SB_MCP_ALLOW_NO_TOKEN=1
                  serve without a token, for local development. Honoured only
                  when SB_MCP_HOST is a loopback address.

Run:
    uv run sb-mcp                                  # stdio
    SB_MCP_TRANSPORT=http SB_MCP_TOKEN=… uv run sb-mcp

Every tool call first re-reads .env if it changed (config.refresh), so a
rotated key or a moved gateway or store reaches a server that has run for days.
The transport settings above apply when the server starts.
"""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from .ask import ask as ask_fn
from .ask import recall as recall_fn
from .config import cfg, collections_for, refresh
from .ingest import ingest_paths
from .keyword_index import KeywordIndex
from .memory import learn as learn_memory
from .store import Store
from .tasks import MAX_PAGE, STATUSES, TaskStore

# stdout is the protocol channel for stdio transport — never log there.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("secondbrain.mcp")

# Every chunk also lives in the keyword index, so a delete here must reach it too.
_index = KeywordIndex()

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


class _Server(FastMCP):
    """FastMCP whose every tool call starts on the current .env (config.refresh)."""

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        if refresh():
            log.info("Settings re-read from .env before %s", name)
        return await super().call_tool(name, arguments)


mcp = _Server(
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



def _within_allowed_roots(path: str) -> Path:
    """Resolve a caller-supplied path and refuse anything outside the allow-list.

    This tool is reachable over the tailnet by anything holding SB_MCP_TOKEN, and
    paired with `recall` an unrestricted ingest is an arbitrary-file-read
    primitive: ingest the file, then ask for it back. Symlinks are resolved
    before the comparison, because otherwise a link inside an allowed root reads
    whatever it points at.
    """
    resolved = Path(path).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in cfg.ingest_roots]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(
            f"Refusing to ingest {resolved}: outside SB_INGEST_ROOTS "
            f"({', '.join(str(r) for r in roots)})"
        )
    return resolved


@mcp.tool(annotations=_READONLY)
async def ask(question: str, top_k: int = 0, corpus: str = "personal") -> dict:
    """Answer a question from the user's second brain, cited to their own sources.

    Retrieves the most relevant stored chunks and has the local model answer using
    ONLY that context, returning the answer plus the sources it cited.

    Args:
        question: The natural-language question to answer.
        top_k: How many chunks to retrieve (0 = use the configured default).
        corpus: Which body of knowledge to search. "personal" (default) is the
            user's own notes, docs and code. "reference" is study and standards
            material they keep but did not write — ask there only when the
            question is about that material. "all" searches both.
    """
    result = await asyncio.to_thread(
        ask_fn,
        question,
        k=top_k if top_k and top_k > 0 else None,
        collection=collections_for(corpus),
    )
    return {
        "answer": result["answer"],
        "citations": [
            {"n": s["n"], "source": s["source"], "distance": round(s["distance"], 4)}
            for s in result["sources"]
        ],
        "ungrounded_citations": result.get("invalid_citations", []),
        # An answer that cites nothing at all is the failure `ungrounded_citations`
        # structurally cannot see: it reports numbers that do not exist, and there
        # are none. `citations_present` false alongside `sources_retrieved` > 0
        # means the model was handed evidence and cited none of it.
        #
        # Indexed, not `.get()`-ed. A missing key here means ask() stopped
        # producing the signal, and failing open to None would publish that
        # breakage as `citations_present: null` — which a caller testing
        # `if not citations_present` reads as a real alarm and one testing
        # `is False` misses entirely. Better to raise where the bug is.
        "citations_present": result["grounding"]["citations_present"],
        "sources_retrieved": result["grounding"]["sources_retrieved"],
    }


@mcp.tool(annotations=_READONLY)
async def recall(query: str, top_k: int = 0) -> dict:
    """Retrieve raw matching chunks from the brain WITHOUT calling the chat model.

    The cheapest grounding primitive: returns the top source passages and their
    similarity distances so the caller can reason over them directly. Use this when
    you want the evidence, not a pre-written answer.

    Args:
        query: What to search for.
        top_k: How many chunks to return (0 = configured default).
    """
    return await asyncio.to_thread(recall_fn, query, top_k=top_k)


@mcp.tool(annotations=_WRITE)
async def ingest(path: str) -> dict:
    """Ingest a file or folder into the second brain (chunk, embed, store).

    Args:
        path: A file or directory path on the host running this server. Supported
            types include .md, .txt, .py/.js/.ts, .json/.yaml/.toml, and .pdf.
    """
    p = _within_allowed_roots(path)
    if not p.exists():
        raise ValueError(f"Path not found: {p}")
    def _run() -> dict:
        files = chunks = 0
        for _f, n in ingest_paths(str(p)):
            files += 1
            chunks += n
        return {"path": str(p), "files": files, "chunks": chunks, "total": Store().count()}

    return await asyncio.to_thread(_run)


@mcp.tool(annotations=_WRITE)
async def learn(fact: str) -> dict:
    """Durably remember a fact: write it as a Markdown memory and ingest it.

    Use for explicit, user-confirmed facts the brain should retain — not for
    unverified guesses.

    Args:
        fact: The fact to remember, as a short self-contained statement.
    """
    result = await asyncio.to_thread(learn_memory, fact, source="mcp")
    return {"memory_file": str(result["path"]), "chunks": result["chunks"]}


@mcp.tool(annotations=_WRITE)
async def forget(memory_file: str) -> dict:
    """Delete a learned memory: the file and every chunk it produced.

    The counterpart to `learn`. Without it, removing a memory file left its
    chunks retrievable forever — the wrong default for a system whose promise is
    that the owner decides what it knows. Refuses any path outside the memory
    directory, so this is not a general-purpose delete.

    Args:
        memory_file: Path to the memory file, as returned by `learn`.
    """
    resolved = Path(memory_file).expanduser().resolve()
    memory_root = Path(cfg.memory_dir).expanduser().resolve()
    if memory_root not in resolved.parents:
        raise ValueError(f"Refusing to forget {resolved}: not inside {memory_root}")

    def _run() -> dict:
        store = Store()
        store.delete_source(str(resolved))
        # Deleting from the store alone left the memory findable by keyword, and
        # `ask` kept building answers from a memory the owner had forgotten.
        _index.delete_source(getattr(store, "collection_name", None) or cfg.collection, str(resolved))
        existed = resolved.exists()
        if existed:
            resolved.unlink()
        return {"memory_file": str(resolved), "deleted": existed}

    return await asyncio.to_thread(_run)


@mcp.tool(annotations=_READONLY)
async def list_tasks(status: str = "open", limit: int = 50, offset: int = 0) -> dict:
    """List tasks tracked in the second brain, newest first, one page at a time.

    `total` is how many tasks have this status; `showing` is how many this page
    holds. When `offset + showing` is less than `total`, older tasks exist: call
    again with `offset` set to `offset + showing`. `count` is the old name for
    `showing`, not a total.

    Args:
        status: Which tasks to return — "open" (default), "done", or "all".
        limit: Page size, 1 to 200 (default 50).
        offset: How many of the newest matching tasks to skip (default 0).
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}, not {status!r}")
    if not 1 <= limit <= MAX_PAGE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE}, not {limit}")
    if offset < 0:
        raise ValueError(f"offset cannot be negative, not {offset}")

    def _read() -> tuple[int, list[dict]]:
        store = TaskStore()
        return store.count(status), store.list(status=status, limit=limit, offset=offset)

    total, tasks = await asyncio.to_thread(_read)
    return {
        "total": total,
        "showing": len(tasks),
        "offset": offset,
        "count": len(tasks),
        "tasks": [
            {"id": t["id"], "title": t["title"], "status": t["status"], "created_at": t["created_at"]}
            for t in tasks
        ],
    }


@mcp.tool(annotations=_WRITE)
async def add_task(title: str, notes: str = "") -> dict:
    """Add a task to the second brain.

    Args:
        title: The task description.
        notes: Optional extra detail.
    """
    t = await asyncio.to_thread(lambda: TaskStore().add(title, notes))
    return {"id": t["id"], "title": t["title"], "status": t["status"], "created_at": t["created_at"]}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
async def complete_task(task_id: int) -> dict:
    """Mark a task as done.

    Args:
        task_id: The id of the task to complete (from list_tasks).
    """
    try:
        t = await asyncio.to_thread(lambda: TaskStore().complete(task_id))
    except KeyError:
        return {"task_id": task_id, "completed": False, "error": "task not found"}
    return {"task_id": task_id, "completed": True, "status": t["status"]}


@mcp.tool(annotations=_READONLY)
async def status() -> dict:
    """Report the active backend, models, store, and how much is stored.

    Returns no secrets — only the gateway base URL and model aliases.
    """
    try:
        chunks = await asyncio.to_thread(lambda: Store().count())
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
    not buffer the Streamable-HTTP response stream. Every HTTP request and every
    websocket handshake needs the token; only the lifespan events pass without
    one. The comparison is constant-time, as the web API's is.
    """

    def __init__(self, app, token: str):
        if not token.strip():
            raise ValueError("the bearer gate needs a non-empty token")
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "lifespan" and not self._authorized(scope):
            if scope["type"] == "websocket":
                # Closing before accepting refuses the handshake.
                await send({"type": "websocket.close", "code": 1008})
                return
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

    def _authorized(self, scope) -> bool:
        presented = dict(scope.get("headers") or []).get(b"authorization", b"")
        return hmac.compare_digest(presented, self._expected)


class MCPTokenRequired(RuntimeError):
    """The HTTP transport was asked to serve without a token where it may not."""


@dataclass(frozen=True)
class HttpPlan:
    """Where the HTTP transport serves, and behind which token."""

    host: str
    port: int
    token: str | None  # None only on a loopback host with SB_MCP_ALLOW_NO_TOKEN set


def _is_loopback(host: str) -> bool:
    name = host.strip().strip("[]")
    if name.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False  # any other name can resolve to any address


def http_plan(env: Mapping[str, str]) -> HttpPlan:
    """Decide whether the HTTP transport may serve, and how. Fails closed.

    Whoever reaches the port reaches every tool, `forget` and `ingest`
    included. A missing token used to log one warning and serve anyway, so a
    renamed line in `.env` would have opened all of them to the tailnet.
    """
    host = env.get("SB_MCP_HOST", "127.0.0.1")
    port = int(env.get("SB_MCP_PORT", "8848"))
    token = env.get("SB_MCP_TOKEN") or ""
    if token.strip():
        return HttpPlan(host, port, token)
    if env.get("SB_MCP_ALLOW_NO_TOKEN", "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise MCPTokenRequired(
            f"SB_MCP_TOKEN is not set, so the HTTP transport will not start on {host}:{port}: "
            "every tool would be open to whoever can reach it. Set SB_MCP_TOKEN, or, for a "
            "development server on a loopback address, SB_MCP_ALLOW_NO_TOKEN=1."
        )
    if not _is_loopback(host):
        raise MCPTokenRequired(
            f"SB_MCP_ALLOW_NO_TOKEN applies to a loopback address only, and {host!r} is not one. "
            "Set SB_MCP_TOKEN to serve there."
        )
    return HttpPlan(host, port, None)


def _serve_http(plan: HttpPlan) -> None:
    import uvicorn

    app = mcp.streamable_http_app()
    if plan.token is not None:
        app = _BearerAuthASGI(app, plan.token)
        log.info("HTTP transport: bearer-token auth ENABLED")
    else:
        log.warning(
            "HTTP transport: NO token, allowed by SB_MCP_ALLOW_NO_TOKEN on loopback %s:%s", plan.host, plan.port
        )
    log.info("Serving second-brain MCP at http://%s:%s/mcp", plan.host, plan.port)
    uvicorn.run(app, host=plan.host, port=plan.port, log_level="info")


def main() -> None:
    transport = os.getenv("SB_MCP_TRANSPORT", "stdio").strip().lower()
    if transport in {"http", "streamable-http", "streamable_http"}:
        try:
            plan = http_plan(os.environ)
        except MCPTokenRequired as exc:
            log.error("%s", exc)
            raise SystemExit(2) from exc
        _serve_http(plan)
    else:
        log.info("Serving second-brain MCP over stdio")
        mcp.run()


if __name__ == "__main__":
    main()
