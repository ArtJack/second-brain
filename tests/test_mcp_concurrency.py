"""The MCP server must not serialise its clients behind one model call.

FastMCP calls a synchronous tool inline in the event loop
(.venv/.../mcp/server/fastmcp/utilities/func_metadata.py: `return fn(**args)`
when the function is not a coroutine). Every tool here was a plain `def`, and
`ask` holds the loop for the whole embedding-plus-generation round trip. That
server runs 24/7 under launchd and is shared by every Claude Code session and
Claude Desktop on the tailnet, so one slow answer froze all of them — and with
the OpenAI client's default 600 s timeout, "slow" had no upper bound.

Running the engine in a worker thread keeps the loop free. These tests assert
the property rather than the implementation: a long-running tool must not
prevent another tool from making progress.
"""
from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from secondbrain import mcp_server


def test_every_tool_is_a_coroutine():
    """A sync tool is run inline by FastMCP and blocks the whole server."""
    for name in ("ask", "recall", "ingest", "learn", "list_tasks", "add_task", "complete_task", "status"):
        fn = getattr(mcp_server, name)
        assert inspect.iscoroutinefunction(fn), f"{name} is sync and would block the event loop"


def test_a_slow_tool_does_not_block_another(monkeypatch):
    """The property that matters: progress while a model call is in flight.

    Asserted as an *ordering*, not a wall-clock budget. The first version of
    this test measured how long `status` took and required under 0.25s against
    a 0.4s sleep — about 0.1s of headroom over the blocked case, which a shared
    CI runner ate: it failed at 0.26s on a green branch. That number was never
    the claim. The claim is that `status` finishes while `ask` is still in
    flight, and if the loop is blocked it cannot, at any speed of machine.
    """
    finished: list[str] = []

    def slow_ask(question, k=None, collection=None, **kw):
        time.sleep(0.4)
        return {
            "answer": "eventually",
            "sources": [],
            "invalid_citations": [],
            "grounding": {"citations_present": False, "sources_retrieved": 0},
        }

    monkeypatch.setattr(mcp_server, "ask_fn", slow_ask)

    class FastStore:
        def count(self):
            return 7

    monkeypatch.setattr(mcp_server, "Store", lambda *a, **kw: FastStore())

    async def scenario():
        async def run_ask():
            result = await mcp_server.ask("anything")
            finished.append("ask")
            return result

        async def run_status():
            result = await mcp_server.status()
            finished.append("status")
            return result

        slow = asyncio.create_task(run_ask())
        await asyncio.sleep(0.05)  # let the slow tool reach its blocking call
        quick = await run_status()
        await slow
        return quick

    quick = asyncio.run(scenario())

    assert quick["chunks"] == 7
    assert finished == ["status", "ask"], (
        f"status did not finish while ask was in flight (order: {finished}) — the loop is blocked"
    )


def test_the_llm_client_has_a_bounded_timeout():
    """600 s of default patience on a home lab is indistinguishable from a hang."""
    from secondbrain.config import cfg
    from secondbrain.llm import _client

    assert cfg.llm_timeout_s <= 120
    assert _client.timeout == cfg.llm_timeout_s
    assert _client.max_retries <= 1


def test_embed_falls_back_per_item_only_for_a_rejected_batch(monkeypatch):
    """A dead endpoint should fail once, not N more times."""
    import openai

    from secondbrain import llm

    calls = {"n": 0}

    class Boom:
        def create(self, **kw):
            calls["n"] += 1
            raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(llm._client, "embeddings", Boom())

    with pytest.raises(openai.APIConnectionError):
        llm.embed(["a", "b", "c"])

    assert calls["n"] == 1, "a connection failure must not be retried per item"
