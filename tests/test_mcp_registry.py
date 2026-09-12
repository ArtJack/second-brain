"""The MCP tool registry and its safety annotations, pinned against the server.

`readOnlyHint` is not documentation. MCP clients use it to decide what to run
without asking: a tool marked read-only can fire unattended. So the annotation on
`ingest` and `learn` is the only mechanical thing standing between "the model
suggested a durable write" and "a durable write happened" — precisely the
explicit-intent gating this project treats as a release blocker.

Nothing pinned that. A regression flipping `learn` to read-only would leave every
existing test green, and the failure would show up as memories appearing without
anyone approving them.

These tests query the FastMCP instance's own registry rather than the module's
attributes. `@mcp.tool` returns the undecorated function, so `callable(m.learn)`
stays true whether or not registration succeeded — an assertion about module
attributes cannot see a registration regression at all.
"""

from __future__ import annotations

import asyncio

import pytest

# name -> (readOnlyHint, destructiveHint, idempotentHint)
# `None` means the annotation is deliberately unset, which MCP reads as "unknown"
# rather than as a promise. Only the values that carry a guarantee are asserted
# as booleans below.
EXPECTED: dict[str, tuple[bool, bool | None, bool | None]] = {
    "ask": (True, None, None),
    "recall": (True, None, None),
    "ingest": (False, False, False),
    "learn": (False, False, False),
    "list_tasks": (True, None, None),
    "add_task": (False, False, False),
    "complete_task": (False, None, True),
    "forget": (False, False, False),
    "status": (True, None, None),
}

# The tools that must never be auto-approvable. Kept as its own list so the
# intent survives even if EXPECTED is edited carelessly.
MUST_NOT_BE_READ_ONLY = {"ingest", "learn", "add_task", "complete_task", "forget"}


@pytest.fixture(scope="module")
def registered() -> dict:
    from secondbrain import mcp_server as m

    return {t.name: t for t in asyncio.run(m.mcp.list_tools())}


def test_registry_holds_exactly_the_expected_tools(registered: dict) -> None:
    """A tool that vanishes from the registry, or appears in it unannounced."""
    assert set(registered) == set(EXPECTED), (
        f"registry drift — missing: {sorted(set(EXPECTED) - set(registered))}, "
        f"unexpected: {sorted(set(registered) - set(EXPECTED))}"
    )


@pytest.mark.parametrize("name", sorted(MUST_NOT_BE_READ_ONLY))
def test_write_tools_are_never_marked_read_only(registered: dict, name: str) -> None:
    """The safety property, stated on its own so the failure message is plain."""
    tool = registered[name]
    assert tool.annotations is not None, f"{name} carries no annotations at all"
    assert tool.annotations.readOnlyHint is False, (
        f"{name} is marked readOnlyHint={tool.annotations.readOnlyHint!r}. MCP "
        f"clients auto-approve read-only tools, so this would let a durable "
        f"write run without the user confirming it."
    )


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_tool_annotations_match_their_declared_intent(
    registered: dict, name: str, expected: tuple[bool, bool | None, bool | None]
) -> None:
    read_only, destructive, idempotent = expected
    ann = registered[name].annotations
    assert ann is not None, f"{name} carries no annotations"
    assert ann.readOnlyHint is read_only, f"{name}.readOnlyHint"
    assert ann.destructiveHint is destructive, f"{name}.destructiveHint"
    assert ann.idempotentHint is idempotent, f"{name}.idempotentHint"


def test_every_tool_is_described(registered: dict) -> None:
    """Descriptions are how a client decides which tool to reach for."""
    for name, tool in registered.items():
        assert tool.description and tool.description.strip(), f"{name} has no description"
