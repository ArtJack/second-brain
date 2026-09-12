"""The citation-free answer — the failure `invalid_citations` cannot see.

`invalid_citations` reports numbers that fall outside the retrieved range. An
answer that cites nothing produces no numbers, so it produces an empty list and
looks identical to a perfectly cited one. This project's whole claim is "no
source, no claim", and until now nothing at runtime could tell the two apart.

`grounding()` reports observations rather than a verdict, so these tests assert
observations. The case worth a caller's attention is the last one: sources were
retrieved and none were cited.

Also covers the `[0]` and negative boundary that the existing citation tests
skipped (SB-F-6) — `invalid_citations` has an `n < 1` branch that no test
exercised with a literal.
"""

from __future__ import annotations

import asyncio

import pytest

from secondbrain.citations import cited_numbers, grounding, invalid_citations


class TestGrounding:
    def test_an_answer_citing_nothing_is_visible_as_such(self) -> None:
        """The headline case: fluent, confident, entirely uncited."""
        g = grounding("The lab gateway is the single front door for all models.", 3)

        assert g["citations_present"] is False
        assert g["sources_retrieved"] == 3
        assert g["cited"] == []
        assert g["invalid"] == []
        # The old signal alone would have said "clean".
        assert invalid_citations("The lab gateway is the single front door.", 3) == []

    def test_a_cited_answer_reports_the_numbers_it_used(self) -> None:
        g = grounding("Qdrant holds the vectors [2] and the gateway routes them [1].", 3)

        assert g["citations_present"] is True
        assert g["cited"] == [1, 2]
        assert g["invalid"] == []
        assert g["sources_retrieved"] == 3

    def test_duplicates_collapse_but_presence_still_holds(self) -> None:
        g = grounding("As [1] says, and again [1], and [1].", 2)

        assert g["citations_present"] is True
        assert g["cited"] == [1]

    def test_out_of_range_citations_are_separated_from_valid_ones(self) -> None:
        g = grounding("Both [1] and [9] agree.", 2)

        assert g["citations_present"] is True
        assert g["cited"] == [1]
        assert g["invalid"] == [9]

    def test_no_sources_and_no_citations_is_not_the_alarming_case(self) -> None:
        """Nothing was retrieved, so an uncited answer is expected, not suspect.

        The distinction matters: `citations_present` false is only interesting
        when `sources_retrieved` is greater than zero.
        """
        g = grounding("Nothing ingested yet.", 0)

        assert g["citations_present"] is False
        assert g["sources_retrieved"] == 0
        assert g["cited"] == []
        assert g["invalid"] == []

    def test_citations_when_no_sources_were_retrieved_are_all_invalid(self) -> None:
        g = grounding("According to [1].", 0)

        assert g["citations_present"] is True
        assert g["cited"] == []
        assert g["invalid"] == [1]


class TestCitationBoundaries:
    """SB-F-6: the `n < 1` branch, never exercised with a literal."""

    @pytest.mark.parametrize("n", [0, 1])
    def test_zero_is_invalid_and_one_is_valid(self, n: int) -> None:
        result = invalid_citations(f"See [{n}].", 2)
        assert result == ([0] if n == 0 else [])

    def test_the_upper_boundary_is_inclusive(self) -> None:
        assert invalid_citations("See [2].", 2) == []
        assert invalid_citations("See [3].", 2) == [3]

    def test_a_bracketed_negative_is_not_read_as_a_citation(self) -> None:
        """`-1` does not match the `[(\\d+)]` pattern, so `[-1]` yields nothing.

        Documented rather than asserted as desirable: the `n < 1` guard exists
        for a `[0]`, and this records that a minus sign never reaches it.
        """
        assert cited_numbers("See [-1].") == []
        assert invalid_citations("See [-1].", 2) == []


class TestAskSurfacesGrounding:
    def test_ask_reports_grounding_when_the_store_is_empty(self, monkeypatch) -> None:
        import secondbrain.ask as a

        monkeypatch.setattr(a, "Store", type("S", (), {"__init__": lambda self, **k: None, "count": lambda self: 0}))

        result = a.ask("anything")

        assert result["grounding"]["citations_present"] is False
        assert result["grounding"]["sources_retrieved"] == 0


class TestProductionWiringCarriesTheSignal:
    """SB-F-11/SB-F-12: the signal must survive at every point a caller reads.

    Deleting the wiring from `ask.py` and from `mcp_server.py` left the suite
    green — `grounding()` was well tested as a function and untested as a
    feature. These assert the three surfaces a caller actually reaches: the
    engine's real answer path, the MCP tool, and the SSE stream the web UI uses.
    """

    def test_ask_carries_grounding_on_the_real_answer_path(self, monkeypatch) -> None:
        """Not the empty-store early return — the path every real answer takes."""
        import secondbrain.ask as a

        hit = {
            "document": "the gateway is the front door",
            "distance": 0.1,
            "metadata": {"source": "lab.md", "name": "lab.md"},
        }
        monkeypatch.setattr(
            a, "Store", type("S", (), {"__init__": lambda self, **k: None,
                                       "count": lambda self: 1,
                                       "query": lambda self, v, k: [hit]}),
        )
        monkeypatch.setattr(a, "embed", lambda texts: [[0.0]])
        monkeypatch.setattr(a, "answer", lambda q, ctx: "It is the front door.")  # no [n]

        result = a.ask("what is the front door?")

        assert "grounding" in result, "ask() dropped the grounding signal"
        assert result["grounding"]["citations_present"] is False
        assert result["grounding"]["sources_retrieved"] == 1
        # The old signal alone cannot tell this from a cited answer.
        assert result["invalid_citations"] == []

    def test_mcp_ask_surfaces_the_signal_to_the_client(self, monkeypatch) -> None:
        from secondbrain import mcp_server as m
        from secondbrain.citations import grounding as g

        monkeypatch.setattr(
            m, "ask_fn",
            lambda q, k=None, collection=None: {
                "answer": "Stated as fact, cited nowhere.",
                "sources": [{"n": 1, "source": "lab.md", "distance": 0.1}],
                "invalid_citations": [],
                "grounding": g("Stated as fact, cited nowhere.", 1),
            },
        )

        out = asyncio.run(m.ask("anything"))

        assert out["citations_present"] is False
        assert out["sources_retrieved"] == 1

    def test_mcp_ask_raises_rather_than_publishing_a_null_signal(self, monkeypatch) -> None:
        """Failing open would emit `citations_present: null`, which reads as a
        real alarm to one caller and as silence to another."""
        from secondbrain import mcp_server as m

        monkeypatch.setattr(
            m, "ask_fn",
            lambda q, k=None, collection=None: {"answer": "A", "sources": [], "invalid_citations": []},
        )

        with pytest.raises(KeyError):
            asyncio.run(m.ask("anything"))

    def test_the_sse_stream_emits_grounding_in_its_done_event(self, monkeypatch) -> None:
        """`/ask/stream` is what the web UI uses for interactive answers."""
        from fastapi.testclient import TestClient

        import secondbrain.server as server

        server._SESSIONS.clear()
        server._rate_hits.clear()
        client = TestClient(server.app)

        monkeypatch.setattr(
            server, "_context_for_stream",
            lambda q, k, collection: {
                "empty": False,
                "context": "[1] (from lab.md)\nthe gateway",
                "sources": [{"n": 1, "source": "lab.md", "distance": 0.1}],
            },
        )
        monkeypatch.setattr(server, "answer_stream", lambda q, ctx: iter(["It is ", "the front door."]))

        resp = client.post("/ask/stream", json={"question": "what?", "k": 1, "corpus": "public"})

        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert "citations_present" in resp.text, "the stream's done event carries no grounding signal"
        assert '"citations_present": false' in resp.text.replace("'", '"')
