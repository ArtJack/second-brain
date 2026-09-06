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
