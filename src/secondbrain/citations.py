"""Validate that an answer only cites sources that were actually retrieved.

The system prompt *asks* the model to cite [n] from the numbered context, but a small
local model can still emit an [n] that was never retrieved. Citations are this product's
trust signal, so we verify them instead of trusting the prompt: any [n] outside
1..n_sources is surfaced to the caller (and flagged in the CLI).
"""
from __future__ import annotations

import re

_CITATION = re.compile(r"\[(\d+)\]")


def cited_numbers(answer: str) -> list[int]:
    """Every [n] reference in the answer, in order of appearance (duplicates kept)."""
    return [int(m) for m in _CITATION.findall(answer)]


def invalid_citations(answer: str, n_sources: int) -> list[int]:
    """Sorted, unique citation numbers that fall outside the valid 1..n_sources range."""
    return sorted({n for n in cited_numbers(answer) if n < 1 or n > n_sources})


def grounding(answer: str, n_sources: int) -> dict:
    """The facts a caller needs to judge whether an answer is actually grounded.

    `invalid_citations` catches an answer that cites a source which does not
    exist. It cannot catch the opposite and more common failure: an answer that
    cites *nothing at all* while stating things as fact. That answer returns an
    empty invalid list and reaches the caller looking clean, which is the one
    thing this product promises it will not do.

    What is deliberately NOT decided here: whether a citation-free answer is
    wrong. The model is instructed to say so plainly when the context does not
    contain the answer, and such a refusal correctly carries no citations. There
    is no structured marker distinguishing a refusal from an ungrounded claim,
    and guessing at one with keyword matching would manufacture false confidence
    in exactly the place this project cannot afford it.

    So this returns observations, not a verdict:

        citations_present   the answer contains at least one [n]
        sources_retrieved   how many sources were available to cite
        cited               the valid numbers actually used, sorted and unique
        invalid             numbers outside 1..n_sources (as before)

    The combination worth a caller's attention is `sources_retrieved > 0` with
    `citations_present` false: the model was handed evidence and used none of
    it. Whether that warrants a warning or a refusal is the caller's policy.
    """
    numbers = cited_numbers(answer)
    return {
        "citations_present": bool(numbers),
        "sources_retrieved": n_sources,
        "cited": sorted({n for n in numbers if 1 <= n <= n_sources}),
        "invalid": sorted({n for n in numbers if n < 1 or n > n_sources}),
    }
