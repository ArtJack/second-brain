"""Validate that an answer only cites sources that were actually retrieved.

The system prompt *asks* the model to cite [n] from the numbered context, but a small
local model can still emit an [n] that was never retrieved. Citations are this product's
trust signal, so we verify them instead of trusting the prompt: any [n] outside
1..n_sources is surfaced to the caller (and flagged in the CLI).
"""
from __future__ import annotations

import re

_CITATION = re.compile(r"\[(\d+)\]")

# The model opens with this, and only this, when the retrieved context does not
# answer the question. It exists so a citation-free answer can be classified
# instead of guessed at: "the notes do not mention your birthday" and "your
# birthday is 3 June" are both citation-free, and only one of them is the failure
# this product exists to prevent.
REFUSAL_MARKER = "NOT_IN_SOURCES:"
_REFUSAL_AT_START = re.compile(rf"^\s*{re.escape(REFUSAL_MARKER)}", re.IGNORECASE)


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

    This used to stop one step short. A citation-free answer could be a refusal
    ("the context does not mention your birthday") or an ungrounded claim ("your
    birthday is 3 June"), and nothing distinguished them, so the alarm could not
    be raised without also firing on every honest refusal. Keyword-matching the
    difference would have manufactured false confidence in the one place this
    project cannot afford it.

    The model now says which it is: a refusal opens with `REFUSAL_MARKER`. So:

        citations_present   the answer contains at least one [n]
        sources_retrieved   how many sources were available to cite
        cited               the valid numbers actually used, sorted and unique
        invalid             numbers outside 1..n_sources
        refused             the answer declares the context did not cover it
        unsupported         evidence was retrieved, none was cited, and no
                            refusal was declared — the actual alarm

    `unsupported` is an observation, not a verdict. Whether it warrants a warning,
    a refusal to display, or nothing at all is the caller's policy.
    """
    numbers = cited_numbers(answer)
    refused = bool(_REFUSAL_AT_START.match(answer or ""))
    return {
        "citations_present": bool(numbers),
        "sources_retrieved": n_sources,
        "cited": sorted({n for n in numbers if 1 <= n <= n_sources}),
        "invalid": sorted({n for n in numbers if n < 1 or n > n_sources}),
        "refused": refused,
        "unsupported": bool(n_sources) and not numbers and not refused,
    }
