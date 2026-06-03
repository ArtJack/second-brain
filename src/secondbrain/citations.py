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
