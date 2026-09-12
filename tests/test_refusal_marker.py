"""Telling a refusal from an ungrounded claim, which used to be impossible.

`grounding()` could see that an answer cited nothing. It could not see *why*.
"The context does not mention your birthday" and "Your birthday is 3 June" are
both citation-free, and one of them is the exact failure this product exists to
prevent — so the signal could report the observation and never raise the alarm.
Its own docstring said so, and said that guessing with keyword matching would
manufacture false confidence in the one place this project cannot afford it.

The fix is not a better guess. It is to make the model say which one it is: a
refusal opens with a marker, so a citation-free answer *without* the marker is
unambiguously a claim with no evidence behind it.

Verdict carried this as SB-F-2, its oldest open Major. Note what is settled here
and what is not: this tells the two cases apart. Whether an unsupported answer
should warn, be refused, or pass silently is the owner's policy call, parked as
SB-Q-1, and nothing here decides it.
"""
from __future__ import annotations

from secondbrain.citations import REFUSAL_MARKER, grounding


def test_a_refusal_is_recognised_as_one():
    answer = f"{REFUSAL_MARKER} the retrieved notes say nothing about your birthday."

    result = grounding(answer, n_sources=3)

    assert result["refused"] is True
    assert result["unsupported"] is False
    assert result["citations_present"] is False


def test_a_citation_free_claim_is_the_alarm():
    """Evidence was retrieved, none was used, and nothing says it was refused."""
    answer = "Your preferred invoice format is PDF with line items."

    result = grounding(answer, n_sources=3)

    assert result["unsupported"] is True
    assert result["refused"] is False


def test_a_cited_answer_is_neither():
    answer = "You prefer PDF invoices with line items [1]."

    result = grounding(answer, n_sources=3)

    assert result["unsupported"] is False
    assert result["refused"] is False
    assert result["cited"] == [1]


def test_refusing_with_nothing_retrieved_is_not_an_alarm():
    """An empty store has nothing to be unsupported by."""
    answer = f"{REFUSAL_MARKER} nothing has been ingested yet."

    result = grounding(answer, n_sources=0)

    assert result["refused"] is True
    assert result["unsupported"] is False


def test_an_uncited_answer_with_no_sources_is_not_flagged_as_unsupported():
    """Without evidence to ignore, a missing citation is not evidence of a defect."""
    result = grounding("I am not sure.", n_sources=0)

    assert result["unsupported"] is False


def test_the_marker_is_matched_at_the_start_and_case_insensitively():
    for prefix in (REFUSAL_MARKER, REFUSAL_MARKER.lower(), f"  {REFUSAL_MARKER}"):
        assert grounding(f"{prefix} no answer here.", 2)["refused"] is True

    # Mid-sentence is not a refusal: a model quoting the marker while answering
    # must not be able to silence the alarm.
    mid = f"The answer is PDF. Some systems emit {REFUSAL_MARKER} when unsure."
    assert grounding(mid, 2)["refused"] is False
    assert grounding(mid, 2)["unsupported"] is True


def test_both_prompts_share_one_definition():
    """The instruction was copied into answer() and answer_stream(), so a change
    to one silently gave the streaming path different behaviour from the other."""
    import inspect

    from secondbrain import llm

    source = inspect.getsource(llm)
    assert source.count("You are the user's personal knowledge assistant") == 1
    assert llm.SYSTEM_PROMPT.count(REFUSAL_MARKER) >= 1


def test_the_eval_harness_knows_the_marker_counts_as_abstention():
    from secondbrain.evals import DEFAULT_ABSTENTION_PHRASES

    assert any(REFUSAL_MARKER.lower() in phrase.lower() for phrase in DEFAULT_ABSTENTION_PHRASES)
