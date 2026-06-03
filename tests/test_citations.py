"""The citation-bounds guard: an answer must only cite retrieved sources."""
from secondbrain.citations import cited_numbers, invalid_citations


def test_cited_numbers_extracts_in_order_with_duplicates():
    assert cited_numbers("foo [1] bar [3] baz [1]") == [1, 3, 1]


def test_cited_numbers_none_present():
    assert cited_numbers("no citations in this sentence") == []


def test_invalid_citations_flags_out_of_range():
    # Only 2 sources were retrieved; [3] and [5] are hallucinated references.
    assert invalid_citations("a [1] b [3] c [5] d [2]", n_sources=2) == [3, 5]


def test_invalid_citations_all_valid():
    assert invalid_citations("grounded [1] and [2]", n_sources=2) == []


def test_invalid_citations_with_zero_sources():
    assert invalid_citations("claims [1] something", n_sources=0) == [1]
