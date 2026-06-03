"""Offline tests for the local retrieval and grounded-answer evaluation harness."""
from __future__ import annotations

import json

import pytest

from secondbrain.evals import (
    load_benchmark,
    resolve_corpus_paths,
    run_benchmark,
    select_cases,
    source_matches,
)


def _benchmark(cases):
    return {"name": "test", "cases": cases}


def _case(**overrides):
    case = {
        "id": "gateway",
        "query": "where is the gateway?",
        "expected_sources": ["notes/lab.md"],
    }
    case.update(overrides)
    return case


def _hit(source):
    return {"document": "context", "metadata": {"source": source}, "distance": 0.1}


def test_source_matches_portable_suffixes():
    assert source_matches("/Users/artjack/notes/lab.md", "notes/lab.md")
    assert source_matches("notes/lab.md", "notes/lab.md")
    assert not source_matches("/Users/artjack/notes/other.md", "notes/lab.md")


def test_load_benchmark_validates_required_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cases": [{"id": "missing-query"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="needs a non-empty query"):
        load_benchmark(path)


def test_load_benchmark_rejects_empty_corpus_collection(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"corpus_collection": "", "cases": [_case()]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corpus_collection"):
        load_benchmark(path)


def test_retrieval_report_scores_hit_rate_source_recall_and_mrr():
    benchmark = _benchmark(
        [
            _case(),
            _case(id="multi-source", expected_sources=["notes/lab.md", "notes/other.md"]),
            _case(id="miss", expected_sources=["notes/missing.md"]),
        ]
    )

    def retrieve(_query, _top_k):
        return [_hit("/tmp/unrelated.md"), _hit("/Users/artjack/notes/lab.md")]

    report = run_benchmark(benchmark, top_k=5, retrieve_fn=retrieve)

    assert report["passed"] is False
    assert report["summary"] == {
        "cases": 3,
        "retrieval_cases": 3,
        "retrieval_skipped": 0,
        "retrieval_passed": 1,
        "retrieval_hit_rate": 0.3333,
        "mean_source_recall": 0.5,
        "mrr": 0.3333,
        "answer_passed": None,
        "answer_rubric_score": None,
        "answerable_cases": 3,
        "abstention_cases": 0,
        "abstention_passed": None,
    }
    assert report["cases"][0]["retrieval"]["first_relevant_rank"] == 2


def test_answer_heuristics_accept_grounded_answer():
    benchmark = _benchmark([_case(expected_answer_contains=["4000"])])

    report = run_benchmark(
        benchmark,
        include_answers=True,
        retrieve_fn=lambda _query, _top_k: [_hit("/Users/artjack/notes/lab.md")],
        answer_fn=lambda _query, _top_k: {
            "answer": "The gateway listens on port 4000 [1].",
            "sources": [{"n": 1, "source": "/Users/artjack/notes/lab.md", "distance": 0.1}],
            "invalid_citations": [],
        },
    )

    assert report["passed"] is True
    assert report["summary"]["answer_passed"] == 1
    assert report["trace_summary"] == {
        "traces": 1,
        "spans": 5,
        "trajectory_steps": 4,
        "error_spans": 0,
    }
    assert report["cases"][0]["trace_id"] == report["traces"][0]["trace_id"]
    assert report["traces"][0]["attributes"]["query"] == "where is the gateway?"
    assert [step["name"] for step in report["traces"][0]["trajectory"]] == [
        "retrieve",
        "evaluate_retrieval",
        "answer_path",
        "evaluate_answer",
    ]
    assert report["cases"][0]["answer"]["checks"] == {
        "does_not_abstain": True,
        "citation_present": True,
        "citations_in_bounds": True,
        "cites_expected_sources": True,
        "contains_expected_phrases": True,
        "contains_any_expected_phrase": True,
    }


def test_answer_heuristics_reject_uncited_or_unexpected_source():
    benchmark = _benchmark([_case(expected_answer_contains=["4000"])])

    report = run_benchmark(
        benchmark,
        include_answers=True,
        retrieve_fn=lambda _query, _top_k: [_hit("/Users/artjack/notes/lab.md")],
        answer_fn=lambda _query, _top_k: {
            "answer": "The gateway listens on port 4000.",
            "sources": [{"n": 1, "source": "/Users/artjack/notes/other.md", "distance": 0.1}],
            "invalid_citations": [],
        },
    )

    answer = report["cases"][0]["answer"]
    assert answer["passed"] is False
    assert answer["checks"]["citation_present"] is False
    assert answer["checks"]["cites_expected_sources"] is False


def test_answer_heuristics_compute_citation_bounds_independently():
    benchmark = _benchmark([_case()])

    report = run_benchmark(
        benchmark,
        include_answers=True,
        retrieve_fn=lambda _query, _top_k: [_hit("/Users/artjack/notes/lab.md")],
        answer_fn=lambda _query, _top_k: {
            "answer": "The gateway is described in the notes [2].",
            "sources": [{"n": 1, "source": "/Users/artjack/notes/lab.md", "distance": 0.1}],
        },
    )

    answer = report["cases"][0]["answer"]
    assert answer["passed"] is False
    assert answer["invalid_citations"] == [2]
    assert answer["checks"]["citations_in_bounds"] is False


def test_multi_source_answer_must_cite_every_expected_source():
    benchmark = _benchmark(
        [_case(expected_sources=["notes/lab.md", "notes/deploy.md"])]
    )

    report = run_benchmark(
        benchmark,
        include_answers=True,
        retrieve_fn=lambda _query, _top_k: [
            _hit("/Users/artjack/notes/lab.md"),
            _hit("/Users/artjack/notes/deploy.md"),
        ],
        answer_fn=lambda _query, _top_k: {
            "answer": "The lab notes describe the gateway [1].",
            "sources": [
                {"n": 1, "source": "/Users/artjack/notes/lab.md", "distance": 0.1},
                {"n": 2, "source": "/Users/artjack/notes/deploy.md", "distance": 0.2},
            ],
        },
    )

    answer = report["cases"][0]["answer"]
    assert answer["passed"] is False
    assert answer["cited_expected_sources"] == ["notes/lab.md"]
    assert answer["checks"]["cites_expected_sources"] is False


def test_abstention_case_skips_retrieval_scoring_and_grades_answer():
    benchmark = _benchmark(
        [{"id": "unknown", "query": "What is the birthday?", "expect_abstain": True}]
    )

    report = run_benchmark(
        benchmark,
        include_answers=True,
        retrieve_fn=lambda _query, _top_k: [_hit("/Users/artjack/notes/lab.md")],
        answer_fn=lambda _query, _top_k: {
            "answer": "The provided context does not mention a birthday.",
            "sources": [{"n": 1, "source": "/Users/artjack/notes/lab.md", "distance": 0.1}],
        },
    )

    assert report["passed"] is True
    assert report["summary"]["retrieval_cases"] == 0
    assert report["summary"]["retrieval_skipped"] == 1
    assert report["summary"]["abstention_passed"] == 1
    assert report["cases"][0]["answer"]["checks"]["abstains"] is True


def test_select_cases_filters_by_any_requested_tag():
    benchmark = _benchmark(
        [
            _case(id="one", tags=["smoke"]),
            _case(id="two", tags=["slow"]),
            _case(id="three", tags=["other"]),
        ]
    )

    selected = select_cases(benchmark, ["smoke", "slow"])

    assert [case["id"] for case in selected["cases"]] == ["one", "two"]


def test_resolve_corpus_paths_relative_to_benchmark(tmp_path):
    benchmark_path = tmp_path / "suite.json"

    assert resolve_corpus_paths({"corpus": ["corpus"]}, benchmark_path) == [tmp_path / "corpus"]
