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
        # Half the retrieved slots went to a document no case asked for. The
        # older metrics cannot see that; this is the point of adding it.
        "mean_precision": 0.3333,
        "mean_passage_recall": None,
        "distractor_rate": None,
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


class TestPassageAndDistractorScoring:
    """Two blind spots that let real defects through a benchmark scoring 1.0.

    The harness scored retrieval at *source* granularity: a case passed if any
    chunk of the right file came back. That is the wrong unit for two failures
    this project has actually had.

    A file of 331 chunks scores a hit on chunk 0. When a backfill bug truncated
    the keyword index to the first few chunks of every long file, every metric
    stayed at 1.0 — the benchmark could not tell "found the document" from
    "found the passage that answers the question". `expected_chunk_contains`
    asks the second question.

    And nothing cost anything for retrieving a *wrong* document, so precision
    was unmeasured and unmeasurable: a change that dragged in three irrelevant
    chunks alongside the right one scored exactly as well as one that did not.
    `forbidden_sources` names the near-miss documents a case must not surface,
    which is what makes a distractor corpus worth building.
    """

    def _hit(self, source, document="context"):
        return {"document": document, "metadata": {"source": source}, "distance": 0.1}

    def test_a_case_fails_when_the_right_file_arrives_without_the_answering_passage(self):
        report = run_benchmark(
            _benchmark([_case(expected_chunk_contains=["fixture-token alpha runs at step 7"])]),
            retrieve_fn=lambda q, k: [self._hit("notes/lab.md", "an early synthetic section")],
        )

        case = report["cases"][0]["retrieval"]
        assert case["source_recall"] == 1.0, "the file itself was retrieved"
        assert case["passage_recall"] == 0.0
        assert case["passed"] is False, "finding the document is not answering the question"
        assert case["missing_passages"] == ["fixture-token alpha runs at step 7"]

    def test_a_case_passes_when_the_answering_passage_is_in_any_retrieved_chunk(self):
        report = run_benchmark(
            _benchmark([_case(expected_chunk_contains=["fixture-token alpha runs at step 7"])]),
            retrieve_fn=lambda q, k: [
                self._hit("notes/lab.md", "intro"),
                self._hit("notes/lab.md", "The synthetic chunk: fixture-token alpha runs at step 7 here."),
            ],
        )

        case = report["cases"][0]["retrieval"]
        assert case["passage_recall"] == 1.0
        assert case["passed"] is True

    def test_passage_matching_ignores_case_and_surrounding_whitespace(self):
        report = run_benchmark(
            _benchmark([_case(expected_chunk_contains=["Fixture-Token Alpha Runs At Step 7"])]),
            retrieve_fn=lambda q, k: [self._hit("notes/lab.md", "...fixture-token  alpha runs\nat step 7...")],
        )

        assert report["cases"][0]["retrieval"]["passage_recall"] == 1.0

    def test_retrieving_a_forbidden_near_miss_fails_the_case(self):
        report = run_benchmark(
            _benchmark([_case(forbidden_sources=["notes/lab-old.md"])]),
            retrieve_fn=lambda q, k: [self._hit("notes/lab.md"), self._hit("notes/lab-old.md")],
        )

        case = report["cases"][0]["retrieval"]
        assert case["source_recall"] == 1.0
        assert case["retrieved_forbidden"] == ["notes/lab-old.md"]
        assert case["passed"] is False

    def test_a_case_naming_no_distractors_is_unaffected(self):
        report = run_benchmark(
            _benchmark([_case()]),
            retrieve_fn=lambda q, k: [self._hit("notes/lab.md"), self._hit("notes/anything.md")],
        )

        case = report["cases"][0]["retrieval"]
        assert case["passed"] is True
        assert case["retrieved_forbidden"] == []
        assert case["passage_recall"] is None

    def test_precision_counts_only_the_slots_that_earned_them(self):
        """One right answer among five slots is not a score of 1.0."""
        report = run_benchmark(
            _benchmark([_case()]),
            retrieve_fn=lambda q, k: [self._hit("notes/lab.md")] + [self._hit(f"notes/junk{i}.md") for i in range(4)],
        )

        assert report["cases"][0]["retrieval"]["precision"] == 0.2

    def test_the_summary_reports_passage_recall_precision_and_distractor_rate(self):
        report = run_benchmark(
            _benchmark(
                [
                    _case(id="a", expected_chunk_contains=["found me"]),
                    _case(id="b", expected_chunk_contains=["never written"], forbidden_sources=["notes/bad.md"]),
                ]
            ),
            retrieve_fn=lambda q, k: [self._hit("notes/lab.md", "found me"), self._hit("notes/bad.md")],
        )

        summary = report["summary"]
        assert summary["mean_passage_recall"] == 0.5
        assert summary["mean_precision"] == 0.5
        # Over the cases that named near misses, not over every case. Only case
        # "b" named any, and it surfaced one, so the rate is 1.0 rather than the
        # 0.5 a whole-suite denominator would report. Spread across a growing
        # suite the number only ever shrinks towards noise.
        assert summary["distractor_rate"] == 1.0

    def test_abstention_cases_are_left_out_of_every_new_metric(self):
        report = run_benchmark(
            _benchmark([_case(id="abstain", expect_abstain=True)]),
            retrieve_fn=lambda q, k: [self._hit("notes/anything.md")],
        )

        case = report["cases"][0]["retrieval"]
        assert case["passed"] is True
        assert case["passage_recall"] is None
        assert case["precision"] is None
        assert report["summary"]["distractor_rate"] is None


class TestHardBenchmarkIsWellFormed:
    """The hard benchmark's own correctness, checked offline.

    A benchmark is a test suite, and a test suite with a typo in it passes for
    the wrong reason. A misspelled path in `forbidden_sources` names a document
    that can never be retrieved, so the distractor case passes every time and
    reports a clean distractor rate while measuring nothing at all. Same for an
    `expected_chunk_contains` phrase that no corpus file actually holds: the
    case could only ever fail.
    """

    def _hard(self):
        from pathlib import Path

        from secondbrain.evals import load_benchmark

        root = Path(__file__).resolve().parents[1]
        return load_benchmark(root / "evals" / "hard.json"), root

    def test_every_named_source_exists_on_disk(self):
        benchmark, root = self._hard()

        named = [
            source
            for case in benchmark["cases"]
            for key in ("expected_sources", "forbidden_sources")
            for source in case.get(key) or []
        ]

        assert named, "the benchmark names no sources at all"
        missing = [source for source in named if not (root / source).is_file()]
        assert missing == [], f"benchmark points at files that do not exist: {missing}"

    def test_every_expected_passage_appears_in_its_expected_source(self):
        benchmark, root = self._hard()

        unfindable = []
        for case in benchmark["cases"]:
            phrases = case.get("expected_chunk_contains") or []
            if not phrases:
                continue
            text = " ".join(
                (root / source).read_text(encoding="utf-8")
                for source in case["expected_sources"]
            )
            flat = " ".join(text.split()).lower()
            unfindable += [
                (case["id"], phrase)
                for phrase in phrases
                if " ".join(phrase.split()).lower() not in flat
            ]

        assert unfindable == [], f"cases asking for text no source contains: {unfindable}"

    def test_no_case_forbids_a_source_it_also_expects(self):
        benchmark, _ = self._hard()

        contradictory = [
            case["id"]
            for case in benchmark["cases"]
            if set(case.get("expected_sources") or []) & set(case.get("forbidden_sources") or [])
        ]

        assert contradictory == [], f"cases that can never pass: {contradictory}"

    def test_it_actually_exercises_the_new_scoring(self):
        """Otherwise it is the old benchmark with more questions."""
        benchmark, _ = self._hard()

        with_passages = [c for c in benchmark["cases"] if c.get("expected_chunk_contains")]
        with_distractors = [c for c in benchmark["cases"] if c.get("forbidden_sources")]

        assert len(with_passages) >= 10
        assert len(with_distractors) >= 3


class TestScoringNothingIsNotPassing:
    """A run that measured nothing exited 0 and printed a table of dashes.

    `sb eval` has been the gate on every retrieval change tonight, run against
    freshly created collections all night long. Its contract is "the change
    ships only if retrieval does not regress" — and a run that scored zero
    cases satisfied that contract vacuously. `retrieval 0/0  hit-rate -  MRR -`,
    exit code 0, reads as a pass in a terminal and *is* a pass to CI.

    Three ways to get there, all reachable by accident: a `--tag` filter that
    happens to select only abstention cases, a benchmark pointed at a
    collection that was never ingested, and a corpus whose files all failed to
    decode. The first cost nothing to produce while writing this test.
    """

    def _abstain_only(self):
        return _benchmark([_case(id="a", expect_abstain=True), _case(id="b", expect_abstain=True)])

    def test_a_report_that_scored_nothing_does_not_pass(self):
        report = run_benchmark(self._abstain_only(), retrieve_fn=lambda q, k: [])

        assert report["summary"]["retrieval_cases"] == 0
        assert report["passed"] is False, "measuring nothing is not the same as measuring success"

    def test_an_abstention_only_run_that_grades_answers_still_passes(self):
        """The boundary: that run measured abstention, which is a real measurement."""
        report = run_benchmark(
            self._abstain_only(),
            include_answers=True,
            retrieve_fn=lambda q, k: [],
            answer_fn=lambda q, k: {"answer": "NOT_IN_SOURCES: the context does not say.", "sources": []},
        )

        assert report["summary"]["retrieval_cases"] == 0
        assert report["passed"] is True
        assert report["failure_reason"] is None

    def test_the_report_names_the_reason_rather_than_only_failing(self):
        report = run_benchmark(self._abstain_only(), retrieve_fn=lambda q, k: [])

        assert "scored no retrieval cases" in report["failure_reason"]

    def test_a_report_that_scored_cases_and_passed_them_still_passes(self):
        report = run_benchmark(
            _benchmark([_case()]),
            retrieve_fn=lambda q, k: [{"document": "d", "metadata": {"source": "notes/lab.md"}, "distance": 0.1}],
        )

        assert report["passed"] is True
        assert report["failure_reason"] is None

    def test_a_benchmark_with_no_cases_at_all_does_not_pass(self):
        report = run_benchmark(_benchmark([]), retrieve_fn=lambda q, k: [])

        assert report["passed"] is False


class TestScoringThroughRecall:
    """`sb eval` scored `hybrid_retrieve` directly, so no reading could show a change to `recall`.

    The MCP server tells agents to ground themselves with `recall`. Until
    2026-09-12 it searched vectors only while every benchmark reading used hybrid, and
    the gate on retrieval changes had no way to see the path agents take.
    `--via-recall` scores what `recall` returns instead.
    """

    def test_recall_retrieve_scores_exactly_what_recall_returned(self, monkeypatch):
        from secondbrain import evals

        seen = {}

        def fake_recall(query, top_k=0):
            seen.update(query=query, top_k=top_k)
            return {
                "count": 1,
                "hits": [{"source": "/Users/artjack/notes/lab.md", "distance": 0.1234, "text": "the answering passage"}],
            }

        monkeypatch.setattr(evals, "recall", fake_recall)

        report = run_benchmark(
            _benchmark([_case(expected_chunk_contains=["answering passage"])]),
            top_k=4,
            retrieve_fn=evals.recall_retrieve,
        )

        assert seen == {"query": "where is the gateway?", "top_k": 4}
        retrieval = report["cases"][0]["retrieval"]
        assert retrieval["retrieved_sources"] == ["/Users/artjack/notes/lab.md"]
        assert retrieval["passage_recall"] == 1.0
        assert report["passed"] is True

    def test_the_cli_flag_scores_through_recall_and_the_report_says_which_path_ran(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from secondbrain import cli, evals

        monkeypatch.setattr(evals, "_default_retrieve", lambda query, top_k: [_hit("/elsewhere/engine.md")])
        monkeypatch.setattr(
            evals,
            "recall",
            lambda query, top_k=0: {
                "count": 1,
                "hits": [{"source": "/Users/artjack/notes/lab.md", "distance": 0.1, "text": "context"}],
            },
        )
        benchmark = tmp_path / "benchmark.json"
        benchmark.write_text(json.dumps(_benchmark([_case()])), encoding="utf-8")

        via_recall = CliRunner().invoke(cli.app, ["eval", str(benchmark), "--via-recall", "--json"])
        default = CliRunner().invoke(cli.app, ["eval", str(benchmark), "--json"])

        recall_report, default_report = json.loads(via_recall.stdout), json.loads(default.stdout)
        assert recall_report["retriever"] == "recall"
        assert recall_report["cases"][0]["retrieval"]["retrieved_sources"] == ["/Users/artjack/notes/lab.md"]
        assert via_recall.exit_code == 0
        # The default is untouched: still hybrid_retrieve, and still failing this case.
        assert default_report["retriever"] == "hybrid_retrieve"
        assert default_report["cases"][0]["retrieval"]["retrieved_sources"] == ["/elsewhere/engine.md"]
        assert default.exit_code == 1
