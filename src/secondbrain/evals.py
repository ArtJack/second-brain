"""Local evaluation harness for retrieval and grounded answers.

The default benchmark is deliberately deterministic: retrieve expected sources, then
optionally run answer heuristics. LLM-as-a-judge can be added later without making the
cheap local checks depend on another model call.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ask import ask
from .citations import cited_numbers, invalid_citations
from .config import cfg
from .llm import embed
from .store import Store
from .tracing import TraceRecorder, summarize_traces

DEFAULT_BENCHMARK = Path(__file__).resolve().parents[2] / "evals" / "retrieval.json"
DEFAULT_ABSTENTION_PHRASES = (
    "cannot determine",
    "does not contain",
    "does not mention",
    "don't have",
    "do not have",
    "insufficient context",
    "no information",
    "not provided",
    "not specified",
    "unable to answer",
)

RetrieveFn = Callable[[str, int], list[dict]]
AnswerFn = Callable[[str, int], dict]


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK) -> dict:
    """Load and validate a benchmark JSON file."""
    benchmark_path = Path(path).expanduser()
    try:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Benchmark not found: {benchmark_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark JSON in {benchmark_path}: {exc}") from exc

    if not isinstance(benchmark, dict) or not isinstance(benchmark.get("cases"), list):
        raise ValueError("Benchmark must be an object with a cases array.")
    if not benchmark["cases"]:
        raise ValueError("Benchmark must contain at least one case.")
    if "corpus" in benchmark and not _is_string_list(benchmark["corpus"]):
        raise ValueError("Benchmark corpus must be a string array.")
    if "corpus_collection" in benchmark and (
        not isinstance(benchmark["corpus_collection"], str) or not benchmark["corpus_collection"].strip()
    ):
        raise ValueError("Benchmark corpus_collection must be a non-empty string.")

    seen_ids: set[str] = set()
    for index, case in enumerate(benchmark["cases"], start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Benchmark case #{index} must be an object.")
        case_id = case.get("id")
        query = case.get("query")
        expected_sources = case.get("expected_sources")
        expect_abstain = case.get("expect_abstain", False)
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Benchmark case #{index} needs a non-empty id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate benchmark case id: {case_id}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Benchmark case {case_id!r} needs a non-empty query.")
        if not isinstance(expect_abstain, bool):
            raise ValueError(f"Benchmark case {case_id!r} expect_abstain must be a boolean.")
        if expected_sources is None and expect_abstain:
            case["expected_sources"] = []
        elif not _is_string_list(expected_sources) or (not expected_sources and not expect_abstain):
            raise ValueError(f"Benchmark case {case_id!r} needs expected_sources.")
        for field in ("expected_answer_contains", "expected_answer_contains_any", "abstention_phrases"):
            if field in case and not _is_string_list(case[field]):
                raise ValueError(f"Benchmark case {case_id!r} {field} must be a string array.")
        if "top_k" in case and (not isinstance(case["top_k"], int) or case["top_k"] < 1):
            raise ValueError(f"Benchmark case {case_id!r} top_k must be a positive integer.")
        seen_ids.add(case_id)

    return benchmark


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def _normalized_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def source_matches(source: str, expected: str) -> bool:
    """Match absolute or relative stored paths against portable benchmark suffixes."""
    source = _normalized_path(source)
    expected = _normalized_path(expected)
    return source == expected or source.endswith(f"/{expected}")


def resolve_corpus_paths(benchmark: dict, benchmark_path: str | Path) -> list[Path]:
    """Resolve optional corpus entries relative to their benchmark JSON file."""
    root = Path(benchmark_path).expanduser().resolve().parent
    paths = []
    for entry in benchmark.get("corpus") or []:
        path = Path(entry).expanduser()
        paths.append(path if path.is_absolute() else root / path)
    return paths


def select_cases(benchmark: dict, tags: list[str] | None = None) -> dict:
    """Return a benchmark copy containing cases with any requested tag."""
    if not tags:
        return benchmark
    selected = {
        **benchmark,
        "cases": [
            case for case in benchmark["cases"] if set(case.get("tags") or []).intersection(tags)
        ],
    }
    if not selected["cases"]:
        raise ValueError(f"No benchmark cases matched tag(s): {', '.join(tags)}")
    return selected


def _default_retrieve(query: str, top_k: int) -> list[dict]:
    store = Store()
    if store.count() == 0:
        return []
    return store.query(embed([query])[0], top_k)


def _default_answer(
    question: str,
    top_k: int,
    *,
    trace: TraceRecorder | None = None,
    parent_span_id: str | None = None,
) -> dict:
    return ask(question, k=top_k, trace=trace, parent_span_id=parent_span_id)


def _round_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _retrieval_result(case: dict, hits: list[dict], duration_ms: float) -> dict:
    expected_sources = case.get("expected_sources") or []
    sources = [str(hit.get("metadata", {}).get("source", "?")) for hit in hits]
    if case.get("expect_abstain"):
        return {
            "scored": False,
            "passed": True,
            "expected_sources": [],
            "matched_expected_sources": [],
            "source_recall": None,
            "first_relevant_rank": None,
            "reciprocal_rank": None,
            "retrieved_sources": sources,
            "duration_ms": duration_ms,
        }
    ranks: dict[str, int | None] = {}
    for expected in expected_sources:
        ranks[expected] = next(
            (rank for rank, source in enumerate(sources, start=1) if source_matches(source, expected)),
            None,
        )
    matched = [source for source, rank in ranks.items() if rank is not None]
    first_rank = min((rank for rank in ranks.values() if rank is not None), default=None)
    recall = len(matched) / len(expected_sources)
    return {
        "scored": True,
        "passed": recall == 1.0,
        "expected_sources": expected_sources,
        "matched_expected_sources": matched,
        "source_recall": round(recall, 4),
        "first_relevant_rank": first_rank,
        "reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.0,
        "retrieved_sources": sources,
        "duration_ms": duration_ms,
    }


def _answer_result(case: dict, result: dict, duration_ms: float) -> dict:
    answer_text = str(result.get("answer", ""))
    sources = result.get("sources") or []
    cited = cited_numbers(answer_text)
    invalid = invalid_citations(answer_text, len(sources))
    cited_sources = [
        str(sources[n - 1].get("source", "?"))
        for n in sorted(set(cited))
        if 1 <= n <= len(sources)
    ]
    expected_sources = case.get("expected_sources") or []
    expected_answer_contains = case.get("expected_answer_contains") or []
    expected_answer_contains_any = case.get("expected_answer_contains_any") or []
    missing_phrases = [
        phrase for phrase in expected_answer_contains if phrase.lower() not in answer_text.lower()
    ]
    contains_any_phrase = (
        not expected_answer_contains_any
        or any(phrase.lower() in answer_text.lower() for phrase in expected_answer_contains_any)
    )
    abstention_phrases = case.get("abstention_phrases") or DEFAULT_ABSTENTION_PHRASES
    abstained = any(phrase.lower() in answer_text.lower() for phrase in abstention_phrases)
    require_citations = case.get("require_citations", True)
    cited_expected_sources = [
        expected
        for expected in expected_sources
        if any(source_matches(source, expected) for source in cited_sources)
    ]
    if case.get("expect_abstain"):
        checks = {
            "abstains": abstained,
            "citations_in_bounds": not invalid,
            "contains_expected_phrases": not missing_phrases,
            "contains_any_expected_phrase": contains_any_phrase,
        }
    else:
        checks = {
            "does_not_abstain": not abstained,
            "citation_present": bool(cited) or not require_citations,
            "citations_in_bounds": not invalid,
            "cites_expected_sources": set(cited_expected_sources) == set(expected_sources),
            "contains_expected_phrases": not missing_phrases,
            "contains_any_expected_phrase": contains_any_phrase,
        }
    return {
        "passed": all(checks.values()),
        "rubric_score": round(sum(checks.values()) / len(checks), 4),
        "checks": checks,
        "citations": cited,
        "invalid_citations": invalid,
        "cited_sources": cited_sources,
        "cited_expected_sources": cited_expected_sources,
        "missing_expected_phrases": missing_phrases,
        "expected_answer_contains_any": expected_answer_contains_any,
        "abstained": abstained,
        "answer": answer_text,
        "duration_ms": duration_ms,
    }


def run_benchmark(
    benchmark: dict,
    *,
    top_k: int | None = None,
    include_answers: bool = False,
    retrieve_fn: RetrieveFn | None = None,
    answer_fn: AnswerFn | None = None,
) -> dict:
    """Run a benchmark and return a JSON-serializable report."""
    retrieve_fn = retrieve_fn or _default_retrieve
    traced_default_answer = answer_fn is None
    answer_fn = answer_fn or _default_answer
    default_top_k = top_k or cfg.top_k
    started = time.perf_counter()
    case_results = []
    traces = []

    for case in benchmark["cases"]:
        case_top_k = case.get("top_k") or default_top_k
        trace = TraceRecorder(
            "eval.case",
            attributes={
                "case_id": case["id"],
                "query": case["query"],
                "top_k": case_top_k,
                "expect_abstain": case.get("expect_abstain", False),
            },
        )
        with trace.span("eval.case", kind="workflow", record_trajectory=False) as root_span:
            with trace.span(
                "retrieve",
                kind="retrieval",
                parent_span_id=root_span.span_id,
                attributes={"top_k": case_top_k},
            ) as retrieve_span:
                hits = retrieve_fn(case["query"], case_top_k)
                retrieved_sources = [
                    str(hit.get("metadata", {}).get("source", "?")) for hit in hits
                ]
                retrieve_span.add_attributes(
                    {"hit_count": len(hits), "retrieved_sources": retrieved_sources}
                )
                retrieve_span.set_summary({"hit_count": len(hits)})

            with trace.span(
                "evaluate_retrieval",
                kind="evaluator",
                parent_span_id=root_span.span_id,
            ) as retrieval_eval_span:
                retrieval = _retrieval_result(case, hits, retrieve_span.duration_ms or 0.0)
                retrieval_eval_span.add_attributes(
                    {
                        "scored": retrieval["scored"],
                        "passed": retrieval["passed"],
                        "source_recall": retrieval["source_recall"],
                        "first_relevant_rank": retrieval["first_relevant_rank"],
                    }
                )
                retrieval_eval_span.set_summary(
                    {"scored": retrieval["scored"], "passed": retrieval["passed"]}
                )

            case_result = {
                "id": case["id"],
                "query": case["query"],
                "tags": case.get("tags") or [],
                "top_k": case_top_k,
                "expect_abstain": case.get("expect_abstain", False),
                "retrieval": retrieval,
            }
            if include_answers:
                with trace.span(
                    "answer_path",
                    kind="rag",
                    parent_span_id=root_span.span_id,
                ) as answer_span:
                    if traced_default_answer:
                        answer_result = answer_fn(
                            case["query"],
                            case_top_k,
                            trace=trace,
                            parent_span_id=answer_span.span_id,
                        )
                    else:
                        answer_result = answer_fn(case["query"], case_top_k)
                    answer_span.add_attributes(
                        {
                            "answer_chars": len(str(answer_result.get("answer", ""))),
                            "source_count": len(answer_result.get("sources") or []),
                        }
                    )
                    answer_span.set_summary(
                        {"source_count": len(answer_result.get("sources") or [])}
                    )

                with trace.span(
                    "evaluate_answer",
                    kind="evaluator",
                    parent_span_id=root_span.span_id,
                ) as answer_eval_span:
                    case_result["answer"] = _answer_result(
                        case, answer_result, answer_span.duration_ms or 0.0
                    )
                    answer_eval_span.add_attributes(
                        {
                            "passed": case_result["answer"]["passed"],
                            "rubric_score": case_result["answer"]["rubric_score"],
                            "abstained": case_result["answer"]["abstained"],
                            "checks": case_result["answer"]["checks"],
                        }
                    )
                    answer_eval_span.set_summary(
                        {
                            "passed": case_result["answer"]["passed"],
                            "rubric_score": case_result["answer"]["rubric_score"],
                        }
                    )
            case_result["passed"] = retrieval["passed"] and (
                not include_answers or case_result["answer"]["passed"]
            )
            root_span.add_attributes({"passed": case_result["passed"]})
        trace.finish(attributes={"passed": case_result["passed"]})
        trace_record = trace.to_dict()
        case_result["trace_id"] = trace_record["trace_id"]
        traces.append(trace_record)
        case_results.append(case_result)

    retrieval_cases = [case for case in case_results if case["retrieval"]["scored"]]
    retrieval_passed = sum(1 for case in retrieval_cases if case["retrieval"]["passed"])
    source_recalls = [case["retrieval"]["source_recall"] for case in retrieval_cases]
    reciprocal_ranks = [case["retrieval"]["reciprocal_rank"] for case in retrieval_cases]
    answer_passed = (
        sum(1 for case in case_results if case["answer"]["passed"]) if include_answers else None
    )
    rubric_scores = [case["answer"]["rubric_score"] for case in case_results] if include_answers else []
    abstention_cases = [case for case in case_results if case.get("expect_abstain")]
    answerable_cases = [case for case in case_results if not case.get("expect_abstain")]
    return {
        "benchmark": benchmark.get("name", "unnamed"),
        "description": benchmark.get("description", ""),
        "mode": "retrieval+answers" if include_answers else "retrieval",
        "top_k": default_top_k,
        "passed": all(case["passed"] for case in case_results),
        "summary": {
            "cases": len(case_results),
            "retrieval_cases": len(retrieval_cases),
            "retrieval_skipped": len(case_results) - len(retrieval_cases),
            "retrieval_passed": retrieval_passed,
            "retrieval_hit_rate": round(retrieval_passed / len(retrieval_cases), 4)
            if retrieval_cases
            else None,
            "mean_source_recall": round(sum(source_recalls) / len(source_recalls), 4)
            if source_recalls
            else None,
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
            if reciprocal_ranks
            else None,
            "answer_passed": answer_passed,
            "answer_rubric_score": round(sum(rubric_scores) / len(rubric_scores), 4)
            if rubric_scores
            else None,
            "answerable_cases": len(answerable_cases),
            "abstention_cases": len(abstention_cases),
            "abstention_passed": sum(1 for case in abstention_cases if case["answer"]["passed"])
            if include_answers
            else None,
        },
        "duration_ms": _round_ms(started),
        "trace_summary": summarize_traces(traces),
        "traces": traces,
        "cases": case_results,
    }
