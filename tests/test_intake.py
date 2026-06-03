"""Private 100-question benchmark intake and artifact generation."""
from __future__ import annotations

import json

from secondbrain.intake import (
    BENCHMARK_FILE,
    PROMPTS,
    SESSION_FILE,
    build_private_artifacts,
    intake_status,
    load_session,
    reset_session,
    run_intake,
)


def _input(*responses):
    values = iter(responses)
    return lambda _prompt: next(values)


def test_prompt_bank_has_100_stable_unique_questions():
    assert len(PROMPTS) == 100
    assert len({prompt["id"] for prompt in PROMPTS}) == 100
    assert len({prompt["question"] for prompt in PROMPTS}) == 100


def test_intake_saves_each_response_and_builds_private_artifacts(tmp_path):
    output = []

    summary = run_intake(
        tmp_path,
        input_fn=_input("Call me Art.", "/skip", "/quit"),
        output_fn=output.append,
    )

    session = load_session(tmp_path)
    assert intake_status(session) == {
        "answered": 1,
        "skipped": 1,
        "completed": 2,
        "remaining": 98,
        "total": 100,
    }
    assert summary["benchmark_cases"] == 1
    corpus = (tmp_path / "corpus" / "personal-context.md").read_text(encoding="utf-8")
    benchmark = json.loads((tmp_path / BENCHMARK_FILE).read_text(encoding="utf-8"))
    assert "Call me Art." in corpus
    assert benchmark["corpus_collection"] == "second_brain_private_eval"
    assert benchmark["cases"][0]["reference_answer"] == "Call me Art."
    assert benchmark["cases"][0]["expected_sources"] == ["corpus/personal-context.md"]


def test_intake_resumes_and_back_reopens_previous_prompt(tmp_path):
    run_intake(tmp_path, input_fn=_input("first", "/quit"), output_fn=lambda _message: None)

    summary = run_intake(
        tmp_path,
        input_fn=_input("/back", "replacement", "/quit"),
        output_fn=lambda _message: None,
    )

    session = load_session(tmp_path)
    assert summary["answered"] == 1
    assert session["responses"][PROMPTS[0]["id"]]["answer"] == "replacement"


def test_build_private_artifacts_removes_stale_category_file(tmp_path):
    session = {
        "responses": {
            PROMPTS[0]["id"]: {"answer": "hello"},
        }
    }
    from secondbrain.intake import save_session

    save_session(tmp_path, session)
    build_private_artifacts(tmp_path)
    corpus = tmp_path / "corpus" / "personal-context.md"
    assert corpus.exists()

    save_session(tmp_path, {"responses": {}})
    summary = build_private_artifacts(tmp_path)
    assert not corpus.exists()
    assert summary["benchmark_cases"] == 0


def test_reset_session_removes_generated_private_files(tmp_path):
    run_intake(tmp_path, input_fn=_input("hello", "/quit"), output_fn=lambda _message: None)
    assert (tmp_path / SESSION_FILE).exists()
    assert (tmp_path / BENCHMARK_FILE).exists()

    reset_session(tmp_path)

    assert not (tmp_path / SESSION_FILE).exists()
    assert not (tmp_path / BENCHMARK_FILE).exists()
    assert not (tmp_path / "corpus" / "personal-context.md").exists()
