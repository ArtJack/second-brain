from pathlib import Path

from secondbrain.morning import (
    _ask_briefing_questions,
    _extract_report_tasks,
    _source_allowed,
    render_morning_markdown,
    run_morning,
)
from secondbrain.project_context import find_project_root


def test_extract_report_tasks_reads_recent_possible_tasks(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        """
# Report

### /notes/project.md

Possible tasks:
- follow up with Alex
- renew certificate

### /notes/other.md

Possible tasks:
- follow up with Alex
- check Qdrant
""".strip()
        + "\n"
    )

    tasks = _extract_report_tasks([report])

    assert tasks == [
        {"task": "follow up with Alex", "source": "/notes/project.md", "report": str(report)},
        {"task": "renew certificate", "source": "/notes/project.md", "report": str(report)},
        {"task": "check Qdrant", "source": "/notes/other.md", "report": str(report)},
    ]


def test_ask_briefing_questions_captures_answers_and_errors():
    def fake_ask(question, k):
        if "bad" in question:
            raise RuntimeError("offline")
        return {"answer": f"answer to {question}", "sources": [{"n": 1, "source": "a.md", "distance": 0.1}]}

    answers = _ask_briefing_questions(questions=["good", "bad"], k=3, ask=fake_ask)

    assert answers[0]["answer"] == "answer to good"
    assert answers[0]["sources"][0]["source"] == "a.md"
    assert answers[1]["error"] == "offline"


def test_render_morning_markdown_includes_core_sections():
    markdown = render_morning_markdown(
        stamp="2026-06-04_08-00-00",
        runs=[
            {
                "scanned": 10,
                "changed": 2,
                "ingested": 2,
                "failed": 0,
                "report_path": "overnight/report.md",
            }
        ],
        open_tasks=[{"id": 7, "title": "Review lease"}],
        report_tasks=[{"task": "follow up with Alex", "source": "/notes/project.md", "report": "r.md"}],
        projects=[{"name": "ifta-agent", "description": "IFTA automation", "path": "/Projects/ifta-agent"}],
        health={"passed": "3", "failed": "1", "skipped": "2", "path": "health/report.md"},
        answers=[
            {
                "question": "What matters?",
                "answer": "The gateway matters [1].",
                "sources": [{"n": 1, "source": "lab.md", "distance": 0.2}],
                "error": None,
            }
        ],
        include_rag=True,
    )

    assert "Latest run scanned 10 file(s), changed 2, ingested 2, failed 0." in markdown
    assert "- #7: Review lease" in markdown
    assert "- follow up with Alex (/notes/project.md)" in markdown
    assert "- ifta-agent: IFTA automation (/Projects/ifta-agent)" in markdown
    assert "Latest health: 3 passed, 1 failed, 2 skipped." in markdown
    assert "The gateway matters [1]." in markdown
    assert "[1] lab.md" in markdown


def test_source_allowed_filters_implementation_noise():
    assert _source_allowed("/Users/artjack/Projects/app/README.md")
    assert not _source_allowed("/Users/artjack/Projects/app/src/main.py")
    assert not _source_allowed("/Users/artjack/Projects/app/tests/test_main.py")
    assert not _source_allowed("/Users/artjack/Projects/app/evals/case.md")


def test_find_project_root_rolls_subfolders_up_to_repo(tmp_path):
    project = tmp_path / "ifta-agent"
    docs = project / "docs"
    docs.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='ifta-agent'\n")

    assert find_project_root(docs) == project


def test_run_morning_writes_markdown_with_injected_ask(tmp_path, monkeypatch):
    import secondbrain.morning as morning

    monkeypatch.setattr(morning, "_load_recent_runs", lambda: [])
    monkeypatch.setattr(morning, "_recent_reports", lambda: [])
    monkeypatch.setattr(morning, "_open_tasks", lambda: [])
    monkeypatch.setattr(morning, "project_inventory", lambda: [])
    monkeypatch.setattr(morning, "_latest_health_summary", lambda: None)

    def fake_ask(question, k):
        return {"answer": "Project summary [1].", "sources": [{"n": 1, "source": "project.md", "distance": 0.3}]}

    result = run_morning(output_dir=tmp_path, include_rag=True, ask=fake_ask)

    assert "Project summary [1]." in result["markdown"]
    assert tmp_path in Path(result["path"]).parents
    assert Path(result["path"]).exists()
