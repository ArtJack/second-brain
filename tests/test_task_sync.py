"""What `sb task-sync` is allowed to put in the ledger.

The nightly runs `sb task-sync` with no arguments (deploy/run-nightly.sh), so
the default path here is the one that writes to the real store unattended. On
2026-09-11 it had written 34 rows that were nobody's work: acceptance criteria
and fragments lifted out of other repositories' reports by the overnight scan.
The rule these tests hold is that a machine-read line reaches the ledger only
when the owner marked it as work, and a report line only when it is asked for.
"""
from __future__ import annotations

import json

from secondbrain import morning
from secondbrain.overnight import ensure_config, overnight_paths, run_overnight
from secondbrain.project_context import render_active_projects_index
from secondbrain.task_sync import (
    candidates_from_project_context,
    dedupe_key,
    normalize_title,
    sync_tasks,
)
from secondbrain.tasks import TaskStore


def test_normalize_title_removes_markdown_noise():
    assert normalize_title(" **Hybrid search** (vector + BM25)  ") == "Hybrid search (vector + BM25)"
    assert normalize_title("Use `sb morning` now.") == "Use sb morning now"


def test_dedupe_key_collapses_punctuation_and_case():
    assert dedupe_key("Hybrid search + BM25") == dedupe_key("hybrid search bm25")
    assert dedupe_key("second-brain: Hybrid search + BM25") == dedupe_key("Hybrid search BM25")


def test_candidates_from_project_context_reads_active_project_followups(tmp_path):
    root = tmp_path / "project-context"
    root.mkdir()
    (root / "active-projects.md").write_text(
        """
# Active Projects Index

## Explicit Follow-Ups Across Projects

- second-brain: **Hybrid search** (vector + BM25 keyword) + reranking (/repo/README.md)
- second-brain: Web UI deploy (/repo/README.md)

## Source Notes
""".strip()
        + "\n"
    )

    candidates = candidates_from_project_context(root)

    assert [candidate.title for candidate in candidates] == [
        "second-brain: Hybrid search (vector + BM25 keyword) + reranking",
        "second-brain: Web UI deploy",
    ]
    assert candidates[0].source == "/repo/README.md"


def test_sync_tasks_adds_and_dedupes(tmp_path):
    root = tmp_path / "project-context"
    root.mkdir()
    (root / "active-projects.md").write_text(
        """
# Active Projects Index

## Explicit Follow-Ups Across Projects

- second-brain: **Hybrid search** (vector + BM25 keyword) + reranking (/repo/README.md)
- second-brain: **Hybrid search** (vector + BM25 keyword) + reranking (/repo/README.md)

## Source Notes
""".strip()
        + "\n"
    )
    store = TaskStore(path=tmp_path / "tasks.sqlite3")

    first = sync_tasks(store=store, project_context_root=root)
    second = sync_tasks(store=store, project_context_root=root)

    assert len(first["added"]) == 1
    assert first["added"][0]["notes"] == "project-context: /repo/README.md"
    assert len(second["added"]) == 0
    assert len(second["skipped"]) == 1
    assert len(store.list(status="open")) == 1


def test_sync_tasks_dry_run_does_not_write(tmp_path):
    root = tmp_path / "project-context"
    root.mkdir()
    (root / "active-projects.md").write_text(
        """
# Active Projects Index

## Explicit Follow-Ups Across Projects

- second-brain: Web UI deploy (/repo/README.md)

## Source Notes
""".strip()
        + "\n"
    )
    store = TaskStore(path=tmp_path / "tasks.sqlite3")

    res = sync_tasks(store=store, project_context_root=root, dry_run=True)

    assert len(res["added"]) == 1
    assert store.list(status="open") == []


# The shapes that actually filled the ledger: a QA checklist from another repo,
# and a bare word that was typed as a box.
SCANNED_REPORT = """
# QA report for another project

## Acceptance
- [ ] venv/bin/pytest passes
- [ ] No horizontal overflow on mobile (390px)
- [ ] tasks

TODO: renew the server certificate
"""


def _scan_into_an_overnight_report(tmp_path, monkeypatch, text: str) -> None:
    """Run a real overnight scan over `text` and point `task-sync` at its report."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.md").write_text(text)

    root = tmp_path / "overnight"
    config = ensure_config(root / "config.json")
    config.update({"targets": [str(source)], "max_files_per_run": 10})
    (root / "config.json").write_text(json.dumps(config))
    run_overnight(root=root, dry_run=True)

    monkeypatch.setattr(morning, "overnight_paths", lambda *_, **__: overnight_paths(root))


def _empty_project_context(tmp_path):
    root = tmp_path / "project-context"
    root.mkdir()
    return root


def test_the_nightly_path_imports_nothing_from_an_overnight_report(tmp_path, monkeypatch):
    _scan_into_an_overnight_report(tmp_path, monkeypatch, SCANNED_REPORT)
    store = TaskStore(path=tmp_path / "tasks.sqlite3")

    res = sync_tasks(store=store, project_context_root=_empty_project_context(tmp_path))

    assert res["added"] == []
    assert store.list(status="all") == []


def test_opting_in_imports_only_the_explicitly_marked_follow_up(tmp_path, monkeypatch):
    _scan_into_an_overnight_report(tmp_path, monkeypatch, SCANNED_REPORT)
    store = TaskStore(path=tmp_path / "tasks.sqlite3")

    res = sync_tasks(
        store=store,
        project_context_root=_empty_project_context(tmp_path),
        include_overnight=True,
    )

    assert [task["title"] for task in res["added"]] == ["renew the server certificate"]
    assert [task["title"] for task in store.list(status="all")] == ["renew the server certificate"]


def test_project_context_checklist_lines_never_reach_the_store(tmp_path, monkeypatch):
    """The index's follow-up section is the half of `task-sync` that stays on by
    default, so an unchecked box in a project doc must never be rendered into it."""
    import secondbrain.project_context as pc

    project = tmp_path / "agent"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "agent"\ndescription = "Agent"\n')
    (project / "README.md").write_text(
        "# Agent\n\nTODO: wire nightly report\n- [ ] venv/bin/pytest passes\n"
    )
    monkeypatch.setattr(pc, "configured_targets", lambda: [project])

    root = _empty_project_context(tmp_path)
    (root / "active-projects.md").write_text(
        render_active_projects_index(pc.collect_project_context(limit=1))
    )
    store = TaskStore(path=tmp_path / "tasks.sqlite3")

    candidates = candidates_from_project_context(root)
    res = sync_tasks(store=store, project_context_root=root)

    assert [candidate.title for candidate in candidates] == ["agent: wire nightly report"]
    assert [task["title"] for task in res["added"]] == ["agent: wire nightly report"]
