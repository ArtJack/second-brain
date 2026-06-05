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

    first = sync_tasks(store=store, project_context_root=root, include_overnight=False)
    second = sync_tasks(store=store, project_context_root=root, include_overnight=False)

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

    res = sync_tasks(store=store, project_context_root=root, dry_run=True, include_overnight=False)

    assert len(res["added"]) == 1
    assert store.list(status="open") == []
