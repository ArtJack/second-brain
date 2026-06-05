"""Import explicit follow-ups into the durable task store."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .morning import _extract_report_tasks, _recent_reports
from .project_context import DEFAULT_OUTPUT_DIR
from .tasks import TaskStore


@dataclass(frozen=True)
class TaskCandidate:
    title: str
    source: str
    kind: str


def normalize_title(title: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", title)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" .:-")
    return text


def dedupe_key(title: str) -> str:
    text = normalize_title(title).lower()
    text = re.sub(r"^[a-z0-9_.-]+:\s+", "", text)
    text = re.sub(r"[^a-z0-9а-яё]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def candidates_from_project_context(root: Path = DEFAULT_OUTPUT_DIR) -> list[TaskCandidate]:
    index = root / "active-projects.md"
    if not index.exists():
        return []
    candidates: list[TaskCandidate] = []
    in_section = False
    for line in index.read_text(errors="ignore").splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Explicit Follow-Ups Across Projects"
            continue
        if not in_section or not line.startswith("- "):
            continue
        raw = line[2:].strip()
        title, source = _split_title_source(raw)
        title = normalize_title(title)
        if title:
            candidates.append(TaskCandidate(title=title, source=source or str(index), kind="project-context"))
    return candidates


def candidates_from_latest_overnight() -> list[TaskCandidate]:
    candidates: list[TaskCandidate] = []
    for item in _extract_report_tasks(_recent_reports(limit=1)):
        title = normalize_title(item["task"])
        if title:
            candidates.append(
                TaskCandidate(
                    title=title,
                    source=item.get("source") or item.get("report") or "overnight report",
                    kind="overnight",
                )
            )
    return candidates


def collect_task_candidates(
    project_context_root: Path = DEFAULT_OUTPUT_DIR,
    *,
    include_overnight: bool = True,
) -> list[TaskCandidate]:
    candidates = candidates_from_project_context(project_context_root)
    if include_overnight:
        candidates += candidates_from_latest_overnight()
    out: list[TaskCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = dedupe_key(candidate.title)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def sync_tasks(
    *,
    store: TaskStore | None = None,
    project_context_root: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
    include_overnight: bool = True,
) -> dict:
    task_store = store or TaskStore()
    existing = {dedupe_key(task["title"]) for task in task_store.list(status="all", limit=1000)}
    added = []
    skipped = []
    for candidate in collect_task_candidates(project_context_root, include_overnight=include_overnight):
        key = dedupe_key(candidate.title)
        if key in existing:
            skipped.append(candidate)
            continue
        if dry_run:
            added.append(candidate)
        else:
            task = task_store.add(candidate.title, notes=f"{candidate.kind}: {candidate.source}")
            added.append(task)
        existing.add(key)
    return {
        "added": added,
        "skipped": skipped,
        "dry_run": dry_run,
    }


def _split_title_source(raw: str) -> tuple[str, str]:
    match = re.match(r"(.+?)\s+\((/[^)]+)\)\s*$", raw)
    if not match:
        return raw, ""
    return match.group(1), match.group(2)
