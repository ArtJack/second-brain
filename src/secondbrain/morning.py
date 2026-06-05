"""Morning briefing assembled from overnight activity, tasks, and RAG."""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .citations import invalid_citations
from .config import cfg
from .health import latest_health_report
from .llm import answer, embed
from .overnight import overnight_paths
from .project_context import project_inventory
from .store import Store
from .tasks import TaskStore

DEFAULT_QUESTIONS = [
    "List the active projects or work areas visible in my notes and project docs. Only use source-backed facts and cite them.",
    "List concrete follow-ups, tasks, or unfinished work visible in my notes and project docs. Only use source-backed facts and cite them.",
]

MORNING_SKIP_SOURCE_PARTS = (
    "/src/",
    "/tests/",
    "/evals/",
    "/__pycache__/",
)


def _morning_dir(root: Path | None = None) -> Path:
    return root or cfg.state_db.parent / "morning"


def _load_recent_runs(limit: int = 5) -> list[dict[str, Any]]:
    db = overnight_paths().state_db
    if not db.exists():
        return []
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, started_at, finished_at, scanned, changed, ingested, failed, report_path
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def _recent_reports(limit: int = 1) -> list[Path]:
    reports = overnight_paths().reports
    if not reports.exists():
        return []
    return sorted(reports.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def _extract_report_tasks(reports: list[Path], limit: int = 12) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    current_source = ""
    in_tasks = False
    for report in reports:
        for line in report.read_text(errors="ignore").splitlines():
            if line.startswith("### "):
                current_source = line[4:].strip()
                in_tasks = False
                continue
            if line.strip() == "Possible tasks:":
                in_tasks = True
                continue
            if in_tasks and line.startswith("- "):
                task = line[2:].strip()
                key = task.lower()
                if task and key not in seen:
                    items.append({"task": task, "source": current_source, "report": str(report)})
                    seen.add(key)
                if len(items) >= limit:
                    return items
                continue
            if in_tasks and line.strip() == "":
                in_tasks = False
    return items


def _open_tasks(limit: int = 8) -> list[dict[str, Any]]:
    try:
        return TaskStore().list(status="open", limit=limit)
    except Exception:
        return []


def _source_allowed(source: str) -> bool:
    return not any(part in source for part in MORNING_SKIP_SOURCE_PARTS)


def _morning_ask(question: str, k: int = 5) -> dict[str, Any]:
    store = Store()
    if store.count() == 0:
        return {"answer": "Nothing ingested yet.", "sources": [], "invalid_citations": []}
    qvec = embed([question])[0]
    hits = store.query(qvec, max(k * 4, k))
    filtered = [hit for hit in hits if _source_allowed(str(hit.get("metadata", {}).get("source", "")))]
    selected = (filtered or hits)[:k]
    context_parts, sources = [], []
    for i, hit in enumerate(selected, start=1):
        meta = hit["metadata"]
        context_parts.append(f"[{i}] (from {meta.get('name', meta.get('source'))})\n{hit['document']}")
        sources.append({"n": i, "source": meta.get("source", "?"), "distance": hit["distance"]})
    answer_text = answer(question, "\n\n".join(context_parts))
    return {
        "answer": answer_text,
        "sources": sources,
        "invalid_citations": invalid_citations(answer_text, len(sources)),
    }


def _format_sources(sources: list[dict[str, Any]]) -> list[str]:
    lines = []
    for source in sources:
        src = source.get("source", "?")
        n = source.get("n", "?")
        distance = source.get("distance")
        suffix = "" if distance is None else f" (dist {distance:.3f})"
        lines.append(f"  [{n}] {src}{suffix}")
    return lines


def _ask_briefing_questions(
    *,
    questions: list[str],
    k: int,
    ask: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for question in questions:
        try:
            result = ask(question, k=k)
            answers.append(
                {
                    "question": question,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "error": None,
                }
            )
        except Exception as exc:
            answers.append({"question": question, "answer": "", "sources": [], "error": str(exc)})
    return answers


def run_morning(
    *,
    output_dir: Path | None = None,
    include_rag: bool = False,
    k: int = 5,
    ask: Callable[..., dict[str, Any]] = _morning_ask,
) -> dict[str, Any]:
    out_dir = _morning_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_recent_runs()
    reports = _recent_reports()
    report_tasks = _extract_report_tasks(reports)
    tasks = _open_tasks()
    projects = project_inventory()
    health = _latest_health_summary()
    answers = _ask_briefing_questions(questions=DEFAULT_QUESTIONS, k=k, ask=ask) if include_rag else []

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = out_dir / f"{stamp}-briefing.md"
    markdown = render_morning_markdown(
        stamp=stamp,
        runs=runs,
        open_tasks=tasks,
        report_tasks=report_tasks,
        projects=projects,
        health=health,
        answers=answers,
        include_rag=include_rag,
    )
    report_path.write_text(markdown)
    return {
        "path": str(report_path),
        "markdown": markdown,
        "runs": runs,
        "open_tasks": tasks,
        "report_tasks": report_tasks,
        "projects": projects,
        "health": health,
        "answers": answers,
    }


def _latest_health_summary() -> dict[str, str] | None:
    report = latest_health_report()
    if not report:
        return None
    summary = {"path": str(report), "passed": "?", "failed": "?", "skipped": "?"}
    for line in report.read_text(errors="ignore").splitlines():
        if line.startswith("- Passed:"):
            summary["passed"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Failed:"):
            summary["failed"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Skipped:"):
            summary["skipped"] = line.split(":", 1)[1].strip()
    return summary


def render_morning_markdown(
    *,
    stamp: str,
    runs: list[dict[str, Any]],
    open_tasks: list[dict[str, Any]],
    report_tasks: list[dict[str, str]],
    projects: list[dict[str, str]],
    health: dict[str, str] | None,
    answers: list[dict[str, Any]],
    include_rag: bool,
) -> str:
    lines = [f"# Morning Briefing - {stamp}", ""]

    lines.extend(["## Overnight", ""])
    if not runs:
        lines.append("No overnight runs found yet.")
    else:
        latest = runs[0]
        lines.append(
            f"Latest run scanned {latest['scanned']} file(s), changed {latest['changed']}, "
            f"ingested {latest['ingested']}, failed {latest['failed']}."
        )
        if latest.get("report_path"):
            lines.append(f"Report: {latest['report_path']}")
    lines.append("")

    lines.extend(["## Open Tasks", ""])
    if not open_tasks:
        lines.append("No open tasks in the task store.")
    for task in open_tasks:
        title = re.sub(r"\s+", " ", str(task.get("title", ""))).strip()
        lines.append(f"- #{task.get('id')}: {title}")
    lines.append("")

    lines.extend(["## Possible Follow-Ups From Files", ""])
    if not report_tasks:
        lines.append("No explicit TODO/checklist/follow-up items found in recent overnight reports.")
    for item in report_tasks:
        lines.append(f"- {item['task']} ({item['source']})")
    lines.append("")

    lines.extend(["## Project Inventory", ""])
    if not projects:
        lines.append("No project metadata found in configured overnight targets.")
    for project in projects:
        lines.append(f"- {project['name']}: {project['description']} ({project['path']})")
    lines.append("")

    lines.extend(["## Health", ""])
    if not health:
        lines.append("No health report found yet. Run `sb health`.")
    else:
        lines.append(
            f"Latest health: {health['passed']} passed, {health['failed']} failed, "
            f"{health['skipped']} skipped."
        )
        lines.append(f"Report: {health['path']}")
    lines.append("")

    lines.extend(["## Cited Brain Summary", ""])
    if not include_rag:
        lines.append("Skipped; run with `--rag` to ask the indexed brain.")
    elif not answers:
        lines.append("No RAG questions were run.")
    for answer in answers:
        lines.append(f"### {answer['question']}")
        lines.append("")
        if answer["error"]:
            lines.append(f"Failed: {answer['error']}")
        else:
            lines.append(answer["answer"] or "(empty answer)")
            source_lines = _format_sources(answer["sources"])
            if source_lines:
                lines.append("")
                lines.append("Sources:")
                lines.extend(source_lines)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
