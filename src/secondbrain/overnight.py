"""Read-only overnight maintenance for the second brain.

The worker scans configured folders, ingests new/changed supported files, extracts
obvious tasks, and writes an auditable morning report. It never edits, moves, or
deletes source files.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import cfg
from .ingest import SUPPORTED, ingest_paths, read_file
from .store import Store

DEFAULT_CONFIG = {
    "targets": [
        "~/Documents",
        "~/Desktop",
        "~/Downloads",
        "~/Projects",
        "/Volumes/DISK/AI/artjeck/inbox",
    ],
    "exclude_dirs": [
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        ".next",
        "Library",
        "data",
    ],
    "max_file_mb": 25,
    "max_files_per_run": 250,
    "summaries": False,
}

TASK_SOURCE_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}

TASK_PATTERNS = [
    re.compile(r"^\s*(?:[-*]\s*)?\[\s\]\s+(.{3,220})$"),
    re.compile(r"^\s*(?:TODO|FIXME|ACTION|FOLLOW[ -]?UP)\s*:?\s+(.{3,220})$", re.IGNORECASE),
    re.compile(r"\b((?:need to|follow up with|remember to)\s+.{5,180})", re.IGNORECASE),
]


@dataclass(frozen=True)
class OvernightPaths:
    root: Path
    config: Path
    state_db: Path
    reports: Path
    logs: Path


def overnight_paths(root: Path | None = None) -> OvernightPaths:
    base = root or cfg.state_db.parent / "overnight"
    return OvernightPaths(
        root=base,
        config=base / "config.json",
        state_db=base / "state.sqlite3",
        reports=base / "reports",
        logs=base / "logs",
    )


def ensure_config(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
        return dict(DEFAULT_CONFIG)
    loaded = json.loads(path.read_text())
    merged = dict(DEFAULT_CONFIG)
    merged.update(loaded)
    return merged


class OvernightState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    chunks INTEGER NOT NULL DEFAULT 0,
                    last_seen TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    scanned INTEGER NOT NULL DEFAULT 0,
                    changed INTEGER NOT NULL DEFAULT 0,
                    ingested INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    report_path TEXT
                )
                """
            )

    def start_run(self, started_at: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO runs (started_at) VALUES (?)", (started_at,))
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, *, stats: dict[str, int], report_path: Path) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = ?, scanned = ?, changed = ?, ingested = ?, failed = ?, report_path = ?
                WHERE id = ?
                """,
                (
                    _now(),
                    stats["scanned"],
                    stats["changed"],
                    stats["ingested"],
                    stats["failed"],
                    str(report_path),
                    run_id,
                ),
            )

    def file_changed(self, path: Path, sha256: str, size: int, mtime: float) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT sha256, size, mtime FROM files WHERE path = ?", (str(path),)).fetchone()
        if row is None:
            return True
        return row["sha256"] != sha256 or row["size"] != size or abs(float(row["mtime"]) - mtime) > 0.001

    def remember_file(self, path: Path, sha256: str, size: int, mtime: float, chunks: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (path, sha256, size, mtime, chunks, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    sha256 = excluded.sha256,
                    size = excluded.size,
                    mtime = excluded.mtime,
                    chunks = excluded.chunks,
                    last_seen = excluded.last_seen
                """,
                (str(path), sha256, size, mtime, chunks, _now()),
            )


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_under_skip_dir(path: Path, skip_dirs: set[str]) -> bool:
    return bool(set(path.parts) & skip_dirs)


def discover_targets(config: dict[str, Any]) -> list[Path]:
    targets: list[Path] = []
    seen: set[Path] = set()
    for raw in config.get("targets", []):
        path = Path(str(raw)).expanduser()
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            targets.append(resolved)
            seen.add(resolved)
    return targets


def discover_supported_files(config: dict[str, Any]) -> list[Path]:
    skip_dirs = set(config.get("exclude_dirs", DEFAULT_CONFIG["exclude_dirs"]))
    max_bytes = int(config.get("max_file_mb", DEFAULT_CONFIG["max_file_mb"])) * 1024 * 1024
    max_files = int(config.get("max_files_per_run", DEFAULT_CONFIG["max_files_per_run"]))
    files: list[Path] = []
    for target in discover_targets(config):
        candidates = [target] if target.is_file() else target.rglob("*")
        for path in candidates:
            if len(files) >= max_files:
                return files
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED:
                continue
            if _is_under_skip_dir(path, skip_dirs):
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            files.append(path)
    return sorted(files)


def extract_tasks(text: str, limit: int = 12) -> list[str]:
    tasks: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        for pattern in TASK_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            task = re.sub(r"\s+", " ", match.group(1)).strip(" .:-")
            if task and task.lower() not in seen:
                tasks.append(task)
                seen.add(task.lower())
            break
        if len(tasks) >= limit:
            break
    return tasks


def _snippet(text: str, limit: int = 360) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rsplit(" ", 1)[0] + "..."


def _ingest_one(path: Path) -> int:
    chunks = 0
    for _file, n in ingest_paths(path):
        chunks += n
    return chunks


def run_overnight(*, root: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    paths = overnight_paths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    config = ensure_config(paths.config)
    state = OvernightState(paths.state_db)
    started_at = _now()
    run_id = state.start_run(started_at)

    stats = {"scanned": 0, "changed": 0, "ingested": 0, "failed": 0}
    changed_files: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []

    for path in discover_supported_files(config):
        stats["scanned"] += 1
        try:
            stat = path.stat()
            sha = _sha256(path)
            if not state.file_changed(path, sha, stat.st_size, stat.st_mtime):
                continue
            stats["changed"] += 1
            text = read_file(path)
            tasks = extract_tasks(text) if path.suffix.lower() in TASK_SOURCE_SUFFIXES else []
            chunks = 0 if dry_run else _ingest_one(path)
            if not dry_run:
                state.remember_file(path, sha, stat.st_size, stat.st_mtime, chunks)
            stats["ingested"] += 0 if dry_run else 1
            changed_files.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "chunks": chunks,
                    "tasks": tasks,
                    "snippet": _snippet(text),
                }
            )
        except Exception as exc:  # keep the overnight run moving; report every failure.
            stats["failed"] += 1
            failed_files.append({"path": str(path), "error": str(exc)})

    report = write_report(
        paths=paths,
        run_id=run_id,
        started_at=started_at,
        stats=stats,
        changed_files=changed_files,
        failed_files=failed_files,
        dry_run=dry_run,
    )
    state.finish_run(run_id, stats=stats, report_path=report)
    total_chunks = None
    if not dry_run:
        try:
            total_chunks = Store().count()
        except RuntimeError:
            total_chunks = None
    return {
        "run_id": run_id,
        "report": str(report),
        "config": str(paths.config),
        "stats": stats,
        "total_chunks": total_chunks,
        "dry_run": dry_run,
    }


def write_report(
    *,
    paths: OvernightPaths,
    run_id: int,
    started_at: str,
    stats: dict[str, int],
    changed_files: list[dict[str, Any]],
    failed_files: list[dict[str, str]],
    dry_run: bool,
) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = paths.reports / f"{stamp}-run-{run_id}.md"
    mode = "dry run" if dry_run else "read-only source scan"
    lines = [
        f"# Second Brain Overnight Report - {stamp}",
        "",
        f"- Mode: {mode}",
        f"- Started: {started_at}",
        f"- Scanned files: {stats['scanned']}",
        f"- New or changed files: {stats['changed']}",
        f"- Ingested files: {stats['ingested']}",
        f"- Failed files: {stats['failed']}",
        "",
        "## Changed Files",
        "",
    ]
    if not changed_files:
        lines.append("No new or changed supported files were found.")
    for item in changed_files:
        lines.extend(
            [
                f"### {item['path']}",
                "",
                f"- Size: {item['size']} bytes",
                f"- Chunks: {item['chunks']}",
                f"- Snippet: {item['snippet'] or '(no readable text)'}",
                "",
            ]
        )
        if item["tasks"]:
            lines.append("Possible tasks:")
            for task in item["tasks"]:
                lines.append(f"- {task}")
            lines.append("")
    lines.extend(["## Failures", ""])
    if not failed_files:
        lines.append("None.")
    for item in failed_files:
        lines.append(f"- `{item['path']}`: {item['error']}")
    report_path.write_text("\n".join(lines).rstrip() + "\n")
    return report_path
