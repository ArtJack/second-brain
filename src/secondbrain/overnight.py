"""Read-only overnight maintenance for the second brain.

The worker scans configured folders, ingests new/changed supported files, extracts
obvious tasks, and writes an auditable morning report. It never edits, moves, or
deletes source files.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
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
        # Agent working trees hold copies of the same documents under a second
        # path, which the store indexes as separate sources: the same note comes
        # back two or three times and spends the answer's whole context budget.
        ".claude",
        ".codex",
        # The evaluation corpora are fiction, written to be *found* by a
        # benchmark. They are shaped exactly like the owner's real notes,
        # because that is what makes them useful fixtures — and it is also what
        # makes them poison here. Asked where the gateway runs, the brain was
        # citing an invented runbook, and nothing in the answer distinguished it
        # from a note the owner actually wrote.
        "evals/corpus",
        "evals/corpus-hard",
        ".pytest_cache",
        ".vercel",
        "dist",
        "build",
    ],
    # Matched against the file name, case-insensitively. A supported suffix is
    # not a licence to read a file: `.json`, `.yaml` and `.toml` are ingestible
    # and are also what credentials are usually written in, and with a remote
    # store the text leaves this machine. Lockfiles are excluded for a duller
    # reason — thousands of chunks of dependency hashes crowd out real notes.
    "exclude_globs": [
        ".env*",
        "*.env",
        "*credential*",
        "*secret*",
        "*token*",
        "*.pem",
        "*.key",
        ".mcp.json",
        ".sops.yaml",
        "settings.local.json",
        "package-lock.json",
        "*.lock",
    ],
    "max_file_mb": 25,
    "max_files_per_run": 250,
    "summaries": False,
}

TASK_SOURCE_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}

# An explicit marker is the owner writing down work. These become durable tasks.
TASK_PATTERNS = [
    re.compile(r"^\s*(?:[-*]\s*)?\[\s\]\s+(.{3,220})$"),
    re.compile(r"^\s*(?:TODO|FIXME|ACTION|FOLLOW[ -]?UP)\s*:?\s+(.{3,220})$", re.IGNORECASE),
]

# Prose that merely sounds like work. "You need to install Node 20 first" in a
# downloaded README is a sentence about someone else's setup, not a commitment;
# it is worth showing in the report and must never reach the task store.
MENTION_PATTERNS = [
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


def excluded_dir_rule(parts: tuple[str, ...], skip_dirs: set[str] | list[str]) -> str | None:
    """The directory rule that excludes this path, or None.

    A rule without a slash matches one path segment at any depth, which is what
    you want for `.venv` or `node_modules`. A rule *with* a slash matches
    consecutive segments, so `evals/corpus` names one directory rather than
    every folder called corpus in the world.

    That distinction is not cosmetic. This file already carried the lesson for a
    different generic word — `data` was removed from the ingest skip list
    because matching it per segment hid every folder of that name, including the
    owner's own notes — and `corpus` was added as a bare name a few hours before
    this, with `~/Projects/verdict/eval/corpus` sitting under a live scan target.
    Combined with retroactive rule enforcement, a bare generic name does not just
    hide a directory; it deletes what was already indexed from it.
    """
    segments = tuple(parts)
    for rule in skip_dirs:
        if "/" not in rule:
            if rule in segments:
                return rule
            continue
        wanted = tuple(part for part in rule.split("/") if part)
        if not wanted:
            continue
        span = len(wanted)
        if any(segments[i:i + span] == wanted for i in range(len(segments) - span + 1)):
            return rule
    return None


def _is_under_skip_dir(path: Path, skip_dirs: set[str]) -> bool:
    return excluded_dir_rule(path.parts, skip_dirs) is not None


def discover_targets(config: dict[str, Any]) -> list[tuple[Path, str | None]]:
    """Each existing target, paired with the collection it belongs to.

    A target is normally a string, meaning the default collection. It may
    instead be `{"path": ..., "collection": ...}`, which is what lets the
    reference corpus have an automatic refresh again: the split that separated
    study material from the owner's own notes had to be protected by excluding
    `istqb` from the scan entirely, because the worker ingested everything into
    the default collection and the next nightly would have undone it. That
    exclusion bought safety by leaving the reference corpus with nothing
    updating it at all.

    String targets are the common case and the live config is hand-edited, so
    they keep working untouched.
    """
    targets: list[tuple[Path, str | None]] = []
    seen: set[Path] = set()
    for raw in config.get("targets", []):
        if isinstance(raw, dict):
            spec = raw.get("path")
            collection = raw.get("collection") or None
        else:
            spec, collection = raw, None
        if not spec:
            continue
        path = Path(str(spec)).expanduser()
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            targets.append((resolved, str(collection) if collection else None))
            seen.add(resolved)
    return targets


def _excluded_by_glob(name: str, globs: list[str]) -> str | None:
    """Return the rule that excludes this file name, or None."""
    lowered = name.lower()
    for rule in globs:
        if fnmatch.fnmatch(lowered, rule.lower()):
            return rule
    return None


@dataclass(frozen=True)
class ScanResult:
    files: list[Path]
    skipped: list[dict[str, str]]
    # Only the files whose target named a collection. Absent means the default,
    # which keeps every existing caller and every string target unchanged.
    routed: dict[str, str] = field(default_factory=dict)

    def collection_for(self, path: Path | str) -> str | None:
        return self.routed.get(str(path))


def _is_foreign_memory(path: Path) -> bool:
    """A memory filed under another collection's subdirectory.

    Memories are written to `<memory_dir>/<collection>/` for anything but the
    default collection. This scan ingests into the default collection, so those
    subdirectories are deliberately not its business: walking them would put a
    sandbox visitor's text into the owner's real brain by the back door — the
    leak the subdirectories exist to close, left open one level down.
    """
    try:
        relative = path.resolve().relative_to(Path(cfg.memory_dir).resolve())
    except (ValueError, OSError):
        return False
    return len(relative.parts) > 1


def scan_targets(config: dict[str, Any]) -> ScanResult:
    """Every supported file under the targets, and why each excluded one was dropped.

    The walk is complete on purpose. `max_files_per_run` bounds the work a run
    does, which is applied in run_overnight against *changed* files; bounding
    discovery instead meant everything past the cap was invisible forever, in
    whatever order the filesystem happened to return.
    """
    skip_dirs = set(config.get("exclude_dirs", DEFAULT_CONFIG["exclude_dirs"]))
    globs = list(config.get("exclude_globs", DEFAULT_CONFIG["exclude_globs"]))
    max_bytes = int(config.get("max_file_mb", DEFAULT_CONFIG["max_file_mb"])) * 1024 * 1024
    files: list[Path] = []
    skipped: list[dict[str, str]] = []
    routed: dict[str, str] = {}

    def consider(path: Path, collection: str | None) -> None:
        if path.suffix.lower() not in SUPPORTED:
            return
        if _is_foreign_memory(path):
            skipped.append({"path": str(path), "rule": "memory for another collection"})
            return
        rule = _excluded_by_glob(path.name, globs)
        if rule:
            skipped.append({"path": str(path), "rule": rule})
            return
        try:
            if path.stat().st_size > max_bytes:
                skipped.append({"path": str(path), "rule": f">{max_bytes // (1024 * 1024)}MB"})
                return
        except OSError as exc:
            skipped.append({"path": str(path), "rule": f"unreadable: {exc.strerror or exc}"})
            return
        files.append(path)
        if collection:
            routed[str(path)] = collection

    for target, target_collection in discover_targets(config):
        if target.is_file():
            consider(target, target_collection)
            continue
        for dirpath, dirnames, filenames in os.walk(target):
            # Pruning in place stops the walk descending into an excluded tree
            # at all, instead of stat-ing every file inside it and discarding
            # the results one by one.
            here = Path(dirpath)
            # Pruning has to ask the anchored question too, or `evals/corpus`
            # would be walked into and only rejected file by file.
            dirnames[:] = [
                name
                for name in dirnames
                if excluded_dir_rule((*here.parts, name), skip_dirs) is None
            ]
            if _is_under_skip_dir(here, skip_dirs):
                continue
            for name in filenames:
                consider(here / name, target_collection)

    return ScanResult(
        files=sorted(files),
        skipped=sorted(skipped, key=lambda item: item["path"]),
        routed=routed,
    )


def discover_supported_files(config: dict[str, Any]) -> list[Path]:
    return scan_targets(config).files


def _extract(text: str, patterns: list[re.Pattern[str]], limit: int) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            item = re.sub(r"\s+", " ", match.group(1)).strip(" .:-")
            if item and item.lower() not in seen:
                found.append(item)
                seen.add(item.lower())
            break
        if len(found) >= limit:
            break
    return found


def extract_tasks(text: str, limit: int = 12) -> list[str]:
    """Work the owner wrote down explicitly. These may become durable tasks."""
    return _extract(text, TASK_PATTERNS, limit)


def extract_mentions(text: str, limit: int = 12) -> list[str]:
    """Sentences that sound like work. Reported for a human to read, never synced."""
    return _extract(text, MENTION_PATTERNS, limit)


def _snippet(text: str, limit: int = 360) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rsplit(" ", 1)[0] + "..."


def _ingest_one(path: Path, collection: str | None = None) -> int:
    chunks = 0
    for _file, n in ingest_paths(path, collection=collection):
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

    scan = scan_targets(config)
    max_changed = int(config.get("max_files_per_run", DEFAULT_CONFIG["max_files_per_run"]))
    stats = {"scanned": 0, "changed": 0, "ingested": 0, "failed": 0, "skipped": len(scan.skipped), "deferred": 0}
    changed_files: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []

    for path in scan.files:
        stats["scanned"] += 1
        try:
            stat = path.stat()
            sha = _sha256(path)
            if not state.file_changed(path, sha, stat.st_size, stat.st_mtime):
                continue
            if stats["changed"] >= max_changed:
                # The budget is spent. Leave the file unrecorded so the next run
                # sees it as changed and picks it up, and say how many there are.
                stats["deferred"] += 1
                continue
            stats["changed"] += 1
            text = read_file(path)
            is_prose = path.suffix.lower() in TASK_SOURCE_SUFFIXES
            tasks = extract_tasks(text) if is_prose else []
            mentions = extract_mentions(text) if is_prose else []
            chunks = 0 if dry_run else _ingest_one(path, scan.collection_for(path))
            if not dry_run:
                state.remember_file(path, sha, stat.st_size, stat.st_mtime, chunks)
            stats["ingested"] += 0 if dry_run else 1
            changed_files.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "chunks": chunks,
                    "tasks": tasks,
                    "mentions": mentions,
                    "collection": scan.collection_for(path),
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
        skipped_files=scan.skipped,
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
    skipped_files: list[dict[str, str]] | None = None,
) -> Path:
    skipped_files = skipped_files or []
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
        f"- Skipped by rule: {stats.get('skipped', 0)}",
        f"- Deferred to a later run: {stats.get('deferred', 0)}",
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
                # Named only when it is not the default, so the common case
                # stays as quiet in the report as it is in the config.
                *([f"- Collection: {item['collection']}"] if item.get("collection") else []),
                f"- Snippet: {item['snippet'] or '(no readable text)'}",
                "",
            ]
        )
        if item["tasks"]:
            lines.append("Possible tasks:")
            for task in item["tasks"]:
                lines.append(f"- {task}")
            lines.append("")
        if item.get("mentions"):
            # A separate heading, because `task-sync` reads "Possible tasks:"
            # and only that. These are for a human to read and act on or ignore.
            lines.append("Mentions (not tasks):")
            for mention in item["mentions"]:
                lines.append(f"- {mention}")
            lines.append("")
    lines.extend(["## Failures", ""])
    if not failed_files:
        lines.append("None.")
    for item in failed_files:
        lines.append(f"- `{item['path']}`: {item['error']}")
    lines.extend(["", "## Skipped By Rule", ""])
    if not skipped_files:
        lines.append("None.")
    for item in skipped_files[:50]:
        lines.append(f"- `{item['path']}` ({item['rule']})")
    if len(skipped_files) > 50:
        lines.append(f"- ...and {len(skipped_files) - 50} more")
    report_path.write_text("\n".join(lines).rstrip() + "\n")
    return report_path
