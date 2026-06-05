"""Generate source-backed project context notes for the second brain.

The discovery here walks configured folders that may live on a slow/flaky SMB
mount. A single ``stat``/``read`` against a stalled mount can block for minutes,
so every filesystem-touching unit runs under a wall-clock budget: each unit is
executed in a daemon thread that we abandon if it overruns, and the whole run
stops once an overall deadline passes. The command therefore always finishes in
bounded time (healthy run is well under a second), which is what keeps the
nightly launchd job from hanging on its first step.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .overnight import DEFAULT_CONFIG, overnight_paths

DEFAULT_OUTPUT_DIR = Path("/Volumes/DISK/AI/artjeck/inbox/project-context")
PROJECT_MARKERS = ("pyproject.toml", "package.json", ".git")
README_NAMES = ("README.md", "readme.md", "README.txt")
DOC_NAMES = ("README.md", "IFTA_RUNBOOK.md", "knowledge_base.md")

# Wall-clock budget for an entire project-context run (seconds). Tunable via env
# so the nightly can tighten/loosen it without a code change. Kept comfortably
# under the ~30s target; a healthy run finishes in a fraction of a second.
PROJECT_CONTEXT_BUDGET_S = float(os.environ.get("SB_PROJECT_CONTEXT_BUDGET_S", "20"))
# No single filesystem unit (one target scan, one describe, one context build)
# may block longer than this before we abandon it and move on.
PER_OP_TIMEOUT_S = float(os.environ.get("SB_PROJECT_CONTEXT_OP_TIMEOUT_S", "6"))
# Never transfer more than this from a single document, even if max_file_mb is
# larger — excerpts/commands/follow-ups only need the head of a file.
READ_HEAD_CAP = 1024 * 1024

log = logging.getLogger("secondbrain.project_context")

_CONFIG_CACHE: dict | None = None


@dataclass(frozen=True)
class ProjectContext:
    name: str
    path: Path
    description: str
    sources: list[Path]
    commands: list[str]
    follow_ups: list[str]
    excerpts: dict[str, str] = field(default_factory=dict)


class _Deadline:
    """A shared wall-clock budget for one project-context run."""

    def __init__(self, budget_s: float) -> None:
        self._end = time.monotonic() + max(0.0, budget_s)

    def remaining(self) -> float:
        return self._end - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def op_timeout(self) -> float:
        """Time to allow one guarded op: the smaller of the per-op cap and what
        is left of the overall budget, so a single op never overruns the run."""
        return max(0.0, min(PER_OP_TIMEOUT_S, self.remaining()))


def _guard(fn, *, timeout: float, default):
    """Run ``fn`` in a daemon thread; return ``default`` if it does not finish
    within ``timeout``. A stalled filesystem syscall leaks the (daemon) thread,
    but it never blocks the run and is reaped when the process exits."""
    if timeout <= 0:
        return default
    box: dict = {}
    done = threading.Event()

    def _run() -> None:
        try:
            box["value"] = fn()
        except Exception:  # treat any failure as "no result"; keep the run moving
            box["value"] = default
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()
    if done.wait(timeout):
        return box.get("value", default)
    return default


def _log_timeout(stage: str, target) -> None:
    log.warning("project-context: %s exceeded time budget for %s; skipping", stage, target)


def _load_config() -> dict:
    """Merged overnight config (defaults + on-disk), cached for the run. Used
    only for the ``exclude_dirs``/``max_file_mb`` bounds — target discovery
    stays in :func:`configured_targets` so a missing file means "no targets"."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        merged = dict(DEFAULT_CONFIG)
        try:
            path = overnight_paths().config
            if path.exists():
                merged.update(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
        _CONFIG_CACHE = merged
    return _CONFIG_CACHE


def _max_file_bytes() -> int:
    mb = int(_load_config().get("max_file_mb", DEFAULT_CONFIG["max_file_mb"]))
    return max(1, mb) * 1024 * 1024


def _read_head(path: Path) -> str:
    """Read at most ``min(READ_HEAD_CAP, max_file_mb)`` bytes of text. Bounds the
    amount transferred from a single (possibly SMB-hosted) document."""
    limit = min(READ_HEAD_CAP, _max_file_bytes())
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def _exclude_dirs() -> set[str]:
    return set(_load_config().get("exclude_dirs", DEFAULT_CONFIG["exclude_dirs"]))


def configured_targets() -> list[Path]:
    config_path = overnight_paths().config
    if not config_path.exists():
        return []
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return []
    return [Path(str(raw)).expanduser() for raw in config.get("targets", [])]


def find_project_root(path: Path) -> Path | None:
    # os.path.abspath (lexical) instead of Path.resolve(): resolve() calls
    # realpath() which stats every symlink component — extra round trips on SMB.
    # We only need a stable, normalized path to walk markers up the tree.
    current = Path(os.path.abspath(os.path.expanduser(str(path))))
    for candidate in [current, *current.parents]:
        if candidate == candidate.parent:
            break
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return None


def _target_candidates(raw: Path, exclude: set[str]) -> list[Path]:
    """Candidate project roots contributed by one configured target. Runs inside
    a :func:`_guard`, so a stalled ``iterdir``/``stat`` here is abandoned."""
    base = raw.parent if raw.is_file() else raw
    if not base.exists():
        return []
    root = find_project_root(base)
    if root:
        return [root]
    found: list[Path] = [base]
    try:
        for path in base.iterdir():
            if path.name in exclude:
                continue
            if not path.is_dir():
                continue
            found.append(find_project_root(path) or path)
    except OSError:
        pass
    return found


def discover_project_roots(
    targets: list[Path] | None = None,
    limit: int = 30,
    deadline: _Deadline | None = None,
) -> list[Path]:
    deadline = deadline or _Deadline(PROJECT_CONTEXT_BUDGET_S)
    roots = targets if targets is not None else configured_targets()
    exclude = _exclude_dirs()

    candidates: list[Path] = []
    for raw in roots:
        if deadline.expired():
            _log_timeout("discovery", raw)
            break
        found = _guard(
            lambda raw=raw: _target_candidates(raw, exclude),
            timeout=deadline.op_timeout(),
            default=None,
        )
        if found is None:
            _log_timeout("target scan", raw)
            continue
        candidates.extend(found)

    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if deadline.expired() or len(out) >= limit:
            break
        key = os.path.abspath(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        desc = _guard(
            lambda candidate=candidate: describe_project(candidate),
            timeout=deadline.op_timeout(),
            default="",
        )
        if desc:
            out.append(candidate)
    return out


def describe_project(path: Path) -> str:
    pyproject = path / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(_read_head(pyproject))
            description = data.get("project", {}).get("description")
            if description:
                return str(description)
        except Exception:
            pass
    package = path / "package.json"
    if package.exists():
        try:
            description = json.loads(_read_head(package)).get("description")
            if description:
                return str(description)
        except Exception:
            pass
    readme = next((path / name for name in README_NAMES if (path / name).exists()), None)
    if not readme:
        return ""
    lines = [line.strip() for line in _read_head(readme).splitlines()]
    heading = next((line.lstrip("# ").strip() for line in lines if line.startswith("#")), "")
    paragraph = next((line for line in lines if line and not line.startswith("#") and not line.startswith("```")), "")
    return paragraph or heading


def project_inventory(limit: int = 12) -> list[dict[str, str]]:
    projects = []
    for root in discover_project_roots(limit=limit):
        projects.append({"name": root.name, "path": str(root), "description": describe_project(root)})
    return projects


def _build_context(root: Path) -> ProjectContext:
    sources = _context_sources(root)
    return ProjectContext(
        name=root.name,
        path=root,
        description=describe_project(root),
        sources=sources,
        commands=_commands(root),
        follow_ups=_follow_ups(sources),
        # Read excerpts here, inside the guarded build, so rendering touches no
        # filesystem and the whole run stays bounded.
        excerpts={str(source): _excerpt(source) for source in sources},
    )


def collect_project_context(limit: int = 20, *, budget: float | None = None) -> list[ProjectContext]:
    deadline = _Deadline(PROJECT_CONTEXT_BUDGET_S if budget is None else budget)
    contexts: list[ProjectContext] = []
    for root in discover_project_roots(limit=limit, deadline=deadline):
        if deadline.expired():
            _log_timeout("context build", root)
            break
        context = _guard(
            lambda root=root: _build_context(root),
            timeout=deadline.op_timeout(),
            default=None,
        )
        if context is not None:
            contexts.append(context)
        else:
            _log_timeout("context build", root)
    return contexts


def write_project_context_notes(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int = 20,
    *,
    budget: float | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    contexts = collect_project_context(limit=limit, budget=budget)
    index_path = output_dir / "active-projects.md"
    index_path.write_text(render_active_projects_index(contexts))
    written.append(index_path)
    for context in contexts:
        path = output_dir / f"{_slug(context.name)}.md"
        path.write_text(render_project_context(context))
        written.append(path)
    return written


def render_active_projects_index(contexts: list[ProjectContext]) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Active Projects Index",
        "",
        f"- Generated: {stamp}",
        "- Purpose: broad source-backed index for questions like \"What active projects should I pay attention to?\"",
        "",
        "## Active Projects",
        "",
    ]
    if not contexts:
        lines.append("No project contexts found.")
    for context in contexts:
        lines.append(f"- {context.name}: {context.description or 'No description found.'} ({context.path})")
    lines.extend(["", "## Explicit Follow-Ups Across Projects", ""])
    follow_ups = [(context.name, item) for context in contexts for item in context.follow_ups]
    if not follow_ups:
        lines.append("No explicit TODO/checklist/follow-up markers found in project docs.")
    for name, item in follow_ups:
        lines.append(f"- {name}: {item}")
    lines.extend(["", "## Source Notes", ""])
    for context in contexts:
        sources = ", ".join(str(source) for source in context.sources) or "No README/doc source files found"
        lines.append(f"- {context.name}: {sources}")
    return "\n".join(lines).rstrip() + "\n"


def render_project_context(context: ProjectContext) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Project Context: {context.name}",
        "",
        f"- Generated: {stamp}",
        f"- Project root: {context.path}",
        f"- Description: {context.description or 'No description found.'}",
        "",
        "## Source Files",
        "",
    ]
    for source in context.sources:
        lines.append(f"- {source}")
    if not context.sources:
        lines.append("- No README/doc source files found.")
    lines.extend(["", "## Useful Commands", ""])
    if not context.commands:
        lines.append("No explicit project commands found.")
    for command in context.commands:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Explicit Follow-Ups", ""])
    if not context.follow_ups:
        lines.append("No explicit TODO/checklist/follow-up markers found in source docs.")
    for item in context.follow_ups:
        lines.append(f"- {item}")
    lines.extend(["", "## Source Excerpts", ""])
    for source in context.sources:
        excerpt = context.excerpts.get(str(source), "")
        if excerpt:
            lines.extend([f"### {source.name}", "", excerpt, ""])
    return "\n".join(lines).rstrip() + "\n"


def _context_sources(root: Path) -> list[Path]:
    max_bytes = _max_file_bytes()

    def _add(path: Path, into: list[Path]) -> None:
        try:
            if path.is_file() and path.stat().st_size <= max_bytes:
                into.append(path)
        except OSError:
            pass

    sources: list[Path] = []
    for name in DOC_NAMES:
        _add(root / name, sources)
    for subdir in ("docs", "evals"):
        base = root / subdir
        if not base.exists():
            continue
        for name in DOC_NAMES:
            _add(base / name, sources)
    return sources


def _commands(root: Path) -> list[str]:
    commands: list[str] = []
    package = root / "package.json"
    if package.exists():
        try:
            scripts = json.loads(_read_head(package)).get("scripts", {})
            commands.extend(f"npm run {name}" for name in sorted(scripts))
        except Exception:
            pass
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(_read_head(pyproject))
            commands.extend(sorted(data.get("project", {}).get("scripts", {}).keys()))
        except Exception:
            pass
    readme = root / "README.md"
    if readme.exists():
        for match in re.finditer(r"^\s*(uv run [^\n]+|pytest[^\n]*|npm run [^\n]+|python -m [^\n]+)", _read_head(readme), re.M):
            commands.append(match.group(1).strip())
    return _unique(commands, limit=12)


def _follow_ups(sources: list[Path], limit: int = 12) -> list[str]:
    patterns = [
        re.compile(r"^\s*(?:[-*]\s*)?\[\s\]\s+(.{3,220})$"),
        re.compile(r"^\s*(?:TODO|FIXME|ACTION|FOLLOW[ -]?UP)\s*:?\s+(.{3,220})$", re.IGNORECASE),
    ]
    items: list[str] = []
    for source in sources:
        for line in _read_head(source).splitlines():
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    items.append(f"{match.group(1).strip()} ({source})")
                    break
            if len(items) >= limit:
                return _unique(items, limit=limit)
    return _unique(items, limit=limit)


def _excerpt(source: Path, limit: int = 1200) -> str:
    text = _read_head(source)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.startswith("#") or stripped.startswith("- ") or len(stripped) > 25:
            lines.append(stripped)
        if len("\n".join(lines)) >= limit:
            break
    excerpt = "\n".join(lines).strip()
    if len(excerpt) > limit:
        excerpt = excerpt[: limit - 3].rsplit(" ", 1)[0] + "..."
    return excerpt


def _unique(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "project"
