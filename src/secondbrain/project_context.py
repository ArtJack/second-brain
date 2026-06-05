"""Generate source-backed project context notes for the second brain."""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .overnight import overnight_paths

DEFAULT_OUTPUT_DIR = Path("/Volumes/DISK/AI/artjeck/inbox/project-context")
PROJECT_MARKERS = ("pyproject.toml", "package.json", ".git")
README_NAMES = ("README.md", "readme.md", "README.txt")
DOC_NAMES = ("README.md", "IFTA_RUNBOOK.md", "knowledge_base.md")


@dataclass(frozen=True)
class ProjectContext:
    name: str
    path: Path
    description: str
    sources: list[Path]
    commands: list[str]
    follow_ups: list[str]


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
    current = path.expanduser().resolve()
    for candidate in [current, *current.parents]:
        if candidate == candidate.parent:
            break
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return None


def discover_project_roots(targets: list[Path] | None = None, limit: int = 30) -> list[Path]:
    roots = targets or configured_targets()
    candidates: list[Path] = []
    for raw in roots:
        base = raw.parent if raw.is_file() else raw
        if not base.exists():
            continue
        root = find_project_root(base)
        if root:
            candidates.append(root)
            continue
        candidates.append(base)
        try:
            candidates.extend(find_project_root(path) or path for path in base.iterdir() if path.is_dir())
        except OSError:
            pass

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if describe_project(resolved):
            out.append(resolved)
        if len(out) >= limit:
            break
    return out


def describe_project(path: Path) -> str:
    pyproject = path / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text())
            description = data.get("project", {}).get("description")
            if description:
                return str(description)
        except Exception:
            pass
    package = path / "package.json"
    if package.exists():
        try:
            description = json.loads(package.read_text()).get("description")
            if description:
                return str(description)
        except Exception:
            pass
    readme = next((path / name for name in README_NAMES if (path / name).exists()), None)
    if not readme:
        return ""
    lines = [line.strip() for line in readme.read_text(errors="ignore").splitlines()]
    heading = next((line.lstrip("# ").strip() for line in lines if line.startswith("#")), "")
    paragraph = next((line for line in lines if line and not line.startswith("#") and not line.startswith("```")), "")
    return paragraph or heading


def project_inventory(limit: int = 12) -> list[dict[str, str]]:
    projects = []
    for root in discover_project_roots(limit=limit):
        projects.append({"name": root.name, "path": str(root), "description": describe_project(root)})
    return projects


def collect_project_context(limit: int = 20) -> list[ProjectContext]:
    contexts = []
    for root in discover_project_roots(limit=limit):
        sources = _context_sources(root)
        contexts.append(
            ProjectContext(
                name=root.name,
                path=root,
                description=describe_project(root),
                sources=sources,
                commands=_commands(root),
                follow_ups=_follow_ups(sources),
            )
        )
    return contexts


def write_project_context_notes(output_dir: Path = DEFAULT_OUTPUT_DIR, limit: int = 20) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    contexts = collect_project_context(limit=limit)
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
        excerpt = _excerpt(source)
        if excerpt:
            lines.extend([f"### {source.name}", "", excerpt, ""])
    return "\n".join(lines).rstrip() + "\n"


def _context_sources(root: Path) -> list[Path]:
    sources: list[Path] = []
    for name in DOC_NAMES:
        path = root / name
        if path.exists() and path.is_file():
            sources.append(path)
    for subdir in ("docs", "evals"):
        base = root / subdir
        if not base.exists():
            continue
        for name in DOC_NAMES:
            path = base / name
            if path.exists() and path.is_file():
                sources.append(path)
    return sources


def _commands(root: Path) -> list[str]:
    commands: list[str] = []
    package = root / "package.json"
    if package.exists():
        try:
            scripts = json.loads(package.read_text()).get("scripts", {})
            commands.extend(f"npm run {name}" for name in sorted(scripts))
        except Exception:
            pass
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text())
            commands.extend(sorted(data.get("project", {}).get("scripts", {}).keys()))
        except Exception:
            pass
    readme = root / "README.md"
    if readme.exists():
        for match in re.finditer(r"^\s*(uv run [^\n]+|pytest[^\n]*|npm run [^\n]+|python -m [^\n]+)", readme.read_text(errors="ignore"), re.M):
            commands.append(match.group(1).strip())
    return _unique(commands, limit=12)


def _follow_ups(sources: list[Path], limit: int = 12) -> list[str]:
    patterns = [
        re.compile(r"^\s*(?:[-*]\s*)?\[\s\]\s+(.{3,220})$"),
        re.compile(r"^\s*(?:TODO|FIXME|ACTION|FOLLOW[ -]?UP)\s*:?\s+(.{3,220})$", re.IGNORECASE),
    ]
    items: list[str] = []
    for source in sources:
        for line in source.read_text(errors="ignore").splitlines():
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    items.append(f"{match.group(1).strip()} ({source})")
                    break
            if len(items) >= limit:
                return _unique(items, limit=limit)
    return _unique(items, limit=limit)


def _excerpt(source: Path, limit: int = 1200) -> str:
    text = source.read_text(errors="ignore")
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
