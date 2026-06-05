from secondbrain.project_context import (
    ProjectContext,
    collect_project_context,
    discover_project_roots,
    render_project_context,
    render_active_projects_index,
    write_project_context_notes,
)


def test_discover_project_roots_from_configured_targets(tmp_path):
    project = tmp_path / "agent"
    docs = project / "docs"
    docs.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        """
[project]
name = "agent"
description = "Local agent project"
""".strip()
        + "\n"
    )

    roots = discover_project_roots(targets=[docs])

    assert roots == [project]


def test_collect_project_context_extracts_sources_commands_and_followups(tmp_path, monkeypatch):
    import secondbrain.project_context as pc

    project = tmp_path / "agent"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "agent"
description = "Local agent project"

[project.scripts]
agent = "agent.cli:app"
""".strip()
        + "\n"
    )
    (project / "README.md").write_text(
        """
# Agent

Local agent project.

uv run pytest

TODO: wire nightly report
- [ ] add dashboard
""".strip()
        + "\n"
    )

    monkeypatch.setattr(pc, "configured_targets", lambda: [project])

    contexts = collect_project_context(limit=1)

    assert contexts[0].name == "agent"
    assert contexts[0].description == "Local agent project"
    assert "agent" in contexts[0].commands
    assert "uv run pytest" in contexts[0].commands
    assert any("wire nightly report" in item for item in contexts[0].follow_ups)


def test_write_project_context_notes_renders_source_backed_note(tmp_path, monkeypatch):
    import secondbrain.project_context as pc

    project = tmp_path / "agent"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "agent"
description = "Local agent project"

[project.scripts]
agent = "agent.cli:app"
""".strip()
        + "\n"
    )
    (project / "README.md").write_text(
        """
# Agent

Local agent project.

uv run pytest

TODO: wire nightly report
- [ ] add dashboard
""".strip()
        + "\n"
    )
    monkeypatch.setattr(pc, "discover_project_roots", lambda limit=20, **_: [project])

    written = write_project_context_notes(output_dir=tmp_path / "out")
    text = (tmp_path / "out" / "agent.md").read_text()

    assert [path.name for path in written] == ["active-projects.md", "agent.md"]
    assert "Description: Local agent project" in text
    assert "`agent`" in text
    assert "`uv run pytest`" in text
    assert "wire nightly report" in text
    assert "add dashboard" in text


def test_render_project_context_handles_empty_context(tmp_path):
    project = tmp_path / "empty"
    project.mkdir()
    context = ProjectContext(
        name="empty",
        path=project,
        description="",
        sources=[],
        commands=[],
        follow_ups=[],
    )

    markdown = render_project_context(context)

    assert "No README/doc source files found." in markdown
    assert "No explicit project commands found." in markdown


def test_render_active_projects_index_lists_all_contexts(tmp_path):
    context = ProjectContext(
        name="agent",
        path=tmp_path / "agent",
        description="Local agent project",
        sources=[tmp_path / "agent" / "README.md"],
        commands=[],
        follow_ups=["wire nightly report"],
    )

    markdown = render_active_projects_index([context])

    assert "# Active Projects Index" in markdown
    assert "agent: Local agent project" in markdown
    assert "agent: wire nightly report" in markdown


def test_collect_project_context_completes_quickly_on_temp_tree(tmp_path, monkeypatch):
    import time

    import secondbrain.project_context as pc

    parent = tmp_path / "projects"
    parent.mkdir()
    for i in range(6):
        project = parent / f"proj{i}"
        (project / "docs").mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            f'[project]\nname = "proj{i}"\ndescription = "Project number {i}"\n'
        )
        (project / "README.md").write_text(
            f"# proj{i}\n\nProject number {i} does useful things.\n\n- [ ] finish proj{i}\n"
        )
    monkeypatch.setattr(pc, "configured_targets", lambda: [parent])

    start = time.monotonic()
    contexts = collect_project_context(limit=10)
    elapsed = time.monotonic() - start

    assert elapsed < 5  # bounded; a healthy temp-tree run is near-instant
    names = {context.name for context in contexts}
    assert {f"proj{i}" for i in range(6)} <= names
    assert any("finish proj0" in item for context in contexts for item in context.follow_ups)


def test_collect_project_context_is_bounded_when_filesystem_stalls(tmp_path, monkeypatch):
    """A stalled filesystem op on one target must not blow the wall-clock budget:
    the run abandons it and still completes (the launchd nightly never hangs)."""
    import time

    import secondbrain.project_context as pc

    project = tmp_path / "agent"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "agent"\ndescription = "Agent"\n')
    monkeypatch.setattr(pc, "configured_targets", lambda: [project])

    def _stall(*_args, **_kwargs):
        time.sleep(30)  # simulate a hung SMB stat/iterdir

    monkeypatch.setattr(pc, "_target_candidates", _stall)

    start = time.monotonic()
    contexts = collect_project_context(limit=5, budget=0.5)
    elapsed = time.monotonic() - start

    assert elapsed < 5  # abandoned after ~budget, not after 30s
    assert contexts == []  # nothing discovered, but the run returned cleanly
