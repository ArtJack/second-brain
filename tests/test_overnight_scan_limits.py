"""The scan's limits must bound work, not silently shrink the corpus.

Two defects motivate this file.

`max_files_per_run` used to stop the *walk*: once the candidate list reached the
cap, discovery returned early, in filesystem order, before change detection. Any
file past that point was never examined on any night, and the report said
"Scanned files: 250" forever, so the loss was invisible. The cap belongs on the
work a run does (files it ingests), not on the set it is allowed to look at.

Exclusions used to be directory names only, so a credential-shaped file sitting
in a scanned tree was ingestible — and with a remote store its text leaves the
machine. Excluding it is not enough on its own: a scan that drops a file has to
say so, or the next person debugging "why is this note not in the brain?" has
nothing to read.
"""
from __future__ import annotations

import json

from secondbrain.overnight import ensure_config, run_overnight, scan_targets


def _config(target, **overrides):
    config = {
        "targets": [str(target)],
        "exclude_dirs": [".git", ".claude"],
        "max_file_mb": 1,
        "max_files_per_run": 250,
    }
    config.update(overrides)
    return config


def test_scan_sees_every_file_even_past_the_run_cap(tmp_path):
    for i in range(5):
        (tmp_path / f"note-{i}.md").write_text(f"note {i}")

    result = scan_targets(_config(tmp_path, max_files_per_run=2))

    assert [path.name for path in result.files] == [
        "note-0.md",
        "note-1.md",
        "note-2.md",
        "note-3.md",
        "note-4.md",
    ]


def test_run_defers_changed_files_past_the_cap_and_reports_them(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i in range(3):
        (source / f"note-{i}.md").write_text(f"note {i}")

    root = tmp_path / "overnight"
    config = ensure_config(root / "config.json")
    config.update(_config(source, max_files_per_run=2))
    (root / "config.json").write_text(json.dumps(config))

    res = run_overnight(root=root, dry_run=True)

    assert res["stats"]["scanned"] == 3
    assert res["stats"]["changed"] == 2
    assert res["stats"]["deferred"] == 1
    assert "- Deferred to a later run: 1" in next((root / "reports").glob("*.md")).read_text()


def test_credential_shaped_files_are_excluded_and_named_in_the_report(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.md").write_text("a real note")
    (source / ".gsc-credentials.json").write_text('{"private_key": "x"}')
    (source / "package-lock.json").write_text("{}")
    (source / ".mcp.json").write_text("{}")
    # Not a supported suffix, so it never reaches the rule check — recorded here
    # so the two mechanisms (suffix filter, deny-list) stay distinguishable.
    (source / "uv.lock").write_text("")

    result = scan_targets(_config(source))

    assert [path.name for path in result.files] == ["keep.md"]
    assert {item["path"].rsplit("/", 1)[-1]: item["rule"] for item in result.skipped} == {
        ".gsc-credentials.json": "*credential*",
        "package-lock.json": "package-lock.json",
        ".mcp.json": ".mcp.json",
    }


def test_agent_working_directories_are_not_scanned(tmp_path):
    source = tmp_path / "source"
    (source / ".claude" / "worktrees" / "qa").mkdir(parents=True)
    (source / ".claude" / "worktrees" / "qa" / "copy.md").write_text("a duplicate of the repo")
    (source / "note.md").write_text("the real one")

    result = scan_targets(_config(source))

    assert [path.name for path in result.files] == ["note.md"]


def test_a_skipped_file_is_counted_in_the_run_stats(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "note.md").write_text("keep")
    (source / "api-tokens.json").write_text('{"OPENAI_API_KEY": "sk-not-real"}')

    root = tmp_path / "overnight"
    config = ensure_config(root / "config.json")
    config.update(_config(source))
    (root / "config.json").write_text(json.dumps(config))

    res = run_overnight(root=root, dry_run=True)

    assert res["stats"]["scanned"] == 1
    assert res["stats"]["skipped"] == 1
    report = next((root / "reports").glob("*.md")).read_text()
    assert "api-tokens.json" in report
    assert "sk-not-real" not in report


class TestBenchmarkFixturesAreNotMemories:
    """Invented documents must never enter the corpus the owner asks questions of.

    The evaluation corpora are fiction. `evals/corpus-hard/runbook-gateway.md`
    says "the gateway's budget alert fires at 80 percent of the monthly cap" and
    `backup-offsite.md` says restic runs at 03:30 — numbers written to be *found*
    by a benchmark, not because anyone measured them. They are shaped exactly
    like the owner's real notes, because that is what makes them useful fixtures.

    They were being ingested into the real collection. Asked "where does the
    gateway run", the brain cited `evals/corpus-hard/runbook-gateway.md` as a
    source, and a reader has no way to tell that document from a note the owner
    wrote. For a system whose whole promise is "no source, no claim", a
    fabricated source is the worst possible thing to hold.
    """

    def test_a_benchmark_corpus_directory_is_not_scanned(self, tmp_path):
        from secondbrain import overnight

        real = tmp_path / "notes.md"
        real.write_text("a real note")
        for name in ("corpus", "corpus-hard"):
            fixture = tmp_path / "evals" / name / "invented.md"
            fixture.parent.mkdir(parents=True)
            fixture.write_text("# Gateway runbook\n\nThe budget alert fires at 80 percent.")

        result = overnight.scan_targets({"targets": [str(tmp_path)]})

        names = [p.name for p in result.files]
        assert "notes.md" in names
        assert "invented.md" not in names, "a fabricated document reached the real corpus"

    def test_the_default_config_carries_the_rule(self):
        """A fresh install must not have to learn this the same way."""
        from secondbrain.overnight import DEFAULT_CONFIG

        assert "corpus" in DEFAULT_CONFIG["exclude_dirs"]
        assert "corpus-hard" in DEFAULT_CONFIG["exclude_dirs"]
