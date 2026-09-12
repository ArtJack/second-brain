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

    The evaluation corpora are fiction. Each fixture states specific, invented
    numbers — a budget-alert threshold, a backup time, a key generation — written
    to be *found* by a benchmark, not because anyone measured them. They are
    shaped exactly like the owner's real notes, because that is what makes them
    useful fixtures. None of those numbers is repeated here: this test file is
    ingested into the brain too, and quoting them is how they leaked back in.

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
            fixture.write_text("# Fixture\n\nInvented content for an exclusion test.")

        result = overnight.scan_targets({"targets": [str(tmp_path)]})

        names = [p.name for p in result.files]
        assert "notes.md" in names
        assert "invented.md" not in names, "a fabricated document reached the real corpus"

    def test_the_default_config_carries_the_rule(self):
        """A fresh install must not have to learn this the same way.

        Anchored, not bare: see TestExclusionRulesAreAnchored below for why the
        first version of this rule was itself a defect.
        """
        from secondbrain.overnight import DEFAULT_CONFIG

        assert "evals/corpus" in DEFAULT_CONFIG["exclude_dirs"]
        assert "evals/corpus-hard" in DEFAULT_CONFIG["exclude_dirs"]


class TestExclusionRulesAreAnchored:
    """SB-F-44: a bare directory name matches at any depth, anywhere.

    I added `corpus` and `corpus-hard` as bare names to keep the benchmark
    fixtures out. This codebase already records that exact mistake, in a comment
    about a different generic word: `"data" is deliberately NOT here: it used to
    be, and since the match is per path segment it skipped every folder named
    data anywhere in the world — including the owner's own notes.` I repeated it
    a few hours later, with a word just as generic.

    It is not hypothetical. `~/Projects/verdict/eval/corpus` is another
    project's directory under a live scan target. Under the retroactive rule
    enforcement added the same night, `sb gc --enforce-rules` would not merely
    skip it in future — it would delete what is already indexed from it.

    A rule containing a slash matches consecutive path segments, so the fixture
    directories can be named exactly. Bare names keep working for the cases
    where the name really is unambiguous, like `.venv`.
    """

    def _config(self, targets, dirs):
        return {"targets": [str(t) for t in targets], "exclude_dirs": dirs, "exclude_globs": []}

    def test_an_anchored_rule_excludes_only_the_directory_it_names(self, tmp_path):
        from secondbrain import overnight

        fixture = tmp_path / "evals" / "corpus" / "invented.md"
        elsewhere = tmp_path / "linguistics" / "corpus" / "fieldnotes.md"
        for path in (fixture, elsewhere):
            path.parent.mkdir(parents=True)
            path.write_text("text")

        result = overnight.scan_targets(self._config([tmp_path], ["evals/corpus"]))

        names = [str(p) for p in result.files]
        assert any("fieldnotes.md" in n for n in names), "an unrelated corpus directory must survive"
        assert not any("invented.md" in n for n in names)

    def test_a_bare_name_still_matches_at_any_depth(self, tmp_path):
        """Unchanged for the names where that is what you want."""
        from secondbrain import overnight

        buried = tmp_path / "a" / "node_modules" / "left-pad" / "readme.md"
        buried.parent.mkdir(parents=True)
        buried.write_text("x")
        (tmp_path / "keep.md").write_text("x")

        result = overnight.scan_targets(self._config([tmp_path], ["node_modules"]))

        assert [p.name for p in result.files] == ["keep.md"]

    def test_the_shipped_rules_name_the_fixture_directories_by_path(self):
        from secondbrain.overnight import DEFAULT_CONFIG

        assert "evals/corpus" in DEFAULT_CONFIG["exclude_dirs"]
        assert "evals/corpus-hard" in DEFAULT_CONFIG["exclude_dirs"]
        assert "corpus" not in DEFAULT_CONFIG["exclude_dirs"], "too generic to match at any depth"
        assert "corpus-hard" not in DEFAULT_CONFIG["exclude_dirs"]


class TestDirectIngestHonoursTheFixtureExclusion:
    """SB-F-43: the exclusion lived only in the nightly scan.

    `sb ingest ~/Projects/second-brain` walks with `ingest.discover()`, which has
    its own independent skip list and never sees the scan's rules. That is the
    command the owner would use to re-ingest the project, and it reopened
    exactly the defect the exclusion was written to close.
    """

    def test_discover_skips_the_benchmark_fixtures(self, tmp_path):
        from secondbrain.ingest import discover

        for rel in ("evals/corpus/old.md", "evals/corpus-hard/invented.md", "docs/real.md"):
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("text")

        found = [p.name for p in discover(tmp_path)]

        assert found == ["real.md"]

    def test_an_unrelated_corpus_directory_is_still_ingested(self, tmp_path):
        from secondbrain.ingest import discover

        path = tmp_path / "linguistics" / "corpus" / "fieldnotes.md"
        path.parent.mkdir(parents=True)
        path.write_text("real research")

        assert [p.name for p in discover(tmp_path)] == ["fieldnotes.md"]

    def test_pointing_straight_at_a_fixture_directory_still_ingests_it(self, tmp_path):
        """Asking for it by name is not the same as sweeping it up.

        The rule exists so `sb ingest ~/Projects/second-brain` does not walk the
        fabricated documents into the owner's corpus. It must not also stop the
        benchmark building its own throwaway collection — `sb eval
        --ingest-corpus` points at `evals/corpus-hard` directly, which is an
        explicit request for exactly that directory.

        Getting this wrong is not theoretical: the first version of the
        exclusion silently reduced `sb eval --ingest-corpus` to a no-op, and the
        hard benchmark went on reporting 14/18 because the collection still held
        the previous run's chunks. On a reset collection it scored 0/18.
        """
        from secondbrain.ingest import discover

        fixture = tmp_path / "evals" / "corpus-hard" / "invented.md"
        fixture.parent.mkdir(parents=True)
        fixture.write_text("# Fixture\n\nInvented content for an exclusion test.")

        assert [p.name for p in discover(fixture.parent)] == ["invented.md"]
        assert [p.name for p in discover(tmp_path)] == [], "but sweeping the tree must not take it"

    def test_a_file_named_directly_inside_a_fixture_directory_is_ingested(self, tmp_path):
        from secondbrain.ingest import discover

        fixture = tmp_path / "evals" / "corpus" / "one.md"
        fixture.parent.mkdir(parents=True)
        fixture.write_text("x")

        assert [p.name for p in discover(fixture)] == ["one.md"]


class TestTheBenchmarkDefinitionsAreFixturesToo:
    """SB-F-45: the JSON files carry the same invented claims as the corpus.

    Excluding `evals/corpus-hard/` kept the fabricated runbook out. It did not
    keep out `evals/hard.json`, which stores each invented claim *directly
    beside a verbatim copy of the question it answers*. As retrieval material
    that is worse than the prose, not better.

    I argued last night that they "read as configuration rather than as notes",
    and that is a claim about a human squinting at a citation, not about what
    retrieval does with the text.
    """

    def test_the_benchmark_definitions_are_not_swept_up(self, tmp_path):
        from secondbrain.ingest import discover

        evals = tmp_path / "evals"
        evals.mkdir()
        for name in ("hard.json", "regression.json", "retrieval.json"):
            (evals / name).write_text('{"cases": []}')
        (tmp_path / "real.md").write_text("a real note")

        assert [p.name for p in discover(tmp_path)] == ["real.md"]

    def test_an_unrelated_hard_json_elsewhere_is_untouched(self, tmp_path):
        """Anchored, like every other rule here — `hard.json` is a generic name."""
        from secondbrain.ingest import discover

        other = tmp_path / "src" / "hard.json"
        other.parent.mkdir(parents=True)
        other.write_text("{}")

        assert [p.name for p in discover(tmp_path)] == ["hard.json"]

    def test_pointing_at_one_directly_still_reads_it(self, tmp_path):
        from secondbrain.ingest import discover

        bench = tmp_path / "evals" / "hard.json"
        bench.parent.mkdir(parents=True)
        bench.write_text("{}")

        assert [p.name for p in discover(bench)] == ["hard.json"]
