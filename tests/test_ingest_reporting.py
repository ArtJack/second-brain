"""Ingestion says what it did not ingest, and never corrupts what it did.

Two Verdict findings, both about silence.

SB-F-21: `SKIP_DIRS` contained the bare name "data" and was matched against every
path segment, so any folder called `data` anywhere in the world was skipped —
and a file that produced zero chunks was dropped with `continue`. Neither left a
trace, so "why is this note not in my brain?" had no answer anywhere.

SB-F-22: every non-PDF was read with `errors="ignore"`, which is not a fallback,
it is data loss with the evidence deleted. A Latin-1 note became "Caf Mller" and
a UTF-16 note became NUL-interleaved garbage — both then embedded, retrieved and
cited verbatim as if they were what the owner wrote.

The rule both findings share: a decision to drop or transform the owner's data is
reportable. Guessing quietly is the defect.
"""
from __future__ import annotations

import pytest

from secondbrain.ingest import discover, ingest_paths, last_report, read_file


def test_a_folder_named_data_outside_the_repo_is_ingested(tmp_path):
    """The skip is for this repo's own data directory, not for the word "data"."""
    notes = tmp_path / "notes" / "data"
    notes.mkdir(parents=True)
    (notes / "budget.md").write_text("The 2026 budget is decided.")

    found = discover(tmp_path)

    assert [path.name for path in found] == ["budget.md"]


def test_the_repos_own_data_directory_is_still_skipped(tmp_path, monkeypatch):
    import secondbrain.ingest as ingest_module

    repo_data = tmp_path / "repo" / "data"
    repo_data.mkdir(parents=True)
    (repo_data / "chroma-notes.md").write_text("internal state, not a source")
    (tmp_path / "repo" / "real.md").write_text("a real note")
    monkeypatch.setattr(ingest_module.cfg, "state_db", repo_data / "artjeck.sqlite3")

    found = discover(tmp_path / "repo")

    assert [path.name for path in found] == ["real.md"]


def test_an_unsupported_suffix_is_reported_rather_than_vanishing(tmp_path):
    (tmp_path / "keep.md").write_text("a note")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")

    discover(tmp_path)
    report = last_report()

    assert [item["path"].rsplit("/", 1)[-1] for item in report["skipped"]] == ["photo.png"]
    assert report["skipped"][0]["reason"] == "unsupported-suffix"


def test_latin1_accents_survive_ingestion(tmp_path):
    note = tmp_path / "cafe.md"
    note.write_bytes("Café Müller naïve".encode("latin-1"))

    assert read_file(note) == "Café Müller naïve"


def test_utf16_text_survives_ingestion(tmp_path):
    note = tmp_path / "wide.md"
    note.write_bytes("hello world".encode("utf-16"))

    text = read_file(note)

    assert text.strip() == "hello world"
    assert "\x00" not in text


def test_an_undecodable_file_names_itself_and_what_was_tried(tmp_path):
    note = tmp_path / "broken.md"
    # Valid in no encoding the reader tries: lone surrogates in UTF-16 with an
    # odd byte count, and bytes that are not legal UTF-8 either.
    note.write_bytes(b"\xff\xfe\x00\xd8\x00\x00\x41")

    with pytest.raises(ValueError) as excinfo:
        read_file(note)

    assert "broken.md" in str(excinfo.value)
    assert "utf-8" in str(excinfo.value)


def test_a_zero_chunk_file_is_reported_and_the_run_continues(tmp_path, monkeypatch):
    import secondbrain.ingest as ingest_module

    (tmp_path / "empty.md").write_text("   \n\n  ")
    (tmp_path / "real.md").write_text("Something worth remembering.")

    class FakeStore:
        collection_name = "t"

        def reset(self):
            pass

        def delete_source(self, source):
            pass

        def upsert(self, **kw):
            pass

    monkeypatch.setattr(ingest_module, "Store", lambda **kw: FakeStore())
    monkeypatch.setattr(ingest_module, "embed", lambda chunks: [[0.1, 0.2]] * len(chunks))
    monkeypatch.setattr(ingest_module._index, "upsert_chunks", lambda *a, **k: None)
    monkeypatch.setattr(ingest_module._index, "delete_source", lambda *a, **k: None)

    ingested = [path.name for path, _ in ingest_paths(tmp_path)]
    report = last_report()

    assert ingested == ["real.md"]
    assert [item["path"].rsplit("/", 1)[-1] for item in report["skipped"]] == ["empty.md"]
    assert report["skipped"][0]["reason"] == "no-content"


def test_an_undecodable_file_does_not_stop_the_run(tmp_path, monkeypatch):
    import secondbrain.ingest as ingest_module

    (tmp_path / "broken.md").write_bytes(b"\xff\xfe\x00\xd8\x00\x00\x41")
    (tmp_path / "real.md").write_text("Something worth remembering.")

    class FakeStore:
        collection_name = "t"

        def reset(self):
            pass

        def delete_source(self, source):
            pass

        def upsert(self, **kw):
            pass

    monkeypatch.setattr(ingest_module, "Store", lambda **kw: FakeStore())
    monkeypatch.setattr(ingest_module, "embed", lambda chunks: [[0.1, 0.2]] * len(chunks))
    monkeypatch.setattr(ingest_module._index, "upsert_chunks", lambda *a, **k: None)
    monkeypatch.setattr(ingest_module._index, "delete_source", lambda *a, **k: None)

    ingested = [path.name for path, _ in ingest_paths(tmp_path)]
    report = last_report()

    assert ingested == ["real.md"]
    reasons = {item["path"].rsplit("/", 1)[-1]: item["reason"] for item in report["skipped"]}
    assert reasons["broken.md"].startswith("undecodable")


def test_the_report_belongs_to_the_latest_run_only(tmp_path, monkeypatch):
    """A stale report read as current would misattribute one run's losses to another."""

    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    discover(tmp_path)
    assert last_report()["skipped"]

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "note.md").write_text("fine")
    discover(clean)

    assert last_report()["skipped"] == []


def test_an_exclusion_rule_that_eats_everything_explains_itself(tmp_path, monkeypatch):
    """A misconfigured state-db path must not look like an empty folder."""
    import secondbrain.ingest as ingest_module

    (tmp_path / "note.md").write_text("a real note")
    # The state DB sitting in the ingest root makes the whole root "own data".
    monkeypatch.setattr(ingest_module.cfg, "state_db", tmp_path / "artjeck.sqlite3")

    found = discover(tmp_path)
    report = last_report()

    assert found == []
    assert report["skipped"], "silently finding nothing is the bug"
    assert "own-data-dir" in report["skipped"][0]["reason"]
    assert "nothing left to ingest" in report["skipped"][0]["reason"]


def test_a_normal_run_does_not_report_the_excluded_trees(tmp_path, monkeypatch):
    """The explanation is for the empty case only; otherwise it is noise."""
    import secondbrain.ingest as ingest_module

    (tmp_path / "note.md").write_text("a real note")
    junk = tmp_path / "node_modules"
    junk.mkdir()
    (junk / "dep.md").write_text("not mine")
    monkeypatch.setattr(ingest_module.cfg, "state_db", tmp_path / "elsewhere" / "artjeck.sqlite3")

    found = discover(tmp_path)

    assert [p.name for p in found] == ["note.md"]
    assert last_report()["skipped"] == []
