"""A scan target can name the collection it belongs to.

Task 4 split the reference material — syllabi, standards, study kits the owner
keeps but did not write — into its own collection, so a question about the
owner's work could not be answered out of someone else's textbook. The overnight
worker knew nothing about that split: it ingested everything it scanned into the
default collection.

So the migration had to be protected by adding `istqb` to `exclude_dirs`.
Without it the next nightly would have walked 2,954 chunks of syllabus PDFs
straight back into the personal corpus and undone the split in one run. That
exclusion bought safety by giving the reference corpus no automatic refresh at
all: nothing updates it, ever, until someone remembers.

A target may now be `{"path": ..., "collection": ...}` as well as a bare string.
String targets are the common case and the live config is hand-edited, so they
keep working untouched.
"""
from __future__ import annotations

from pathlib import Path

from secondbrain import overnight


def _config(tmp_path, targets, **extra):
    return {"targets": targets, "exclude_dirs": [], "exclude_globs": [], "max_file_mb": 10, **extra}


class TestDiscoverTargets:
    def test_a_string_target_carries_no_collection(self, tmp_path):
        (tmp_path / "note.md").write_text("x")

        found = overnight.discover_targets(_config(tmp_path, [str(tmp_path)]))

        assert found == [(tmp_path.resolve(), None)]

    def test_an_object_target_carries_its_collection(self, tmp_path):
        found = overnight.discover_targets(
            _config(tmp_path, [{"path": str(tmp_path), "collection": "second_brain_reference"}])
        )

        assert found == [(tmp_path.resolve(), "second_brain_reference")]

    def test_a_mixed_config_keeps_each_target_s_own_answer(self, tmp_path):
        personal = tmp_path / "personal"
        reference = tmp_path / "reference"
        for d in (personal, reference):
            d.mkdir()

        found = dict(
            overnight.discover_targets(
                _config(
                    tmp_path,
                    [str(personal), {"path": str(reference), "collection": "ref"}],
                )
            )
        )

        assert found[personal.resolve()] is None
        assert found[reference.resolve()] == "ref"

    def test_an_object_with_no_collection_behaves_like_a_string(self, tmp_path):
        found = overnight.discover_targets(_config(tmp_path, [{"path": str(tmp_path)}]))

        assert found == [(tmp_path.resolve(), None)]

    def test_a_target_that_does_not_exist_is_skipped_in_either_form(self, tmp_path):
        found = overnight.discover_targets(
            _config(tmp_path, [str(tmp_path / "gone"), {"path": str(tmp_path / "also-gone"), "collection": "x"}])
        )

        assert found == []

    def test_the_same_path_twice_is_discovered_once(self, tmp_path):
        found = overnight.discover_targets(_config(tmp_path, [str(tmp_path), str(tmp_path)]))

        assert len(found) == 1


class TestScanCarriesTheCollectionThrough:
    def test_each_file_records_the_collection_of_the_target_it_came_from(self, tmp_path):
        personal = tmp_path / "personal"
        reference = tmp_path / "reference"
        for d in (personal, reference):
            d.mkdir()
        (personal / "mine.md").write_text("my own note")
        (reference / "syllabus.md").write_text("someone else's textbook")

        result = overnight.scan_targets(
            _config(tmp_path, [str(personal), {"path": str(reference), "collection": "ref"}])
        )

        by_name = {p.name: result.collection_for(p) for p in result.files}
        assert by_name == {"mine.md": None, "syllabus.md": "ref"}

    def test_a_file_target_works_as_well_as_a_directory(self, tmp_path):
        note = tmp_path / "knowledge_base.md"
        note.write_text("x")

        result = overnight.scan_targets(
            _config(tmp_path, [{"path": str(note), "collection": "ref"}])
        )

        assert [p.name for p in result.files] == ["knowledge_base.md"]
        assert result.collection_for(result.files[0]) == "ref"


class TestIngestIsRoutedThere:
    def test_a_file_from_a_routed_target_is_ingested_into_that_collection(self, tmp_path, monkeypatch):
        seen: list[tuple[str, str | None]] = []

        def fake_ingest(path, reset=False, collection=None):
            seen.append((Path(path).name, collection))
            return iter([(Path(path), 1)])

        monkeypatch.setattr(overnight, "ingest_paths", fake_ingest)

        overnight._ingest_one(tmp_path / "syllabus.md", collection="ref")
        overnight._ingest_one(tmp_path / "mine.md", collection=None)

        assert seen == [("syllabus.md", "ref"), ("mine.md", None)]


class TestTheReportNamesANonDefaultCollection:
    """Silence means the default. A file that went elsewhere has to say where."""

    def _report_text(self, tmp_path, changed):
        paths = overnight.overnight_paths(tmp_path)
        paths.reports.mkdir(parents=True, exist_ok=True)
        written = overnight.write_report(
            paths=paths,
            run_id=1,
            started_at="2026-09-12T00:00:00+00:00",
            stats={"scanned": 1, "changed": 1, "ingested": 1, "failed": 0, "deferred": 0},
            changed_files=changed,
            failed_files=[],
            dry_run=False,
        )
        return Path(written).read_text()

    def _item(self, **over):
        item = {
            "path": "/x/syllabus.md",
            "size": 10,
            "chunks": 2,
            "tasks": [],
            "mentions": [],
            "collection": None,
            "snippet": "some text",
        }
        item.update(over)
        return item

    def test_a_routed_file_says_where_it_went(self, tmp_path):
        text = self._report_text(tmp_path, [self._item(collection="second_brain_reference")])

        assert "- Collection: second_brain_reference" in text

    def test_a_default_file_says_nothing_about_collections(self, tmp_path):
        text = self._report_text(tmp_path, [self._item()])

        assert "Collection:" not in text
