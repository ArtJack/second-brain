import json
from pathlib import Path

from secondbrain.overnight import (
    OvernightState,
    discover_supported_files,
    ensure_config,
    extract_tasks,
    run_overnight,
)


def test_extract_tasks_finds_common_markers():
    text = """
    TODO: renew the server certificate
    - [ ] Review the new lease document
    - [x] Already finished
    We should follow up with Alex about the Qdrant key.
    """

    tasks = extract_tasks(text)

    assert "renew the server certificate" in tasks
    assert "Review the new lease document" in tasks
    assert "follow up with Alex about the Qdrant key" in tasks
    assert "Already finished" not in tasks


def test_discover_supported_files_respects_limits_and_skips(tmp_path):
    (tmp_path / "note.md").write_text("hello")
    (tmp_path / "image.png").write_text("no")
    skipped = tmp_path / "node_modules"
    skipped.mkdir()
    (skipped / "dep.md").write_text("skip")
    config = {
        "targets": [str(tmp_path)],
        "exclude_dirs": ["node_modules"],
        "max_file_mb": 1,
        "max_files_per_run": 10,
    }

    files = discover_supported_files(config)

    assert files == [tmp_path / "note.md"]


def test_state_detects_and_remembers_changed_files(tmp_path):
    state = OvernightState(tmp_path / "state.sqlite3")
    note = tmp_path / "note.md"
    note.write_text("one")
    stat = note.stat()

    assert state.file_changed(note, "abc", stat.st_size, stat.st_mtime)
    state.remember_file(note, "abc", stat.st_size, stat.st_mtime, chunks=1)
    assert not state.file_changed(note, "abc", stat.st_size, stat.st_mtime)
    assert state.file_changed(note, "def", stat.st_size, stat.st_mtime)


def test_run_overnight_dry_run_writes_report_without_remembering_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    note = source / "plan.md"
    note.write_text("TODO: test the overnight worker\nUseful project notes.")

    root = tmp_path / "overnight"
    config = ensure_config(root / "config.json")
    config.update({"targets": [str(source)], "max_files_per_run": 10})
    (root / "config.json").write_text(json.dumps(config))

    res = run_overnight(root=root, dry_run=True)
    report = Path(res["report"]).read_text()

    assert res["stats"]["scanned"] == 1
    assert res["stats"]["changed"] == 1
    assert res["stats"]["ingested"] == 0
    assert "test the overnight worker" in report

    second = run_overnight(root=root, dry_run=True)
    assert second["stats"]["changed"] == 1


def test_run_overnight_does_not_extract_tasks_from_code_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    code = source / "test_fixture.py"
    code.write_text('assert "TODO: not a real task" in text\n')

    root = tmp_path / "overnight"
    config = ensure_config(root / "config.json")
    config.update({"targets": [str(source)], "max_files_per_run": 10})
    (root / "config.json").write_text(json.dumps(config))

    res = run_overnight(root=root, dry_run=True)
    report = Path(res["report"]).read_text()

    assert res["stats"]["changed"] == 1
    assert "Possible tasks:" not in report
