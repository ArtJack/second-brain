"""Only an explicit marker may become a durable task.

`extract_tasks` used to treat "we need to install Node first" in a downloaded
README the same as "TODO: renew the certificate" in the owner's own notes, and
`task-sync` wrote both into the SQLite ledger that the assistant plans from. The
prose reading is still worth showing in the overnight report — it is a hint —
but a hint is not a commitment, and the ledger is the one place that must hold
only things the owner actually wrote down as work.

The separation is enforced at two levels: the extractor returns the two kinds
separately, and the report renders them under different headings, because
`task-sync` reads the report's "Possible tasks:" section.
"""
from __future__ import annotations

import json

from secondbrain.overnight import ensure_config, extract_mentions, extract_tasks, run_overnight

TEXT = """
TODO: renew the server certificate
- [ ] Review the new lease document
- [x] Already finished
We should follow up with Alex about the Qdrant key.
You need to install Node 20 before running this.
"""


def test_explicit_markers_are_tasks():
    tasks = extract_tasks(TEXT)

    assert "renew the server certificate" in tasks
    assert "Review the new lease document" in tasks
    assert "Already finished" not in tasks


def test_prose_is_a_mention_not_a_task():
    tasks = extract_tasks(TEXT)
    mentions = extract_mentions(TEXT)

    assert not any("install Node 20" in task for task in tasks)
    assert not any("follow up with Alex" in task for task in tasks)
    assert any("install Node 20" in mention for mention in mentions)
    assert any("follow up with Alex" in mention for mention in mentions)


def test_report_keeps_mentions_out_of_the_section_task_sync_reads(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "readme.md").write_text(TEXT)

    root = tmp_path / "overnight"
    config = ensure_config(root / "config.json")
    config.update({"targets": [str(source)], "max_files_per_run": 10})
    (root / "config.json").write_text(json.dumps(config))

    run_overnight(root=root, dry_run=True)
    report = next((root / "reports").glob("*.md")).read_text()

    tasks_section = report.split("Possible tasks:")[1].split("\n\n")[0]
    assert "renew the server certificate" in tasks_section
    assert "install Node 20" not in tasks_section
    assert "Mentions (not tasks):" in report
    assert "install Node 20" in report.split("Mentions (not tasks):")[1]
