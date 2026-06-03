"""TaskStore: durable SQLite task state (exercised against a temp DB)."""
import pytest

from secondbrain.tasks import TaskStore


def test_add_list_complete_roundtrip(tmp_path):
    store = TaskStore(path=tmp_path / "tasks.sqlite3")

    task = store.add("file IFTA Q2")
    assert task["id"] == 1
    assert task["status"] == "open"
    assert task["completed_at"] is None

    assert len(store.list(status="open")) == 1

    done = store.complete(task["id"])
    assert done["status"] == "done"
    assert done["completed_at"]

    assert store.list(status="open") == []
    assert len(store.list(status="all")) == 1


def test_empty_title_is_rejected(tmp_path):
    store = TaskStore(path=tmp_path / "tasks.sqlite3")
    with pytest.raises(ValueError):
        store.add("   ")


def test_completing_unknown_task_raises(tmp_path):
    store = TaskStore(path=tmp_path / "tasks.sqlite3")
    with pytest.raises(KeyError):
        store.complete(999)
