"""The migration script must remove a source from both copies of the corpus.

Verdict SB-F-25. The store and the keyword index hold the same chunks twice, and
this script deleted from one of them. Keyword search went on returning the
syllabus material the script exists to move out, under a `second_brain`
citation, for a document that now lives in `second_brain_reference`.

`gc` and `ingest` both already mirror their deletes. This was the third place
and the one nobody looked at, because it runs once.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "split_reference_corpus.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("split_reference_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["split_reference_corpus"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("split_reference_corpus", None)


class FakeStore:
    def __init__(self, collection=None, chunks=7):
        self.collection = collection
        self.deleted: list[str] = []
        self._chunks = chunks

    def delete_source(self, source):
        self.deleted.append(source)

    def count(self):
        return self._chunks


def test_a_moved_source_is_deleted_from_the_keyword_index_too(script, tmp_path, monkeypatch):
    stores: dict[str, FakeStore] = {}

    def make_store(collection=None, **kw):
        stores.setdefault(collection, FakeStore(collection))
        return stores[collection]

    index_deletes: list[tuple[str, str]] = []

    class FakeIndex:
        def delete_source(self, collection, source):
            index_deletes.append((collection, source))

    monkeypatch.setattr(script, "Store", make_store)
    monkeypatch.setattr(script, "KeywordIndex", lambda *a, **kw: FakeIndex())
    monkeypatch.setattr(script, "plan", lambda root: {"sources": ["/ref/a.pdf", "/ref/b.pdf"], "chunks": 9})
    monkeypatch.setattr(script, "ingest_paths", lambda root, collection=None: iter([(Path("/ref/a.pdf"), 5)]))

    exit_code = script.main([str(tmp_path), "--apply"])

    assert exit_code == 0
    assert [source for _collection, source in index_deletes] == ["/ref/a.pdf", "/ref/b.pdf"]
    assert stores[script.cfg.collection].deleted == ["/ref/a.pdf", "/ref/b.pdf"]


def test_nothing_is_deleted_from_either_copy_when_the_ingest_wrote_nothing(script, tmp_path, monkeypatch):
    """The script's stated safety property, asserted for the index as well."""
    stores: dict[str, FakeStore] = {}
    index_deletes: list[str] = []

    class FakeIndex:
        def delete_source(self, collection, source):
            index_deletes.append(source)

    monkeypatch.setattr(script, "Store", lambda collection=None, **kw: stores.setdefault(collection, FakeStore(collection)))
    monkeypatch.setattr(script, "KeywordIndex", lambda *a, **kw: FakeIndex())
    monkeypatch.setattr(script, "plan", lambda root: {"sources": ["/ref/a.pdf"], "chunks": 3})
    monkeypatch.setattr(script, "ingest_paths", lambda root, collection=None: iter([]))

    assert script.main([str(tmp_path), "--apply"]) == 1
    assert index_deletes == []
    assert all(store.deleted == [] for store in stores.values())
