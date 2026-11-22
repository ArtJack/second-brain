import importlib.util
from pathlib import Path


def _load_seed_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "seed_public_corpora.py"
    spec = importlib.util.spec_from_file_location("seed_public_corpora_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_collection_resets_and_ingests_relative_sources(tmp_path, monkeypatch):
    from secondbrain.keyword_index import KeywordIndex

    seed = _load_seed_module()
    # The reset drops a real keyword-index table; keep it off the index other tests share.
    monkeypatch.setattr(seed.ingest, "_index", KeywordIndex(tmp_path / "keyword-index.sqlite3"))

    calls: list[tuple] = []

    class FakeStore:
        def __init__(self, collection=None):
            calls.append(("store", collection))

        def reset(self):
            calls.append(("reset",))

        def count(self):
            calls.append(("count",))
            return 1

    def fake_ingest_paths(path, collection=None):
        calls.append(("ingest", path, collection))
        yield path, 1

    monkeypatch.setattr(seed, "Store", FakeStore)
    monkeypatch.setattr(seed, "_safe_repo_path", lambda relative: seed.REPO_ROOT / relative)
    monkeypatch.setattr(seed, "ingest_paths", fake_ingest_paths)

    result = seed.seed_collection("second_brain_public", ["README.md"])

    assert ("reset",) in calls
    assert ("ingest", Path("README.md"), "second_brain_public") in calls
    assert result == {
        "collection": "second_brain_public",
        "files": 1,
        "chunks": 1,
        "total": 1,
    }


def test_a_reseed_leaves_no_keyword_rows_from_a_file_dropped_from_the_corpus(tmp_path, monkeypatch):
    """SB-F-52's second site: the seed reset the store and left the keyword index.

    `ingest_paths` replaces a file's keyword rows only when it ingests that file
    again. The rows of a file dropped from CORPORA survived every reseed, so the
    public corpus's keyword search kept citing a document it no longer holds.

    The reset has to go through the index ingest writes with. A reset through a
    second `KeywordIndex` dropped the table behind the first one's cache of
    existing tables, and ingest's next write raised (SB-F-64).
    """
    from secondbrain.keyword_index import KeywordIndex

    seed = _load_seed_module()
    index = KeywordIndex(tmp_path / "keyword-index.sqlite3")
    index.upsert_chunks(
        "second_brain_public",
        [{"source": "docs/dropped.md", "chunk": 0, "name": "dropped.md", "document": "zephyrquill: a dropped file"}],
    )

    class FakeStore:
        def __init__(self, collection=None):
            pass

        def reset(self):
            pass

        def count(self):
            return 0

    monkeypatch.setattr(seed, "Store", FakeStore)
    monkeypatch.setattr(seed.ingest, "_index", index)
    monkeypatch.setattr(seed, "_safe_repo_path", lambda relative: seed.REPO_ROOT / relative)
    monkeypatch.setattr(seed, "ingest_paths", lambda path, collection=None: iter([]))

    seed.seed_collection("second_brain_public", ["README.md"])
    index.upsert_chunks(
        "second_brain_public",
        [{"source": "README.md", "chunk": 0, "name": "README.md", "document": "zephyrquill: a file still in the corpus"}],
    )

    found = [hit["metadata"]["source"] for hit in index.query("second_brain_public", "zephyrquill", limit=5)]
    assert found == ["README.md"]
