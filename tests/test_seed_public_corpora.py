import importlib.util
from pathlib import Path


def _load_seed_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "seed_public_corpora.py"
    spec = importlib.util.spec_from_file_location("seed_public_corpora_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_collection_resets_and_ingests_relative_sources(monkeypatch):
    seed = _load_seed_module()

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
