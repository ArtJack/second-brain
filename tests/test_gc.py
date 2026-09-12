"""Collecting garbage from the index must never mistake an absent disk for absent data.

31% of the live collection cites files that no longer exist — moved repositories,
deleted worktrees — and those chunks can never be opened from a citation, so they
are pure loss: they crowd out real evidence in every top-k.

The obvious implementation of "delete what no longer exists" is also the most
dangerous code in this repository. The owner's learned memories live on an SMB
share over a tailnet. If that share is unmounted when the sweep runs, every
memory file answers False to `exists()` and a naive sweep deletes the one part of
the corpus with no other copy. The same is true of any target under /Volumes.

So absence is only believed when the surrounding filesystem is present to be
asked. Two guards, both tested here: a path under an unmounted volume aborts the
run rather than counting as missing, and a sweep that would remove more than half
the collection aborts as well, because at that scale the likeliest explanation is
that something is wrong with the question, not with the data.
"""
from __future__ import annotations

import pytest

from secondbrain.gc import GarbageCollectionRefused, collect_garbage


class FakeStore:
    collection_name = "test_collection"

    def __init__(self, sources) -> None:
        # A list means "one chunk each", which is all the older tests care
        # about; a mapping lets a test say how many chunks a source holds.
        self._sources = sources if isinstance(sources, dict) else {s: 1 for s in sources}
        self.deleted: list[str] = []

    def sources(self) -> dict[str, int]:
        return dict(self._sources)

    def delete_source(self, source: str) -> None:
        self.deleted.append(source)


def test_a_source_whose_file_is_gone_is_removed(tmp_path):
    present = tmp_path / "kept.md"
    present.write_text("still here")
    gone = tmp_path / "deleted.md"
    store = FakeStore([str(present), str(gone)])

    result = collect_garbage(store=store)

    assert store.deleted == [str(gone)]
    assert result["removed_sources"] == 1
    assert result["kept_sources"] == 1


def test_dry_run_reports_without_deleting(tmp_path):
    present = tmp_path / "kept.md"
    present.write_text("still here")
    store = FakeStore([str(present), str(tmp_path / "deleted.md")])

    result = collect_garbage(store=store, dry_run=True)

    assert store.deleted == []
    assert result["removed_sources"] == 1
    assert result["dry_run"] is True


def test_an_unmounted_volume_aborts_instead_of_counting_as_missing(tmp_path):
    """The failure this whole module exists to prevent."""
    present = tmp_path / "kept.md"
    present.write_text("still here")
    on_the_share = "/Volumes/NotMountedRightNow/AI/artjeck/memory/a-memory.md"
    store = FakeStore([str(present), on_the_share])

    with pytest.raises(GarbageCollectionRefused) as excinfo:
        collect_garbage(store=store)

    assert store.deleted == []
    assert "/Volumes/NotMountedRightNow" in str(excinfo.value)


def test_a_relative_source_is_never_deleted(tmp_path):
    present = tmp_path / "kept.md"
    present.write_text("still here")
    store = FakeStore([str(present), "examples/lab-notes.md"])

    result = collect_garbage(store=store)

    assert store.deleted == []
    assert result["skipped_sources"] == 1


def test_removing_most_of_the_collection_aborts_unless_forced(tmp_path):
    present = tmp_path / "kept.md"
    present.write_text("still here")
    store = FakeStore([str(present)] + [str(tmp_path / f"gone-{i}.md") for i in range(9)])

    with pytest.raises(GarbageCollectionRefused) as excinfo:
        collect_garbage(store=store)

    assert store.deleted == []
    assert "90%" in str(excinfo.value)

    result = collect_garbage(store=store, force=True)

    assert len(store.deleted) == 9
    assert result["removed_sources"] == 9


def test_an_empty_collection_is_not_an_error():
    store = FakeStore([])

    result = collect_garbage(store=store)

    assert result == {
        "removed_sources": 0,
        "removed_chunks": 0,
        "kept_sources": 0,
        "skipped_sources": 0,
        "excluded_sources": 0,
        "excluded_chunks": 0,
        "excluded_enforced": False,
        "excluded": [],
        "excluded_by_rule": {},
        "dry_run": False,
        "removed": [],
    }


def test_the_real_stores_can_report_their_sources(tmp_path):
    """`sources()` is the store contract gc depends on; ChromaStore must honour it."""
    from secondbrain.store import ChromaStore

    store = ChromaStore(persist_dir=tmp_path, collection="gc_contract")
    store.upsert(
        ids=["/a/one.md#0", "/a/one.md#1", "/b/two.md#0"],
        embeddings=[[0.1, 0.2], [0.1, 0.2], [0.3, 0.4]],
        documents=["one a", "one b", "two"],
        metadatas=[
            {"source": "/a/one.md", "chunk": 0},
            {"source": "/a/one.md", "chunk": 1},
            {"source": "/b/two.md", "chunk": 0},
        ],
    )

    assert store.sources() == {"/a/one.md": 2, "/b/two.md": 1}

    store.delete_source("/a/one.md")

    assert store.sources() == {"/b/two.md": 1}


def test_a_mount_point_that_exists_but_is_not_mounted_aborts(tmp_path, monkeypatch):
    """Verdict SB-F-26, Major: the guard asked the wrong question.

    It tested whether the mount *point* directory exists. A tailnet drop — the
    exact failure this module was written to survive — leaves /Volumes/DISK in
    place as an empty directory, so the guard saw a directory, believed the disk
    was there, and read every memory file under it as deleted. Whether they were
    then destroyed came down to the unrelated 50% threshold.

    The original test used a root that did not exist at all, so it never posed
    the distinction between "root absent" and "root present but not mounted".
    """
    import secondbrain.gc as gc_module

    present = tmp_path / "kept.md"
    present.write_text("still here")
    # A real directory that is not a mount point, standing in for a share whose
    # tailnet went away.
    stale_mount = tmp_path / "Volumes" / "DISK"
    stale_mount.mkdir(parents=True)
    monkeypatch.setattr(gc_module, "_VOLUME_PARENTS", (str(tmp_path / "Volumes"),))

    store = FakeStore([str(present), str(stale_mount / "artjeck" / "memory" / "a-memory.md")])

    with pytest.raises(GarbageCollectionRefused) as excinfo:
        collect_garbage(store=store)

    assert store.deleted == []
    assert "DISK" in str(excinfo.value)


def test_a_genuinely_mounted_volume_does_not_block_the_sweep(tmp_path, monkeypatch):
    """The guard must not make gc unusable whenever a share is involved."""
    import secondbrain.gc as gc_module

    present = tmp_path / "kept.md"
    present.write_text("still here")
    monkeypatch.setattr(gc_module, "_VOLUME_PARENTS", (str(tmp_path / "Volumes"),))
    monkeypatch.setattr(gc_module, "_is_mounted", lambda path: True)

    gone = tmp_path / "Volumes" / "DISK" / "deleted-note.md"
    gone.parent.mkdir(parents=True)
    store = FakeStore([str(present), str(gone)])

    result = collect_garbage(store=store)

    assert result["removed_sources"] == 1
    assert store.deleted == [str(gone)]


def test_gc_removes_the_keyword_rows_too(tmp_path, monkeypatch):
    """Verdict SB-F-25: deleting from the store alone leaves keyword search
    serving the dead citations gc exists to remove."""
    import secondbrain.gc as gc_module

    present = tmp_path / "kept.md"
    present.write_text("still here")
    gone = tmp_path / "deleted.md"
    store = FakeStore([str(present), str(gone)])
    store.collection_name = "c"

    dropped: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gc_module._index, "delete_source", lambda collection, source: dropped.append((collection, source))
    )

    collect_garbage(store=store)

    assert dropped == [("c", str(gone))]


class TestRulesAreRetroactive:
    """Adding an exclusion rule did nothing about what the old rules let in.

    The scan rules are prospective only: `*secret*` stops the next ingest, and
    the chunks from the ingest that already happened stay retrievable and
    citable forever. That is the wrong default for the one list whose purpose is
    keeping material out of the corpus — the moment an owner writes a rule is
    exactly the moment they have discovered something they want gone.

    `gc` is where this belongs. It already walks every source deciding what no
    longer belongs in the corpus; "the owner's own rules now exclude this" is
    another answer to that same question. It stays opt-in, because unlike a
    missing file this deletes something that is still on disk.
    """

    def _rules(self, tmp_path):
        return {"exclude_globs": ["*secret*", "package-lock.json"], "exclude_dirs": ["node_modules"]}

    def test_a_dry_run_reports_the_drift_without_being_asked(self, tmp_path):
        """Surfacing it is free; deleting is not. The default does the free half."""
        from secondbrain import gc as gc_mod

        keep = tmp_path / "notes.md"
        secret = tmp_path / "aws-secrets.md"
        for path in (keep, secret):
            path.write_text("x")
        store = FakeStore({str(keep): 3, str(secret): 2})

        res = gc_mod.collect_garbage(store=store, dry_run=True, rules=self._rules(tmp_path))

        assert res["excluded_sources"] == 1
        assert res["excluded"] == [str(secret)]
        assert store.deleted == [], "a dry run deletes nothing"

    def test_the_default_sweep_reports_but_does_not_delete_them(self, tmp_path):
        from secondbrain import gc as gc_mod

        secret = tmp_path / "aws-secrets.md"
        secret.write_text("x")
        keep = tmp_path / "notes.md"
        keep.write_text("x")
        store = FakeStore({str(keep): 3, str(secret): 2})

        res = gc_mod.collect_garbage(store=store, rules=self._rules(tmp_path))

        assert res["excluded_sources"] == 1
        assert store.deleted == [], "removing a file that still exists needs an explicit ask"

    def test_enforcing_removes_them_from_the_store_and_the_keyword_index(self, tmp_path, monkeypatch):
        from secondbrain import gc as gc_mod

        secret = tmp_path / "aws-secrets.md"
        lock = tmp_path / "package-lock.json"
        keepers = [tmp_path / f"notes{i}.md" for i in range(5)]
        for path in (secret, lock, *keepers):
            path.write_text("x")
        store = FakeStore({str(secret): 2, str(lock): 90, **{str(k): 3 for k in keepers}})
        index_deletes: list[str] = []
        monkeypatch.setattr(
            gc_mod._index, "delete_source", lambda collection, source: index_deletes.append(source)
        )

        res = gc_mod.collect_garbage(store=store, rules=self._rules(tmp_path), enforce_rules=True)

        assert sorted(store.deleted) == sorted([str(secret), str(lock)])
        assert sorted(index_deletes) == sorted([str(secret), str(lock)])
        assert res["excluded_chunks"] == 92
        assert all(str(k) not in store.deleted for k in keepers)

    def test_a_directory_rule_excludes_everything_under_it_at_any_depth(self, tmp_path):
        from secondbrain import gc as gc_mod

        buried = tmp_path / "app" / "node_modules" / "left-pad" / "readme.md"
        buried.parent.mkdir(parents=True)
        buried.write_text("x")
        keep = tmp_path / "app" / "readme.md"
        keep.write_text("x")
        store = FakeStore({str(keep): 1, str(buried): 4})

        res = gc_mod.collect_garbage(store=store, rules=self._rules(tmp_path), enforce_rules=True)

        assert store.deleted == [str(buried)]
        assert res["excluded_chunks"] == 4

    def test_an_empty_rule_set_excludes_nothing(self, tmp_path, monkeypatch):
        from secondbrain import gc as gc_mod

        monkeypatch.setattr(gc_mod, "_load_scan_rules", dict)
        secret = tmp_path / "aws-secrets.md"
        secret.write_text("x")
        store = FakeStore({str(secret): 2})

        res = gc_mod.collect_garbage(store=store, enforce_rules=True)

        assert store.deleted == []
        assert res["excluded_sources"] == 0

    def test_the_majority_guard_counts_rule_removals_too(self, tmp_path):
        """Otherwise a careless rule silently empties the corpus.

        `*.md` is one keystroke from `*.mdx`, and the guard that catches a wrong
        question about missing files has to catch a wrong rule as well.
        """
        from secondbrain import gc as gc_mod

        sources = {}
        for i in range(10):
            path = tmp_path / f"note{i}.md"
            path.write_text("x")
            sources[str(path)] = 1
        store = FakeStore(sources)

        with pytest.raises(gc_mod.GarbageCollectionRefused) as excinfo:
            gc_mod.collect_garbage(store=store, rules={"exclude_globs": ["*.md"]}, enforce_rules=True)

        assert store.deleted == []
        assert "100%" in str(excinfo.value)

    def test_an_unreadable_rules_file_does_not_take_the_sweep_down(self, tmp_path, monkeypatch):
        """The missing-file sweep is the part that must keep working."""
        from secondbrain import gc as gc_mod

        def unreadable():
            raise OSError("nope")

        monkeypatch.setattr(gc_mod, "_load_scan_rules", unreadable)
        gone = tmp_path / "gone.md"
        keepers = [tmp_path / f"here{i}.md" for i in range(3)]
        for path in keepers:
            path.write_text("x")
        store = FakeStore({str(gone): 2, **{str(k): 1 for k in keepers}})

        res = gc_mod.collect_garbage(store=store, enforce_rules=True)

        assert res["removed_sources"] == 1
        assert res["excluded_sources"] == 0
