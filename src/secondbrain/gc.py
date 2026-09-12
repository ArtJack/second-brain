"""Remove chunks whose source file no longer exists.

Ingestion keys a chunk by the absolute path it was read from, so a file that
moves is not updated — it is duplicated, and the old copy stays citable forever.
Moving this repository once left 31% of the collection pointing at paths that
cannot be opened: those chunks still win retrieval slots, and the citation they
produce is a dead link.

The sweep is deliberately conservative, because the naive version of it is the
most destructive routine in this codebase. `Path.exists()` answers False both for
"the owner deleted this note" and for "the network share holding every learned
memory is not mounted right now", and only one of those should delete anything.
Two guards separate them:

* A source under `/Volumes/<name>` where `<name>` itself is absent is not missing
  data, it is an absent disk. That aborts the run.
* A sweep that would remove more than half the collection aborts too. At that
  scale the likeliest explanation is that the question is wrong, not the corpus.

Relative sources are skipped rather than resolved, since what they are relative
to depends on the working directory of whoever ingested them.
"""
from __future__ import annotations

from pathlib import Path

from .keyword_index import KeywordIndex
from .store import Store

# Above this share of sources, refuse and make the caller say --force.
REFUSAL_THRESHOLD = 0.5

# Where removable volumes appear. A path under one of these is only believed to
# be missing if its volume is genuinely mounted.
_VOLUME_PARENTS = ("/Volumes",)

# Deleting from the store alone leaves keyword search still serving the dead
# citations this module exists to remove.
_index = KeywordIndex()


class GarbageCollectionRefused(RuntimeError):
    """Raised when the evidence for deleting looks more like a mistake than a fact."""


def _is_mounted(path: Path) -> bool:
    try:
        return path.is_mount()
    except OSError:
        # A hung mount can raise rather than answer. Unreachable either way.
        return False


def _unreachable_volume(path: Path) -> str | None:
    """The volume root this path needs, if that volume is not currently mounted.

    The question is whether the *volume* is there, not whether its mount point
    directory is. Asking the second was the defect: a tailnet drop leaves
    `/Volumes/DISK` in place as an empty directory, so the guard saw a directory,
    concluded the disk was present, and read every file under it as deleted —
    the precise failure this guard exists to prevent, and the one it could not
    see. `is_mount()` distinguishes them: a live share answers True, a leftover
    mount point answers False.
    """
    for parent in _VOLUME_PARENTS:
        parent_path = Path(parent)
        try:
            relative = path.relative_to(parent_path)
        except ValueError:
            continue
        if not relative.parts:
            continue
        volume = parent_path / relative.parts[0]
        if not volume.exists() or not _is_mounted(volume):
            return str(volume)
    return None


def collect_garbage(
    *,
    store=None,
    collection: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Delete every chunk whose source file is gone. Returns what it did."""
    store = store if store is not None else Store(collection=collection)
    sources = store.sources()

    missing: dict[str, int] = {}
    kept = 0
    skipped = 0
    unreachable: set[str] = set()

    for source, chunks in sources.items():
        path = Path(source)
        if not path.is_absolute():
            skipped += 1
            continue
        volume = _unreachable_volume(path)
        if volume:
            unreachable.add(volume)
            continue
        if path.exists():
            kept += 1
        else:
            missing[source] = chunks

    if unreachable:
        raise GarbageCollectionRefused(
            "refusing to sweep: "
            + ", ".join(sorted(unreachable))
            + " is not mounted, so every source under it would look deleted. "
            "Mount it and run again."
        )

    considered = len(missing) + kept
    if considered and not force:
        share = len(missing) / considered
        if share > REFUSAL_THRESHOLD:
            raise GarbageCollectionRefused(
                f"refusing to sweep: {share:.0%} of sources look missing "
                f"({len(missing)} of {considered}). That is more likely a wrong "
                "question than a real deletion. Re-run with --force if it is real."
            )

    if not dry_run:
        collection = getattr(store, "collection_name", None)
        for source in missing:
            store.delete_source(source)
            # The keyword index is a second copy of the same corpus. Removing a
            # source from only one of them leaves keyword search returning the
            # dead citation this sweep just removed.
            if collection:
                _index.delete_source(collection, source)

    return {
        "removed_sources": len(missing),
        "removed_chunks": sum(missing.values()),
        "kept_sources": kept,
        "skipped_sources": skipped,
        "dry_run": dry_run,
        "removed": sorted(missing),
    }
