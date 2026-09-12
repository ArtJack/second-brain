"""Remove chunks that no longer belong in the corpus.

Two reasons a chunk stops belonging. The first is that its source file is gone.
The second is that the owner's own scan rules now exclude it — and that one used
to have no remedy at all: the exclusion list is consulted at ingest time only, so
adding `*secret*` stopped the next ingest and left everything the previous ingest
had already embedded fully retrievable and citable. For the one list whose entire
purpose is keeping material out of the corpus, that is backwards. Writing a rule
is precisely the moment the owner has found something they want gone.

Rule enforcement is opt-in (`--enforce-rules`), because unlike a missing file it
deletes chunks whose source is still sitting on disk. The default sweep counts
them and says so, which costs nothing and makes the drift visible.

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

import fnmatch
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


def _load_scan_rules() -> dict:
    """The nightly scan's exclusion rules, read from the scan's own config.

    Through the scan's loader rather than a second copy of the path, for two
    reasons: it honours `SB_STATE_DB`, so a test or a second corpus does not read
    the live owner's rules; and it merges the built-in defaults, so a hand-edited
    config that omits `exclude_globs` still enforces the rules the scan enforces.
    A rule has to mean the same thing to both halves of the system or this is
    worse than not having it.
    """
    from .overnight import ensure_config, overnight_paths

    return ensure_config(overnight_paths().config)


def _excluding_rule(path: Path, rules: dict) -> str | None:
    """The rule that excludes this path from the corpus, or None.

    Deliberately the same two questions the scan asks, in the same order: does
    the file name match an excluded glob, and does any component of the path sit
    in an excluded directory. A third question here would mean the corpus could
    hold a file the scan would have rejected, which is the bug this closes.
    """
    name = path.name.lower()
    for rule in rules.get("exclude_globs", []):
        if fnmatch.fnmatch(name, str(rule).lower()):
            return str(rule)
    excluded_dirs = set(rules.get("exclude_dirs", []))
    for part in path.parts[:-1]:
        if part in excluded_dirs:
            return f"dir:{part}"
    return None


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
    enforce_rules: bool = False,
    rules: dict | None = None,
) -> dict:
    """Delete every chunk whose source is gone, and optionally every one the rules exclude."""
    store = store if store is not None else Store(collection=collection)
    sources = store.sources()

    if rules is None and enforce_rules:
        try:
            rules = _load_scan_rules()
        except (OSError, ValueError):
            # Losing the rules must not lose the sweep. The missing-file half is
            # the part that runs nightly and the part with no other remedy.
            rules = None
    rules = rules or {}

    missing: dict[str, int] = {}
    excluded: dict[str, tuple[int, str]] = {}
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
        if not path.exists():
            missing[source] = chunks
            continue
        rule = _excluding_rule(path, rules) if rules else None
        if rule:
            excluded[source] = (chunks, rule)
        else:
            kept += 1

    if unreachable:
        raise GarbageCollectionRefused(
            "refusing to sweep: "
            + ", ".join(sorted(unreachable))
            + " is not mounted, so every source under it would look deleted. "
            "Mount it and run again."
        )

    # A rule removal is as destructive as a missing-file removal, so it is
    # counted by the same guard. `*.md` is one keystroke away from `*.mdx`, and
    # a typo in the exclusion list should hit the same wall a wrong question does.
    doomed = len(missing) + (len(excluded) if enforce_rules else 0)
    considered = doomed + kept
    if considered and not force:
        share = doomed / considered
        if share > REFUSAL_THRESHOLD:
            raise GarbageCollectionRefused(
                f"refusing to sweep: {share:.0%} of sources would be removed "
                f"({doomed} of {considered}). That is more likely a wrong "
                "question than a real deletion. Re-run with --force if it is real."
            )

    if not dry_run:
        collection = getattr(store, "collection_name", None)
        removing = list(missing) + (list(excluded) if enforce_rules else [])
        for source in removing:
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
        "excluded_sources": len(excluded),
        "excluded_chunks": sum(chunks for chunks, _ in excluded.values()),
        "excluded_enforced": enforce_rules,
        "excluded": sorted(excluded),
        "excluded_by_rule": {source: rule for source, (_, rule) in sorted(excluded.items())},
    }
