"""Move a tree of reference material out of the personal collection.

Study texts, standards and other people's documentation are worth keeping and
worth retrieving — but not in the same ranking as the owner's own notes, where
they win on sheer volume. On 2026-09-12 the ISTQB syllabus PDFs were 2,935 of
4,483 chunks, and a question about this system's architecture spent two of its
five answer slots on them.

Order matters and is the whole safety property: **ingest into the reference
collection first, verify the chunks arrived, and only then delete from the
personal one.** A crash between the two leaves the material in both collections,
which is untidy; the reverse order would lose it.

    uv run python scripts/split_reference_corpus.py /path/to/reference/root
    uv run python scripts/split_reference_corpus.py /path/to/reference/root --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from secondbrain.config import cfg
from secondbrain.ingest import ingest_paths
from secondbrain.store import Store


def plan(root: Path) -> dict:
    """Which sources under `root` the personal collection is currently holding."""
    personal = Store(collection=cfg.collection)
    prefix = str(root.resolve())
    sources = {
        source: chunks
        for source, chunks in personal.sources().items()
        if source == prefix or source.startswith(prefix + "/")
    }
    return {"sources": sources, "chunks": sum(sources.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="Directory of reference material to move")
    parser.add_argument("--apply", action="store_true", help="Actually write and delete (default: dry run)")
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    before = plan(root)
    reference = Store(collection=cfg.reference_collection)
    print(f"source root     : {root}")
    print(f"personal        : {cfg.collection}")
    print(f"reference       : {cfg.reference_collection} ({reference.count()} chunks now)")
    print(f"to move         : {len(before['sources'])} source(s), {before['chunks']} chunk(s)")

    if not before["sources"]:
        print("nothing to move.")
        return 0

    if not args.apply:
        for source in sorted(before["sources"])[:10]:
            print(f"  would move: {source}")
        if len(before["sources"]) > 10:
            print(f"  ...and {len(before['sources']) - 10} more")
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    print("\ningesting into the reference collection...")
    ingested_files = ingested_chunks = 0
    for path, chunks in ingest_paths(root, collection=cfg.reference_collection):
        ingested_files += 1
        ingested_chunks += chunks
        print(f"  + {path.name}: {chunks} chunk(s)")

    if ingested_chunks == 0:
        print("refusing to delete: nothing was written to the reference collection.", file=sys.stderr)
        return 1

    # Only now is it safe to remove the originals.
    print("\nremoving those sources from the personal collection...")
    personal = Store(collection=cfg.collection)
    for source in before["sources"]:
        personal.delete_source(source)

    after_personal = personal.count()
    after_reference = reference.count()
    print(f"\npersonal  : {after_personal} chunks")
    print(f"reference : {after_reference} chunks ({ingested_files} file(s), {ingested_chunks} chunk(s) ingested)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
