"""Seed audited public demo corpora into explicit non-private collections.

Only committed repo files listed below are eligible. The script refuses paths that
resolve outside this repository so it cannot accidentally ingest a private note.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from secondbrain import ingest
from secondbrain.ingest import ingest_paths
from secondbrain.store import Store

REPO_ROOT = Path(__file__).resolve().parents[1]

CORPORA: dict[str, list[str]] = {
    "second_brain_public": [
        "README.md",
        "docs/design.md",
        "docs/requirements.md",
        "docs/MCP.md",
        "docs/EVALUATION.md",
        # The deployment-planning note (docs/web-ui-scope.md, gitignored and
        # local-only) is deliberately absent, and so is anything like it: it
        # describes the deployment's own topology — tunnel hostnames, ports,
        # service layout. A visitor asking "what's your infrastructure?" was
        # getting a cited tour of it. The corpus a demo serves must never
        # document the demo.
    ],
    # The corpus the public artjeck.com demo actually serves
    # (SB_WEB_PUBLIC_COLLECTION=second_brain_concierge): sales-oriented,
    # pricing anchors included. Keep those anchors in lockstep with
    # artjeck-technology/src/lib/site-data.ts — the doc's own header says so.
    "second_brain_concierge": [
        "docs/concierge-corpus.md",
    ],
    "second_brain_neutral": [
        "evals/corpus/admin-preferences.md",
        "evals/corpus/ai-lab-routing.md",
        "evals/corpus/backup-policy.md",
        "evals/corpus/career-plan.md",
        "evals/corpus/deployment-access.md",
        "evals/corpus/document-workflows.md",
        "evals/corpus/logistics-product.md",
        "evals/corpus/second-brain-operations.md",
    ],
}


def _safe_repo_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"Refusing to ingest outside repo: {path}")
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Refusing to ingest untracked file: {relative}") from exc
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {relative}")
    if not path.is_file():
        raise ValueError(f"Corpus entry must be a file: {relative}")
    return path


def seed_collection(collection: str, relatives: list[str]) -> dict:
    files = chunks = 0
    print(f"\nSeeding {collection}")
    Store(collection=collection).reset()
    # ingest_paths replaces a file's keyword rows only when it ingests that file
    # again, so the rows of a file dropped from CORPORA would outlive this reseed.
    # Reset through ingest's own index: another KeywordIndex would drop the table
    # behind the cache ingest's instance keeps, and ingest's next write would raise.
    ingest._index.reset(collection)
    for relative in relatives:
        _safe_repo_path(relative)
        for file_path, n_chunks in ingest_paths(Path(relative), collection=collection):
            files += 1
            chunks += n_chunks
            print(f"  + {file_path} ({n_chunks} chunks)")
    total = Store(collection=collection).count()
    print(f"  total: {total} chunks")
    return {"collection": collection, "files": files, "chunks": chunks, "total": total}


def main() -> None:
    for relatives in CORPORA.values():
        for relative in relatives:
            _safe_repo_path(relative)
    import os

    os.chdir(REPO_ROOT)
    for collection, relatives in CORPORA.items():
        seed_collection(collection, relatives)


if __name__ == "__main__":
    main()
