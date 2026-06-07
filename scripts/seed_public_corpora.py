"""Seed audited public demo corpora into explicit non-private collections.

Only committed repo files listed below are eligible. The script refuses paths that
resolve outside this repository so it cannot accidentally ingest a private note.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

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
        "docs/web-ui-scope.md",
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
    for path in [_safe_repo_path(relative) for relative in relatives]:
        for file_path, n_chunks in ingest_paths(path, collection=collection):
            files += 1
            chunks += n_chunks
            print(f"  + {file_path.relative_to(REPO_ROOT)} ({n_chunks} chunks)")
    total = Store(collection=collection).count()
    print(f"  total: {total} chunks")
    return {"collection": collection, "files": files, "chunks": chunks, "total": total}


def main() -> None:
    for relatives in CORPORA.values():
        for relative in relatives:
            _safe_repo_path(relative)
    for collection, relatives in CORPORA.items():
        seed_collection(collection, relatives)


if __name__ == "__main__":
    main()
