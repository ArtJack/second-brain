"""`reset()` must rebuild the collection it was opened on.

`sb --collection scratch ingest --reset` was a data-loss trap: reset() deleted
the named collection and then recreated `cfg.collection`, so the handle silently
became the owner's real brain and every subsequent upsert in that run landed
there. The Qdrant backend never had the bug; Chroma is the zero-infra default,
which is what most people run first.
"""
from __future__ import annotations

from secondbrain.store import ChromaStore


def test_reset_rebuilds_the_collection_it_was_opened_on(tmp_path):
    store = ChromaStore(persist_dir=tmp_path, collection="scratch")
    store.upsert(
        ids=["a#0"],
        embeddings=[[0.1, 0.2]],
        documents=["before the reset"],
        metadatas=[{"source": "/a.md", "chunk": 0}],
    )

    store.reset()
    store.upsert(
        ids=["b#0"],
        embeddings=[[0.1, 0.2]],
        documents=["after the reset"],
        metadatas=[{"source": "/b.md", "chunk": 0}],
    )

    assert store.collection_name == "scratch"
    assert store.count() == 1
    assert ChromaStore(persist_dir=tmp_path, collection="scratch").count() == 1


def test_reset_does_not_touch_the_default_collection(tmp_path):
    """The actual harm: writes intended for a scratch collection reaching the real one."""
    from secondbrain.config import cfg

    real = ChromaStore(persist_dir=tmp_path, collection=cfg.collection)
    real.upsert(
        ids=["real#0"],
        embeddings=[[0.3, 0.4]],
        documents=["the owner's actual note"],
        metadatas=[{"source": "/real.md", "chunk": 0}],
    )

    scratch = ChromaStore(persist_dir=tmp_path, collection="scratch")
    scratch.reset()
    scratch.upsert(
        ids=["s#0"],
        embeddings=[[0.1, 0.2]],
        documents=["scratch data"],
        metadatas=[{"source": "/s.md", "chunk": 0}],
    )

    assert ChromaStore(persist_dir=tmp_path, collection=cfg.collection).count() == 1
    assert ChromaStore(persist_dir=tmp_path, collection="scratch").count() == 1
