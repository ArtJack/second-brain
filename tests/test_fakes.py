"""What the shared FakeStore does beyond the store contract: it records, and it can be told what to answer.

tests/test_store_contract.py holds the fake to the real stores. These are the
knobs only the fake has, so they have no real store to be compared with.
"""
from __future__ import annotations

import pytest

from secondbrain.fakes import FakeStore, FakeStoreFactory


def test_every_call_is_recorded_in_order_with_its_arguments():
    store = FakeStore("c")

    store.upsert(ids=["1"], embeddings=[[0.1]], documents=["d"], metadatas=[{"source": "/s.md", "chunk": 0}])
    store.query([0.1], 3)
    store.delete_source("/s.md")
    store.count()

    assert store.calls == [
        ("upsert", ["1"], [[0.1]], ["d"], [{"source": "/s.md", "chunk": 0}]),
        ("query", [0.1], 3),
        ("delete_source", "/s.md"),
        ("count",),
    ]
    assert store.deleted == ["/s.md"]


def test_query_returns_the_hits_it_was_given_exactly():
    hits = [{"document": "d", "metadata": {"source": "/s.md", "chunk": 0}, "distance": 0.3}]

    assert FakeStore(hits=hits).query([0.1], 5) is hits


def test_a_given_count_is_the_answer_whatever_the_store_holds():
    store = FakeStore(count=7)

    store.upsert(ids=["1"], embeddings=[[0.1]], documents=["d"], metadatas=[{"source": "/s.md", "chunk": 0}])

    assert store.count() == 7


def test_a_given_error_is_raised_by_every_call_once_it_is_recorded():
    store = FakeStore(error=RuntimeError("store unreachable"))
    calls = [
        ("upsert", ["1"], [[0.1]], ["d"], [{"source": "/s.md"}]),
        ("delete_source", "/s.md"),
        ("query", [0.1], 1),
        ("documents",),
        ("get_source_chunk", "/s.md", 0),
        ("sources",),
        ("count",),
        ("reset",),
    ]

    for method, *arguments in calls:
        with pytest.raises(RuntimeError, match="store unreachable"):
            getattr(store, method)(*arguments)

    assert store.calls == calls


def test_the_factory_records_each_collection_and_keeps_one_fake_per_collection():
    given = FakeStore("public", count=4)
    factory = FakeStoreFactory({"public": given}, count=1)

    assert factory(collection="public") is given
    built = factory(collection="neutral")
    assert built.collection_name == "neutral"
    assert built.count() == 1
    assert factory(collection="neutral") is built
    assert factory.collections == ["public", "neutral", "neutral"]
