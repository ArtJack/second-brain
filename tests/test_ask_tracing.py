"""Optional nested tracing inside the grounded RAG answer path."""
from __future__ import annotations

import importlib

from secondbrain.tracing import TraceRecorder


def test_ask_records_store_embedding_retrieval_and_generation_spans(monkeypatch):
    ask_module = importlib.import_module("secondbrain.ask")

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def count(self):
            return 1

        def query(self, _embedding, _k):
            return [
                {
                    "document": "The gateway is on port 4000.",
                    "metadata": {"source": "notes/lab.md", "name": "lab.md"},
                    "distance": 0.1,
                }
            ]

    monkeypatch.setattr(ask_module, "Store", FakeStore)
    monkeypatch.setattr(ask_module, "embed", lambda _texts: [[0.1, 0.2]])
    monkeypatch.setattr(ask_module, "answer", lambda _question, _context: "Port 4000 [1].")
    trace = TraceRecorder("test.answer")

    with trace.span("answer_path", kind="rag", record_trajectory=False) as root:
        result = ask_module.ask("Where is the gateway?", trace=trace, parent_span_id=root.span_id)
    trace.finish()

    assert result["answer"] == "Port 4000 [1]."
    record = trace.to_dict()
    assert [step["name"] for step in record["trajectory"]] == [
        "store_count",
        "embed_question",
        "retrieve_context",
        "generate_answer",
    ]
    assert all(step["parent_span_id"] == root.span_id for step in record["trajectory"])
