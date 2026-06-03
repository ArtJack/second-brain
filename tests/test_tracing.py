"""Dependency-free local trace/span recorder and standalone JSON export."""
from __future__ import annotations

import json

import pytest

from secondbrain.tracing import TraceRecorder, summarize_traces, write_trace_export


def test_nested_spans_produce_ordered_trajectory():
    trace = TraceRecorder("test.trace", attributes={"case_id": "one"})

    with trace.span("root", kind="workflow", record_trajectory=False) as root:
        with trace.span("retrieve", kind="retrieval", parent_span_id=root.span_id) as retrieve:
            retrieve.add_attributes({"hit_count": 2})
            retrieve.set_summary({"hit_count": 2})
        with trace.span("evaluate", kind="evaluator", parent_span_id=root.span_id) as evaluate:
            evaluate.add_attributes({"passed": True})
            evaluate.set_summary({"passed": True})
    trace.finish(attributes={"passed": True})

    record = trace.to_dict()
    assert record["root_span_id"] == root.span_id
    assert record["attributes"]["passed"] is True
    assert [step["name"] for step in record["trajectory"]] == ["retrieve", "evaluate"]
    assert record["trajectory"][0]["sequence"] == 1
    assert record["trajectory"][0]["parent_span_id"] == root.span_id
    assert record["trajectory"][0]["summary"] == {"hit_count": 2}
    assert len(record["spans"]) == 3


def test_span_context_records_operational_error():
    trace = TraceRecorder("test.trace")

    with pytest.raises(RuntimeError, match="offline"):
        with trace.span("retrieve", kind="retrieval"):
            raise RuntimeError("offline")
    trace.finish(status="error")

    record = trace.to_dict()
    assert record["status"] == "error"
    assert record["spans"][0]["status"] == "error"
    assert record["spans"][0]["error"] == "RuntimeError: offline"


def test_trace_export_is_json_and_has_summary(tmp_path):
    trace = TraceRecorder("test.trace")
    with trace.span("retrieve", kind="retrieval"):
        pass
    trace.finish()

    path = write_trace_export(tmp_path / "trace.json", benchmark="test", traces=[trace.to_dict()])
    exported = json.loads(path.read_text(encoding="utf-8"))

    assert exported["benchmark"] == "test"
    assert exported["summary"] == {
        "traces": 1,
        "spans": 1,
        "trajectory_steps": 1,
        "error_spans": 0,
    }
    assert summarize_traces(exported["traces"]) == exported["summary"]
