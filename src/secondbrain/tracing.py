"""Small local trace/span recorder with ordered trajectories.

This is deliberately dependency-free. JSON traces stay useful offline and can later be
exported to OpenTelemetry or another backend without coupling core evaluation to a service.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRACE_SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


class Span:
    """One timed operation inside a trace."""

    def __init__(
        self,
        recorder: "TraceRecorder",
        *,
        name: str,
        kind: str,
        parent_span_id: str | None,
        attributes: dict[str, Any] | None,
        record_trajectory: bool,
    ) -> None:
        self._recorder = recorder
        self.span_id = uuid.uuid4().hex
        self.parent_span_id = parent_span_id
        self.name = name
        self.kind = kind
        self.attributes = dict(attributes or {})
        self.record_trajectory = record_trajectory
        self.started_at = _timestamp()
        self._started = time.perf_counter()
        self._summary: dict[str, Any] = {}
        self._finished = False
        self.duration_ms: float | None = None

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc, _traceback) -> None:
        if exc is None:
            self.finish()
        else:
            self.finish(status="error", error=f"{exc_type.__name__}: {exc}")

    def add_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)

    def set_summary(self, summary: dict[str, Any]) -> None:
        self._summary = dict(summary)

    def finish(self, *, status: str = "ok", error: str | None = None) -> None:
        if self._finished:
            return
        self.duration_ms = round((time.perf_counter() - self._started) * 1000, 2)
        record = {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "status": status,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
        }
        if error:
            record["error"] = error
        self._recorder._record_span(record, summary=self._summary, include_in_trajectory=self.record_trajectory)
        self._finished = True


class TraceRecorder:
    """Collect nested spans and their ordered trajectory for one end-to-end operation."""

    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.trace_id = uuid.uuid4().hex
        self.name = name
        self.attributes = dict(attributes or {})
        self.started_at = _timestamp()
        self._started = time.perf_counter()
        self._finished = False
        self._status = "ok"
        self._duration_ms: float | None = None
        self._spans: list[dict] = []
        self._trajectory: list[dict] = []
        self.root_span_id: str | None = None

    def span(
        self,
        name: str,
        *,
        kind: str = "internal",
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        record_trajectory: bool = True,
    ) -> Span:
        span = Span(
            self,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            attributes=attributes,
            record_trajectory=record_trajectory,
        )
        if parent_span_id is None and self.root_span_id is None:
            self.root_span_id = span.span_id
        return span

    def _record_span(self, span: dict, *, summary: dict[str, Any], include_in_trajectory: bool) -> None:
        self._spans.append(span)
        if include_in_trajectory:
            self._trajectory.append(
                {
                    "sequence": len(self._trajectory) + 1,
                    "span_id": span["span_id"],
                    "parent_span_id": span["parent_span_id"],
                    "name": span["name"],
                    "kind": span["kind"],
                    "status": span["status"],
                    "duration_ms": span["duration_ms"],
                    "summary": summary,
                }
            )

    def finish(self, *, status: str = "ok", attributes: dict[str, Any] | None = None) -> None:
        if attributes:
            self.attributes.update(attributes)
        if self._finished:
            return
        self._status = status
        self._duration_ms = round((time.perf_counter() - self._started) * 1000, 2)
        self._finished = True

    def to_dict(self) -> dict:
        if not self._finished:
            self.finish()
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "root_span_id": self.root_span_id,
            "name": self.name,
            "status": self._status,
            "started_at": self.started_at,
            "duration_ms": self._duration_ms,
            "attributes": self.attributes,
            "spans": self._spans,
            "trajectory": self._trajectory,
        }


def summarize_traces(traces: list[dict]) -> dict:
    """Compact operational summary for CLI output and report metadata."""
    spans = [span for trace in traces for span in trace["spans"]]
    return {
        "traces": len(traces),
        "spans": len(spans),
        "trajectory_steps": sum(len(trace["trajectory"]) for trace in traces),
        "error_spans": sum(1 for span in spans if span["status"] == "error"),
    }


def write_trace_export(path: str | Path, *, benchmark: str, traces: list[dict]) -> Path:
    """Write local traces as a standalone JSON artifact."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "benchmark": benchmark,
                "summary": summarize_traces(traces),
                "traces": traces,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
