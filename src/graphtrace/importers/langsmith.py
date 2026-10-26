"""LangSmith trace importer.

Field names (`run_type`, `inputs`, `outputs`, `start_time`, `end_time`, `error`)
match LangSmith's own documented `Run` schema -- this is the part verified
against current docs. The exact shape of `inputs`/`outputs` for an LLM run
(assumed here to hold a `messages` list, per LangSmith's documented convention)
and whether a real export wraps `runs` at the top level the way this importer
expects are **not independently verified against a live export** -- check your
own export if a real trace doesn't parse cleanly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from graphtrace.importers.base import NormalizedSpan, NormalizedTrace


def _parse_latency(start: str | None, end: str | None) -> float:
    if not start or not end:
        return 0.0
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return 0.0


class LangSmithImporter:
    name = "langsmith"

    def can_read(self, data: dict[str, Any]) -> bool:
        return "runs" in data

    def to_normalized(self, data: dict[str, Any]) -> NormalizedTrace:
        spans: list[NormalizedSpan] = []
        for run in data.get("runs", []):
            run_type = run.get("run_type")
            if run_type == "llm":
                kind = "llm"
            elif run_type == "tool":
                kind = "tool"
            else:
                # chain/retriever/prompt/parser runs aren't LLM or tool calls on
                # their own -- skipped, not guessed at.
                continue

            error = run.get("error")
            span = NormalizedSpan(
                node=run.get("name", "?"),
                kind=kind,
                request=run.get("inputs"),
                response=None if error else run.get("outputs"),
                error=error,
                latency_s=_parse_latency(run.get("start_time"), run.get("end_time")),
                model=(run.get("extra") or {}).get("model") if kind == "llm" else None,
                name=run.get("name") if kind == "tool" else None,
            )
            spans.append(span)
        return NormalizedTrace(spans=spans)
