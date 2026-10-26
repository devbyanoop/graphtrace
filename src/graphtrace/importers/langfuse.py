"""Langfuse trace importer.

Field names below match Langfuse's own documented "observations" model as of
this writing (a trace has `observations`, each one typed `GENERATION` for an
LLM call or `SPAN` for everything else; a generation carries `model`/`usage`).
Langfuse's export format has moved fast (a v4 data model and "enriched
observations" shipped in 2026) -- **verify field names against a real export
from your own project before trusting this against anything beyond the bundled
test fixture**. This is the honest state of things, not a guess presented as
certain.
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


class LangfuseImporter:
    name = "langfuse"

    def can_read(self, data: dict[str, Any]) -> bool:
        return "observations" in data

    def to_normalized(self, data: dict[str, Any]) -> NormalizedTrace:
        spans: list[NormalizedSpan] = []
        for obs in data.get("observations", []):
            obs_type = obs.get("type")
            if obs_type == "GENERATION":
                kind = "llm"
            elif obs_type == "SPAN":
                kind = "tool"
            else:
                # EVENT and anything else isn't a call this project's Trajectory
                # model represents -- skipped, not guessed at.
                continue

            is_error = obs.get("level") == "ERROR"
            span = NormalizedSpan(
                node=obs.get("name", "?"),
                kind=kind,
                request=obs.get("input"),
                response=None if is_error else obs.get("output"),
                error=obs.get("statusMessage") if is_error else None,
                latency_s=_parse_latency(obs.get("startTime"), obs.get("endTime")),
                model=obs.get("model") if kind == "llm" else None,
                name=obs.get("name") if kind == "tool" else None,
            )
            spans.append(span)
        return NormalizedTrace(spans=spans, initial_input=data.get("input"))
