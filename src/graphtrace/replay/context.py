"""ReplayContext: everything an interceptor needs to behave correctly for the
active mode, built once per run_graph()/arun_graph() call and shared by every
patched method for that call.

Lives in `replay/`, not `frameworks/langgraph/`, on purpose -- nothing here is
LangGraph-specific. A future CrewAI/AutoGen interceptor (v0.4) reuses this exact
class; only how a framework's own calls get routed through `_resolve_call`
(frameworks/langgraph/interceptors.py) is framework-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphtrace.replay.cassette import Cassette
from graphtrace.replay.matcher import Matcher
from graphtrace.replay.modes import Mode


@dataclass
class ReplayContext:
    mode: Mode
    cassette: Cassette
    matcher: Matcher = field(default_factory=Matcher)
    new_entries: list[dict[str, Any]] = field(default_factory=list)
