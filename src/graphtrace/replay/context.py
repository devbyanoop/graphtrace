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

from graphtrace.replay.cassette import Cassette, load_cassette, save_cassette
from graphtrace.replay.matcher import Matcher
from graphtrace.replay.modes import Mode


@dataclass
class ReplayContext:
    mode: Mode
    cassette: Cassette
    matcher: Matcher = field(default_factory=Matcher)
    new_entries: list[dict[str, Any]] = field(default_factory=list)


def build_replay_context(cassette_path: str, mode: Mode) -> ReplayContext:
    """`record` starts from an empty Cassette (whatever's on disk gets fully
    overwritten, per the modes table) -- `replay`/`record_missing` load the
    existing file first, since both need to look entries up. `live` never reads
    or writes a cassette at all, but still gets a (never-consulted) empty one so
    callers don't need a None-check for this one mode specifically.

    Shared by `frameworks/langgraph/runner.py` (one context spans a whole,
    possibly paused-and-resumed, graph run) and `judge/client.py`'s `grade()`
    (one context per judge call) -- the load/save rules are identical either way.
    """
    cassette = load_cassette(cassette_path) if mode in ("replay", "record_missing") else Cassette()
    return ReplayContext(mode=mode, cassette=cassette)


def save_if_recording(cassette_path: str, mode: Mode, ctx: ReplayContext) -> None:
    """Write the cassette back out after a run, for the two modes that produce
    new entries. `record_missing` keeps whatever already matched plus whatever
    was newly recorded; `record` is *only* whatever was newly recorded this run
    -- matching the modes table's "Overwritten" (`record`) vs "Appended"
    (`record_missing`) distinction exactly.
    """
    if mode not in ("record", "record_missing") or not ctx.new_entries:
        return
    final = Cassette(
        entries=ctx.cassette.entries + ctx.new_entries,
        version=ctx.cassette.version,
        graphtrace=ctx.cassette.graphtrace,
        recorded_at=ctx.cassette.recorded_at,
    )
    save_cassette(final, cassette_path)
