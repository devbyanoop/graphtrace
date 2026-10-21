"""run_graph() / arun_graph(): drive a compiled LangGraph and return a Trajectory.

Implements `frameworks.base.FrameworkRunner` (see that module for why the Protocol
exists). This is the only place in the v0.1 codebase that actually calls
`graph.stream()`/`astream()` -- everything upstream of here (recorder.py,
interceptors.py) only reacts to what this function feeds it.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graphtrace.core.trajectory import Trajectory
from graphtrace.frameworks.langgraph.interceptors import CallBuffer, intercept
from graphtrace.frameworks.langgraph.recorder import Recorder
from graphtrace.replay.cassette import Cassette, load_cassette, save_cassette
from graphtrace.replay.context import ReplayContext
from graphtrace.replay.modes import resolve_mode


def _ensure_checkpointer(graph: Any) -> None:
    """Attach a MemorySaver if the caller's graph was compiled without one.

    Interrupts and resumes don't work at all without a checkpointer -- LangGraph
    persists a paused run's state through it, not through anything run_graph could
    hold onto itself between two separate stream() calls. Confirmed by hand that
    `compiled.checkpointer` is a plain, settable attribute post-compile (not baked
    in immutably), so this doesn't require the caller's original StateGraph builder.
    """
    if graph.checkpointer is None:
        graph.checkpointer = MemorySaver()


def _build_replay_context(cassette_path: str, mode: str) -> ReplayContext:
    """`record` starts from an empty Cassette (whatever's on disk gets fully
    overwritten, per the modes table) -- `replay`/`record_missing` load the
    existing file first, since both need to look entries up. `live` never reads
    or writes a cassette at all, but still gets a (never-consulted) empty one so
    the interceptors don't need a None-check for this one mode specifically.
    """
    cassette = load_cassette(cassette_path) if mode in ("replay", "record_missing") else Cassette()
    return ReplayContext(mode=mode, cassette=cassette)


def _save_if_recording(cassette_path: str, mode: str, replay_ctx: ReplayContext) -> None:
    """Write the cassette back out after the run, for the two modes that produce
    new entries. `record_missing` keeps whatever already matched plus whatever
    was newly recorded; `record` is *only* whatever was newly recorded this run --
    matching the modes table's "Overwritten" (`record`) vs "Appended"
    (`record_missing`) distinction exactly.
    """
    if mode not in ("record", "record_missing") or not replay_ctx.new_entries:
        return
    final = Cassette(
        entries=replay_ctx.cassette.entries + replay_ctx.new_entries,
        version=replay_ctx.cassette.version,
        graphtrace=replay_ctx.cassette.graphtrace,
        recorded_at=replay_ctx.cassette.recorded_at,
    )
    save_cassette(final, cassette_path)


def _run_one_stream(graph: Any, stream_input: Any, config: dict, recorder: Recorder, buffer: CallBuffer, replay_ctx: ReplayContext) -> None:
    """Drive exactly one graph.stream(...) call to exhaustion, feeding every debug
    event to `recorder`. Shared by the sync and async paths below (langgraph's sync
    .stream() is what both ultimately use -- there's no async debug-event API
    different enough to need its own event handling, only the surrounding await).
    """
    with intercept(buffer, replay_ctx):
        for namespace, _mode, event in graph.stream(
            stream_input, config, stream_mode=["debug"], subgraphs=True
        ):
            recorder.on_debug_event(namespace, event, buffer)


def run_graph(
    graph: Any,
    inputs: Any,
    *,
    cassette: str,
    config: dict | None = None,
    thread_id: str = "gc-test",
    resume: list[Any] | None = None,
    mode: str | None = None,
) -> Trajectory:
    """Run `graph` once (plus once more per queued resume value, if it pauses on a
    human-in-the-loop interrupt), and return the resulting Trajectory.

    `mode` resolves via `replay.modes.resolve_mode`: an explicit argument here
    wins, otherwise the `GRAPHTRACE_MODE` environment variable, otherwise `"live"`
    -- unchanged Week 1 behaviour unless something actually opts into record/replay.

    `resume` is a list of values fed to successive interrupts, one per pause, in
    order. If the graph pauses and there's no next resume value queued, run_graph
    stops there and returns a Trajectory with `interrupted=True` -- exactly what a
    test asserting `assert_interrupted_at` is checking for.
    """
    resolved_mode = resolve_mode(mode)
    _ensure_checkpointer(graph)
    run_config = {"configurable": {"thread_id": thread_id, **(config or {})}}
    recorder = Recorder()
    buffer = CallBuffer()
    replay_ctx = _build_replay_context(cassette, resolved_mode)
    pending_resumes = list(resume) if resume else []

    stream_input: Any = inputs
    while True:
        _run_one_stream(graph, stream_input, run_config, recorder, buffer, replay_ctx)

        state = graph.get_state(run_config)
        recorder.flush(state.values)
        if not state.next:
            _save_if_recording(cassette, resolved_mode, replay_ctx)
            return Trajectory(steps=recorder.steps, final_state=state.values, interrupted=False, mode=resolved_mode)

        if not pending_resumes:
            _save_if_recording(cassette, resolved_mode, replay_ctx)
            return Trajectory(steps=recorder.steps, final_state=state.values, interrupted=True, mode=resolved_mode)

        stream_input = Command(resume=pending_resumes.pop(0))


async def arun_graph(
    graph: Any,
    inputs: Any,
    *,
    cassette: str,
    config: dict | None = None,
    thread_id: str = "gc-test",
    resume: list[Any] | None = None,
    mode: str | None = None,
) -> Trajectory:
    """Async sibling of run_graph() -- same contract, drives graph.astream()/async
    interceptor paths instead. Most real LangGraph agents compile async nodes (async
    tool calls, async chat model _agenerate), so this isn't an edge case kept for
    symmetry; it's the path most real callers actually need.
    """
    resolved_mode = resolve_mode(mode)
    _ensure_checkpointer(graph)
    run_config = {"configurable": {"thread_id": thread_id, **(config or {})}}
    recorder = Recorder()
    buffer = CallBuffer()
    replay_ctx = _build_replay_context(cassette, resolved_mode)
    pending_resumes = list(resume) if resume else []

    stream_input: Any = inputs
    while True:
        with intercept(buffer, replay_ctx):
            async for namespace, _mode, event in graph.astream(
                stream_input, run_config, stream_mode=["debug"], subgraphs=True
            ):
                recorder.on_debug_event(namespace, event, buffer)

        state = await graph.aget_state(run_config)
        recorder.flush(state.values)
        if not state.next:
            _save_if_recording(cassette, resolved_mode, replay_ctx)
            return Trajectory(steps=recorder.steps, final_state=state.values, interrupted=False, mode=resolved_mode)

        if not pending_resumes:
            _save_if_recording(cassette, resolved_mode, replay_ctx)
            return Trajectory(steps=recorder.steps, final_state=state.values, interrupted=True, mode=resolved_mode)

        stream_input = Command(resume=pending_resumes.pop(0))
