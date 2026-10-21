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


def _run_one_stream(graph: Any, stream_input: Any, config: dict, recorder: Recorder, buffer: CallBuffer) -> None:
    """Drive exactly one graph.stream(...) call to exhaustion, feeding every debug
    event to `recorder`. Shared by the sync and async paths below (langgraph's sync
    .stream() is what both ultimately use -- there's no async debug-event API
    different enough to need its own event handling, only the surrounding await).
    """
    with intercept(buffer):
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
) -> Trajectory:
    """Run `graph` once (plus once more per queued resume value, if it pauses on a
    human-in-the-loop interrupt), and return the resulting Trajectory.

    `cassette` is accepted now, to match the API this will have once record/replay
    exists (Week 2) -- it is **not used yet**. Every call today is effectively
    `live` mode: real node execution, real LLM/tool calls if the graph makes any.
    This is a deliberate, temporary scope limit for Week 1, not a silent gap --
    see Plan/graphtrace.md section 7 and docs/guide/recording-and-replay.md.

    `resume` is a list of values fed to successive interrupts, one per pause, in
    order. If the graph pauses and there's no next resume value queued, run_graph
    stops there and returns a Trajectory with `interrupted=True` -- exactly what a
    test asserting `assert_interrupted_at` is checking for.
    """
    _ensure_checkpointer(graph)
    run_config = {"configurable": {"thread_id": thread_id, **(config or {})}}
    recorder = Recorder()
    buffer = CallBuffer()
    pending_resumes = list(resume) if resume else []

    stream_input: Any = inputs
    while True:
        _run_one_stream(graph, stream_input, run_config, recorder, buffer)

        state = graph.get_state(run_config)
        recorder.flush(state.values)
        if not state.next:
            return Trajectory(steps=recorder.steps, final_state=state.values, interrupted=False, mode="live")

        if not pending_resumes:
            return Trajectory(steps=recorder.steps, final_state=state.values, interrupted=True, mode="live")

        stream_input = Command(resume=pending_resumes.pop(0))


async def arun_graph(
    graph: Any,
    inputs: Any,
    *,
    cassette: str,
    config: dict | None = None,
    thread_id: str = "gc-test",
    resume: list[Any] | None = None,
) -> Trajectory:
    """Async sibling of run_graph() -- same contract, drives graph.astream()/async
    interceptor paths instead. Most real LangGraph agents compile async nodes (async
    tool calls, async chat model _agenerate), so this isn't an edge case kept for
    symmetry; it's the path most real callers actually need.
    """
    _ensure_checkpointer(graph)
    run_config = {"configurable": {"thread_id": thread_id, **(config or {})}}
    recorder = Recorder()
    buffer = CallBuffer()
    pending_resumes = list(resume) if resume else []

    stream_input: Any = inputs
    while True:
        with intercept(buffer):
            async for namespace, _mode, event in graph.astream(
                stream_input, run_config, stream_mode=["debug"], subgraphs=True
            ):
                recorder.on_debug_event(namespace, event, buffer)

        state = await graph.aget_state(run_config)
        recorder.flush(state.values)
        if not state.next:
            return Trajectory(steps=recorder.steps, final_state=state.values, interrupted=False, mode="live")

        if not pending_resumes:
            return Trajectory(steps=recorder.steps, final_state=state.values, interrupted=True, mode="live")

        stream_input = Command(resume=pending_resumes.pop(0))
