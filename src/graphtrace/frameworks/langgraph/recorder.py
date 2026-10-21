"""Turn LangGraph's own debug-mode event stream into a list of Step objects.

Verified against a real compiled graph (langgraph 1.2, langchain-core 1.6) before
writing any of this, not assumed from the architecture plan's description -- the
plan called for `stream_mode=["debug", "updates"]`, but "updates" events turned out
to duplicate exactly what "debug"'s own task_result.result already carries, so this
recorder subscribes to "debug" alone. If a future LangGraph release changes these
event shapes, this is the one file that needs to change -- see this project's own
layering rule in Plan/graphtrace.md section 3.

The three debug event types this recorder actually uses:
  - "task":        a node is about to run. payload: {id, name, input, triggers}
  - "task_result": a node finished. payload: {id, name, error, result, interrupts}
  - "checkpoint":  the graph's full, authoritative state after a step settles.
                   payload: {values, next, ...}

Why state_before/state_after come from "checkpoint", not from accumulating
task_result's own writes by hand: "checkpoint".values is LangGraph's own, fully
reduced state (it already applied whatever reducer each state key uses -- a plain
replace, operator.add, add_messages, ...). Recomputing that ourselves from raw
writes would mean re-implementing LangGraph's own reducer logic, badly, and getting
it wrong for any key that isn't a plain dict-update. Using checkpoint.values instead
means graphtrace never has to know or care which reducer a project's state uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from graphtrace.core.trajectory import Step
from graphtrace.frameworks.langgraph.interceptors import CallBuffer


@dataclass
class Recorder:
    """Consumes one graph.stream(..., stream_mode=["debug"], subgraphs=True) call's
    worth of events and produces the Step objects for a Trajectory.

    One Recorder instance is reused across a paused-then-resumed run (see runner.py)
    so a cut-short-by-interrupt run and its resume both contribute Steps to the same
    continuous `steps` list, in order.
    """

    steps: list[Step] = field(default_factory=list)
    last_values: dict[str, Any] = field(default_factory=dict)
    """The most recent checkpoint's full state -- runner.py reads this as the
    Trajectory's final_state once the run is done or paused."""
    _pending: dict[str, tuple[Step, float]] = field(default_factory=dict)  # task id -> (Step, start_time)
    _awaiting_state_after: list[tuple[str, Step]] = field(default_factory=list)  # (task id, Step)
    _committed_by_id: dict[str, Step] = field(default_factory=dict)  # task id -> Step already in `steps`
    _step_index: int = 0

    def on_debug_event(self, namespace: tuple[str, ...], event: dict[str, Any], buffer: CallBuffer) -> None:
        kind = event.get("type")
        if kind == "task":
            self._on_task(namespace, event["payload"])
        elif kind == "task_result":
            self._on_task_result(event["payload"], buffer)
        elif kind == "checkpoint":
            self._on_checkpoint(event["payload"])
        # Any other debug event type (LangGraph may add more over time) is simply
        # not one this recorder needs -- ignored, not an error.

    def flush(self, current_values: dict[str, Any]) -> None:
        """Commit any step still waiting for a checkpoint that's never coming.

        This is the case where a run paused on an interrupt right after a task's
        task_result -- there's no settling checkpoint after it (confirmed by hand:
        the real debug-event stream for a paused run simply ends there). Call once
        after a stream() call is fully exhausted, with the graph's current state
        (from `get_state(config).values`) as the best-known state_after.
        """
        for task_id, step in self._awaiting_state_after:
            step.state_after = dict(current_values)
            self.steps.append(step)
            self._committed_by_id[task_id] = step
        self._awaiting_state_after.clear()

    def _on_task(self, namespace: tuple[str, ...], payload: dict[str, Any]) -> None:
        task_id = payload["id"]
        if task_id in self._committed_by_id:
            # This exact task id was already committed once -- by flush(), because
            # its first attempt paused on an interrupt. Resuming re-runs the node
            # function from the top with the *same* task id (confirmed by hand), so
            # without this, the interrupted attempt's incomplete Step would sit in
            # `steps` alongside the resumed, complete one as a duplicate. Drop the
            # stale one; the resumed attempt replaces it.
            stale = self._committed_by_id.pop(task_id)
            self.steps.remove(stale)

        step = Step(
            index=self._step_index,
            node=payload["name"],
            kind="node",
            state_before=dict(self.last_values),
            state_after={},
            state_diff={},
            namespace=namespace,
        )
        self._step_index += 1
        self._pending[task_id] = (step, time.perf_counter())

    def _on_task_result(self, payload: dict[str, Any], buffer: CallBuffer) -> None:
        task_id = payload["id"]
        step, start_time = self._pending.pop(task_id)
        step.latency_s = time.perf_counter() - start_time
        step.state_diff = payload.get("result") or {}
        step.llm_calls, step.tool_calls = buffer.pop(task_id)
        if payload.get("interrupts"):
            step.kind = "interrupt"
        # Not appended to `steps` yet -- state_after isn't known until the
        # checkpoint that settles this step arrives (see _on_checkpoint), or until
        # flush() decides no such checkpoint is coming.
        self._awaiting_state_after.append((task_id, step))

    def _on_checkpoint(self, payload: dict[str, Any]) -> None:
        values = payload.get("values") or {}
        for task_id, step in self._awaiting_state_after:
            step.state_after = dict(values)
            self.steps.append(step)
            self._committed_by_id[task_id] = step
        self._awaiting_state_after.clear()
        self.last_values = values
