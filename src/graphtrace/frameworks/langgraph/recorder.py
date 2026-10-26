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

## Subgraphs (confirmed by hand, not assumed from the original design)

A node whose value is itself a compiled graph produces its own, fully nested
"task"/"task_result"/"checkpoint" stream, tagged with a deeper namespace tuple
(e.g. `("sub:<outer_task_id>",)`) -- confirmed with a real two-level graph and a
real tool call inside the inner node before writing any of this. Two real bugs
were found and are fixed here, not assumed away:

1. **State clobbering across namespaces, confirmed to land on `state_after`,
   not `state_before`.** `last_values`/the pending-task bookkeeping were
   module-wide, single-namespace state. `state_before` turned out to be safe
   regardless -- it's captured once, at `_on_task` time, before any later
   event (from any namespace) can touch it. The real, confirmed corruption is
   in `_on_checkpoint`'s old unconditional "commit everything in
   `_awaiting_state_after`" behavior: when an outer-level node runs
   *concurrently* with a subgraph (parallel fan-out), the subgraph's own
   inner checkpoint -- a *different* graph's state entirely -- would commit
   the sibling node's `state_after` early, using the subgraph's partial
   state instead of the outer graph's own. Reproduced by hand with a graph
   that fans out into a plain node and a subgraph at once: the plain
   sibling's `state_after` came back as the subgraph's own `{"y": 0}`,
   nothing of the outer state at all. A purely sequential subgraph never hit
   this (the outer graph's own next checkpoint always arrives later and
   corrects `last_values` before anything reads it again), which is exactly
   why it went unnoticed until deliberately tested. Fixed by scoping every
   piece of this bookkeeping (`last_values`, pending tasks, awaiting-commit
   list, committed-by-id) per namespace -- the outer namespace `()` behaves
   identically to before (it's the only namespace that ever existed prior to
   subgraphs), so this is a pure addition, not a behavior change, for every
   graph without one.
2. **`Step.kind`'s `"subgraph_enter"`/`"subgraph_exit"` were declared in the
   type but never actually produced.** A subgraph-wrapping node's own
   task/task_result pair looks identical in shape to an ordinary node's --
   confirmed the only way to know it wraps a subgraph is observing a *child*
   event arrive at a namespace that extends it. The alternative -- treating the
   whole subgraph as one opaque "node" step -- was considered and rejected:
   confirmed by hand that calls made *inside* a subgraph's own node are
   attributed to that inner node's own task id, never to the wrapping node's,
   so collapsing it to one step would silently drop those LLM/tool calls from
   the trajectory entirely, which is the one thing this project exists to
   capture faithfully. Instead: a `subgraph_enter` Step is emitted the moment a
   child event reveals the wrapping task is a subgraph (so it appears *before*
   its children in the path, not after), and a `subgraph_exit` Step replaces
   the ordinary "node" commit when that wrapping task's own task_result/
   checkpoint finally arrive (necessarily last, since a subgraph's wrapping
   task genuinely can't be known done before its children are).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from graphtrace.core.trajectory import Step
from graphtrace.frameworks.langgraph.interceptors import CallBuffer

_Namespace = tuple[str, ...]


@dataclass
class Recorder:
    """Consumes one graph.stream(..., stream_mode=["debug"], subgraphs=True) call's
    worth of events and produces the Step objects for a Trajectory.

    One Recorder instance is reused across a paused-then-resumed run (see runner.py)
    so a cut-short-by-interrupt run and its resume both contribute Steps to the same
    continuous `steps` list, in order.
    """

    steps: list[Step] = field(default_factory=list)
    _last_values: dict[_Namespace, dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    _pending: dict[_Namespace, dict[str, tuple[Step, float]]] = field(default_factory=lambda: defaultdict(dict))
    _awaiting_state_after: dict[_Namespace, list[tuple[str, Step]]] = field(default_factory=lambda: defaultdict(list))
    _committed_by_id: dict[_Namespace, dict[str, Step]] = field(default_factory=lambda: defaultdict(dict))
    _subgraph_task_ids: set[tuple[_Namespace, str]] = field(default_factory=set)
    """(namespace, task_id) pairs confirmed -- by a child event actually
    arriving -- to wrap a subgraph, so their eventual task_result is recorded
    as "subgraph_exit", not an ordinary "node" commit."""
    _step_index: int = 0

    @property
    def last_values(self) -> dict[str, Any]:
        """The outer graph's most recent checkpoint state -- the only namespace
        anything outside this class ever cared about before subgraphs existed,
        kept as a plain property so that external shape doesn't change."""
        return self._last_values[()]

    def on_debug_event(self, namespace: _Namespace, event: dict[str, Any], buffer: CallBuffer) -> None:
        kind = event.get("type")
        if kind == "task":
            self._on_task(namespace, event["payload"])
        elif kind == "task_result":
            self._on_task_result(namespace, event["payload"], buffer)
        elif kind == "checkpoint":
            self._on_checkpoint(namespace, event["payload"])
        # Any other debug event type (LangGraph may add more over time) is simply
        # not one this recorder needs -- ignored, not an error.

    def flush(self, current_values: dict[str, Any]) -> None:
        """Commit any step still waiting for a checkpoint that's never coming.

        This is the case where a run paused on an interrupt right after a task's
        task_result -- there's no settling checkpoint after it (confirmed by hand:
        the real debug-event stream for a paused run simply ends there). Call once
        after a stream() call is fully exhausted, with the graph's current state
        (from `get_state(config).values`) as the best-known state_after.

        Only the outer namespace is flushed with the caller's `current_values` --
        an interrupt genuinely can only pause the *outer* run between stream()
        calls (resuming re-enters via a fresh top-level stream() call either
        way), so a still-pending inner-namespace step here would mean something
        this recorder doesn't yet model, not a case to guess a value for.
        """
        for task_id, step in self._awaiting_state_after[()]:
            step.state_after = dict(current_values)
            self.steps.append(step)
            self._committed_by_id[()][task_id] = step
        self._awaiting_state_after[()].clear()

    def _on_task(self, namespace: _Namespace, payload: dict[str, Any]) -> None:
        task_id = payload["id"]
        node_name = payload["name"]

        # Does this task's own namespace reveal that some *pending parent* task
        # wraps a subgraph? LangGraph tags a subgraph's inner namespace as the
        # parent's own namespace plus exactly one new segment, "{parent_node}:
        # {parent_task_id}" -- confirmed by hand. Checked on every task event,
        # not only the first, since a subgraph can contain more than one node.
        #
        # The parent's own *pending* step -- already correctly indexed to when
        # its wrapping task actually started -- becomes the enter marker,
        # emitted right away; a brand-new step, indexed *now* (always later,
        # since detection can only happen after the parent started), takes its
        # place in `_pending` to eventually become the exit marker. Found and
        # fixed a real bug from the original version of this logic (confirmed
        # against a real subgraph-using external app, langgraph-swarm-py):
        # creating the *enter* step fresh here instead gave it a higher index
        # than the exit step (which kept the parent's original, earlier one)
        # -- exit showing a lower number than its own enter, backwards from
        # what actually happened.
        if namespace:
            parent_ns, marker = namespace[:-1], namespace[-1]
            for parent_task_id, (parent_step, parent_start) in self._pending[parent_ns].items():
                if marker == f"{parent_step.node}:{parent_task_id}" and (parent_ns, parent_task_id) not in self._subgraph_task_ids:
                    self._subgraph_task_ids.add((parent_ns, parent_task_id))
                    parent_step.kind = "subgraph_enter"
                    parent_step.state_after = dict(parent_step.state_before)
                    self.steps.append(parent_step)
                    exit_step = Step(
                        index=self._step_index,
                        node=parent_step.node,
                        kind="node",  # becomes "subgraph_exit" in _on_task_result, same as any subgraph-wrapping task
                        state_before=dict(parent_step.state_before),
                        state_after={},
                        state_diff={},
                        namespace=parent_ns,
                    )
                    self._step_index += 1
                    self._pending[parent_ns][parent_task_id] = (exit_step, parent_start)
                    break

        if task_id in self._committed_by_id[namespace]:
            # This exact task id was already committed once -- by flush(), because
            # its first attempt paused on an interrupt. Resuming re-runs the node
            # function from the top with the *same* task id (confirmed by hand), so
            # without this, the interrupted attempt's incomplete Step would sit in
            # `steps` alongside the resumed, complete one as a duplicate. Drop the
            # stale one; the resumed attempt replaces it.
            stale = self._committed_by_id[namespace].pop(task_id)
            self.steps.remove(stale)

        step = Step(
            index=self._step_index,
            node=node_name,
            kind="node",
            state_before=dict(self._last_values[namespace]),
            state_after={},
            state_diff={},
            namespace=namespace,
        )
        self._step_index += 1
        self._pending[namespace][task_id] = (step, time.perf_counter())

    def _on_task_result(self, namespace: _Namespace, payload: dict[str, Any], buffer: CallBuffer) -> None:
        task_id = payload["id"]
        step, start_time = self._pending[namespace].pop(task_id)
        step.latency_s = time.perf_counter() - start_time
        step.state_diff = payload.get("result") or {}
        step.llm_calls, step.tool_calls = buffer.pop(task_id)
        if payload.get("interrupts"):
            step.kind = "interrupt"
        elif (namespace, task_id) in self._subgraph_task_ids:
            step.kind = "subgraph_exit"
        # Not appended to `steps` yet -- state_after isn't known until the
        # checkpoint that settles this step arrives (see _on_checkpoint), or until
        # flush() decides no such checkpoint is coming.
        self._awaiting_state_after[namespace].append((task_id, step))

        # A real gap found and fixed testing against a real external app
        # (langchain-ai/langgraph-swarm-py's handoff tools): a tool whose real
        # call returns `Command(graph=Command.PARENT, ...)` (LangGraph's own
        # mechanism for a tool to redirect control up to the parent graph)
        # never gets its *own* task_result/checkpoint at all -- confirmed by
        # hand against the real debug-event stream, its completion is folded
        # directly into the *parent's* task_result instead, with no settling
        # point for the child in between. Without this, that child's step
        # would simply never commit (unlike the flush() case this project
        # already handles, there is no later event that will ever resolve
        # it) -- silently dropping a real, already-buffered tool call from
        # the trajectory, even though the call itself was captured correctly
        # by the interceptor. Absorbed here: any task still pending directly
        # under *this* task's own namespace, the instant this one resolves,
        # since the parent resolving is the only signal that will ever come.
        # state_after is conservatively left equal to state_before -- the
        # real state change already landed in the *parent's* own result, and
        # there's no cleanly attributable "state right after just the child"
        # to show instead of guessing one.
        child_ns = namespace + (f"{step.node}:{task_id}",)
        if child_ns in self._pending:
            for child_task_id, (child_step, child_start) in self._pending[child_ns].items():
                child_step.latency_s = time.perf_counter() - child_start
                child_step.llm_calls, child_step.tool_calls = buffer.pop(child_task_id)
                child_step.state_after = dict(child_step.state_before)
                self.steps.append(child_step)
                self._committed_by_id[child_ns][child_task_id] = child_step
            del self._pending[child_ns]

    def _on_checkpoint(self, namespace: _Namespace, payload: dict[str, Any]) -> None:
        values = payload.get("values") or {}
        for task_id, step in self._awaiting_state_after[namespace]:
            step.state_after = dict(values)
            self.steps.append(step)
            self._committed_by_id[namespace][task_id] = step
        self._awaiting_state_after[namespace].clear()
        self._last_values[namespace] = values
