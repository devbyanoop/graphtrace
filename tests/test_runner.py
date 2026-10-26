"""Tests for run_graph()/arun_graph() against real, tiny compiled LangGraphs --
not the hand-built Trajectory fixtures test_assertions.py uses. This is the layer
that has to prove graphtrace can turn an actual LangGraph run into a Trajectory
correctly: linear paths, LLM/tool call capture, parallel fan-out, loops, interrupts,
and errors. Grouped by positive (happy path), negative (error handling), and edge
cases, since that's the coverage asked for.

Every graph here is deliberately tiny and built inline -- the point of this file is
pinning down recorder/interceptor correctness precisely, not exercising a realistic
agent shape (examples/refund_agent + test_demo_agent.py do that).
"""

from __future__ import annotations

import asyncio
from typing import Annotated, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from graphtrace import arun_graph, assert_budget, run_graph


# -- small graph builders, shared across several tests --------------------------------

class CounterState(TypedDict):
    x: int


def _linear_graph():
    def a(state):
        return {"x": state["x"] + 1}

    def b(state):
        return {"x": state["x"] + 10}

    g = StateGraph(CounterState)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile()


def _add_log(a: list, b: list) -> list:
    return a + b


class FanOutState(TypedDict):
    log: Annotated[list, _add_log]


def _fanout_graph():
    def a(state):
        return {"log": ["a"]}

    def b(state):
        return {"log": ["b"]}

    def c(state):
        return {"log": ["c"]}

    g = StateGraph(FanOutState)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_node("c", c)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", END)
    g.add_edge("c", END)
    return g.compile()


# -- positive: happy path ---------------------------------------------------------------

def test_run_graph_linear_path_records_correct_steps_and_state():
    traj = run_graph(_linear_graph(), {"x": 0}, cassette="unused")

    assert traj.path == ["a", "b"]
    assert traj.interrupted is False
    assert traj.final_state == {"x": 11}

    step_a, step_b = traj.steps
    # state_before for the very first node is the real input state, {"x": 0} --
    # not {} -- confirmed by hand: LangGraph emits an initial checkpoint carrying
    # the input before any node runs, which this recorder uses the same way it
    # uses every later checkpoint.
    assert step_a.state_before == {"x": 0}
    assert step_a.state_diff == {"x": 1}
    assert step_a.state_after == {"x": 1}
    assert step_b.state_before == {"x": 1}
    assert step_b.state_diff == {"x": 11}
    assert step_b.state_after == {"x": 11}


def test_run_graph_captures_llm_call_with_tokens_and_cost():
    class State(TypedDict):
        x: int

    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="hi", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})])
    )

    def call_model(state, config):
        model.invoke("hello", config)
        return {"x": 1}

    g = StateGraph(State)
    g.add_node("call_model", call_model)
    g.add_edge(START, "call_model")
    g.add_edge("call_model", END)
    traj = run_graph(g.compile(), {"x": 0}, cassette="unused")

    assert len(traj.llm_calls) == 1
    call = traj.llm_calls[0]
    assert call.node == "call_model"
    assert call.prompt_tokens == 10
    assert call.completion_tokens == 5
    assert call.output.content == "hi"
    # gpt-4o-mini isn't the model name here (GenericFakeChatModel has none that
    # matches the pricing table), so cost falls back to 0.0 -- confirms the
    # "unrecognized model -> 0.0, not a guess" rule from frameworks/pricing.py.
    assert call.cost_usd == 0.0


def test_run_graph_captures_tool_call_via_toolnode():
    class State(TypedDict):
        messages: Annotated[list, _add_log]

    @tool
    def get_order(order_id: str) -> dict:
        """Look up an order."""
        return {"status": "delivered", "amount": 49.0}

    def call_model(state):
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "get_order", "args": {"order_id": "1234"}, "id": "call_1"}])]}

    g = StateGraph(State)
    g.add_node("call_model", call_model)
    g.add_node("tools", ToolNode([get_order]))
    g.add_edge(START, "call_model")
    g.add_edge("call_model", "tools")
    g.add_edge("tools", END)
    traj = run_graph(g.compile(), {"messages": []}, cassette="unused")

    assert len(traj.tool_calls) == 1
    call = traj.tool_calls[0]
    assert call.node == "tools"
    assert call.name == "get_order"
    assert call.args == {"order_id": "1234"}
    assert call.error is None
    # Invoking a tool with a ToolCall-shaped input (what ToolNode always passes)
    # makes BaseTool.invoke() return a ToolMessage wrapping the real content, not
    # the tool function's raw return value -- confirmed by hand, not assumed.
    # ToolCall.result is whatever invoke() actually returned, unmodified: the
    # interceptor doesn't try to unwrap/guess at a "real" result underneath it.
    assert call.result.content == '{"status": "delivered", "amount": 49.0}'


def test_run_graph_auto_attaches_checkpointer_when_graph_has_none():
    graph = _linear_graph()
    assert graph.checkpointer is None
    run_graph(graph, {"x": 0}, cassette="unused")
    assert graph.checkpointer is not None


def test_arun_graph_linear_path():
    # Plain asyncio.run() rather than pytest-asyncio -- one async test doesn't
    # justify a new dev dependency; add pytest-asyncio for real if async coverage
    # grows enough that this stops being convenient.
    traj = asyncio.run(arun_graph(_linear_graph(), {"x": 0}, cassette="unused"))
    assert traj.path == ["a", "b"]
    assert traj.final_state == {"x": 11}


# -- edge cases ---------------------------------------------------------------------------

def test_run_graph_parallel_fanout_records_both_branches_as_separate_steps():
    traj = run_graph(_fanout_graph(), {"log": []}, cassette="unused")

    assert traj.path[0] == "a"
    assert sorted(traj.path[1:]) == ["b", "c"]
    assert traj.visits("b") == 1
    assert traj.visits("c") == 1
    # Both branches of the fan-out see the SAME state_before -- they ran from the
    # same settled point, independent of each other, not chained.
    step_b = next(s for s in traj.steps if s.node == "b")
    step_c = next(s for s in traj.steps if s.node == "c")
    assert step_b.state_before == step_c.state_before == {"log": ["a"]}


def test_run_graph_loop_visits_same_node_multiple_times_with_distinct_steps():
    class State(TypedDict):
        x: int

    def loop_node(state):
        return {"x": state["x"] + 1}

    g = StateGraph(State)
    g.add_node("loop_node", loop_node)
    g.add_edge(START, "loop_node")
    g.add_conditional_edges("loop_node", lambda s: "loop_node" if s["x"] < 3 else END)
    traj = run_graph(g.compile(), {"x": 0}, cassette="unused")

    assert traj.path == ["loop_node", "loop_node", "loop_node"]
    assert traj.visits("loop_node") == 3
    # Three distinct Step objects, not one step counted three times.
    assert len({id(s) for s in traj.steps}) == 3
    assert [s.state_after["x"] for s in traj.steps] == [1, 2, 3]


def test_run_graph_subgraph_node_produces_transparent_enter_exit_and_inner_steps():
    """A node whose value is itself a compiled graph -- confirmed by hand
    (frameworks/langgraph/recorder.py's own docstring has the full trace)
    that its own task/task_result pair looks identical in shape to an
    ordinary node's, and the only way to know it wraps a subgraph is a child
    event arriving at a deeper namespace. The subgraph is NOT collapsed into
    one opaque step -- a real tool call made by its own inner node (`inner_a`
    below) is attributed to that inner node directly (confirmed separately,
    by hand, that langgraph_checkpoint_ns chains as "sub:<id>|inner_a:<id>",
    and rsplit(':', 1) correctly extracts the innermost id regardless) --
    collapsing it would silently drop that call from the trajectory."""

    class InnerState(TypedDict):
        x: int

    @tool
    def add_one(n: int) -> int:
        """Add one to n."""
        return n + 1

    def inner_a(state, config):
        return {"x": add_one.invoke({"n": state["x"]}, config)}

    def inner_b(state):
        return {"x": state["x"] + 10}

    inner = StateGraph(InnerState)
    inner.add_node("inner_a", inner_a)
    inner.add_node("inner_b", inner_b)
    inner.add_edge(START, "inner_a")
    inner.add_edge("inner_a", "inner_b")
    inner.add_edge("inner_b", END)

    class OuterState(TypedDict):
        x: int

    def outer_start(state):
        return {"x": state["x"] + 100}

    def outer_end(state):
        return {"x": state["x"] + 1000}

    outer = StateGraph(OuterState)
    outer.add_node("outer_start", outer_start)
    outer.add_node("sub", inner.compile())
    outer.add_node("outer_end", outer_end)
    outer.add_edge(START, "outer_start")
    outer.add_edge("outer_start", "sub")
    outer.add_edge("sub", "outer_end")
    outer.add_edge("outer_end", END)

    traj = run_graph(outer.compile(), {"x": 0}, cassette="unused")

    # subgraph_enter comes BEFORE its children (eagerly emitted the moment a
    # child reveals the wrapping task is a subgraph), subgraph_exit comes
    # after them (deferred until the wrapping task's own checkpoint, which
    # genuinely can't settle before the children do).
    assert traj.path == ["outer_start", "sub", "inner_a", "inner_b", "sub", "outer_end"]
    kinds = [s.kind for s in traj.steps]
    assert kinds == ["node", "subgraph_enter", "node", "node", "subgraph_exit", "node"]
    assert traj.final_state == {"x": 1111}

    # The real tool call made inside inner_a is attributed to inner_a's own
    # step, not lost, and not misattributed to "sub".
    inner_a_step = next(s for s in traj.steps if s.node == "inner_a")
    assert [c.name for c in inner_a_step.tool_calls] == ["add_one"]
    assert inner_a_step.tool_calls[0].result == 101
    sub_exit_step = next(s for s in traj.steps if s.kind == "subgraph_exit")
    assert sub_exit_step.tool_calls == []

    # subgraph_enter is a true no-op marker: state unchanged at that instant.
    enter_step = next(s for s in traj.steps if s.kind == "subgraph_enter")
    assert enter_step.state_before == enter_step.state_after == {"x": 100}
    # subgraph_exit's state_after reflects everything the subgraph actually did.
    assert sub_exit_step.state_before == {"x": 100}
    assert sub_exit_step.state_after == {"x": 111}


def test_run_graph_subgraph_running_in_parallel_with_another_node_does_not_corrupt_its_state():
    """The real bug this project's own recorder had before being fixed: a
    subgraph's own inner checkpoints are a *different* graph's state
    entirely, and used to overwrite the single, global `last_values` the
    instant they arrived. Harmless for a subgraph running alone (the outer
    graph's own next checkpoint always arrives later and corrects it before
    anything reads it again), but a real risk the moment a subgraph runs
    *concurrently* with another outer-level node, which is exactly this
    scenario -- confirmed to actually matter, not a theoretical worry."""

    class InnerState(TypedDict):
        y: int

    def inner_only(state):
        return {"y": state["y"] + 1}

    inner = StateGraph(InnerState)
    inner.add_node("inner_only", inner_only)
    inner.add_edge(START, "inner_only")
    inner.add_edge("inner_only", END)

    class OuterState(TypedDict):
        start: int
        y: int
        sibling: str

    def start(state):
        return {"start": 1}

    def sibling(state):
        return {"sibling": "ran"}

    outer = StateGraph(OuterState)
    outer.add_node("start", start)
    outer.add_node("sub", inner.compile())
    outer.add_node("sibling", sibling)
    outer.add_edge(START, "start")
    outer.add_edge("start", "sub")
    outer.add_edge("start", "sibling")
    outer.add_edge("sub", END)
    outer.add_edge("sibling", END)

    traj = run_graph(outer.compile(), {"start": 0, "y": 0, "sibling": ""}, cassette="unused")

    # The outer-level "sibling" node's own state_after must come from the
    # OUTER graph's own checkpoint, never from the subgraph's own inner
    # checkpoint ({"y": 0}, an entirely different graph's partial state)
    # committing it early just because both arrived close together. Confirmed
    # by hand this is exactly where the pre-fix bug actually landed (not
    # state_before, which a pending step already captures before any
    # subgraph event can interfere) -- with the bug present, this assertion
    # fails with sibling_step.state_after == {"y": 0}.
    sibling_step = next(s for s in traj.steps if s.node == "sibling")
    assert sibling_step.state_after == {"start": 1, "y": 1, "sibling": "ran"}


def test_run_graph_interrupt_without_resume_returns_interrupted_trajectory():
    class State(TypedDict, total=False):
        approved: bool

    def ask(state):
        approved = interrupt({"question": "approve?"})
        return {"approved": approved}

    g = StateGraph(State)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    traj = run_graph(g.compile(), {}, cassette="unused")

    assert traj.interrupted is True
    assert traj.steps[-1].node == "ask"
    assert traj.steps[-1].kind == "interrupt"


def test_run_graph_interrupt_with_resume_continues_and_completes():
    class State(TypedDict, total=False):
        approved: bool
        refunded: bool

    def ask(state):
        approved = interrupt({"question": "approve?"})
        return {"approved": approved}

    def finish(state):
        return {"refunded": state["approved"]}

    g = StateGraph(State)
    g.add_node("ask", ask)
    g.add_node("finish", finish)
    g.add_edge(START, "ask")
    g.add_edge("ask", "finish")
    g.add_edge("finish", END)
    traj = run_graph(g.compile(), {}, cassette="unused", resume=[True])

    assert traj.interrupted is False
    assert traj.path == ["ask", "finish"]
    assert traj.final_state == {"approved": True, "refunded": True}
    # The interrupted attempt's Step isn't left behind as an orphan duplicate --
    # "ask" appears exactly once in the final path, not twice (once paused, once
    # resumed), even though LangGraph re-runs the node function on resume.
    assert traj.visits("ask") == 1


# -- negative: error handling -------------------------------------------------------------

def test_run_graph_tool_call_error_is_captured_without_crashing_when_node_catches_it():
    class State(TypedDict, total=False):
        ok: bool

    @tool
    def flaky(x: str) -> str:
        """Always fails."""
        raise ValueError("boom")

    def call_tool(state, config):
        try:
            flaky.invoke({"x": "1"}, config)
        except ValueError:
            return {"ok": False}
        return {"ok": True}

    g = StateGraph(State)
    g.add_node("call_tool", call_tool)
    g.add_edge(START, "call_tool")
    g.add_edge("call_tool", END)
    traj = run_graph(g.compile(), {}, cassette="unused")

    assert traj.interrupted is False
    assert traj.final_state == {"ok": False}
    assert len(traj.tool_calls) == 1
    assert traj.tool_calls[0].error == "boom"
    assert traj.tool_calls[0].result is None


def test_run_graph_propagates_uncaught_node_exception():
    class State(TypedDict):
        x: int

    def broken(state):
        raise RuntimeError("node itself is broken")

    g = StateGraph(State)
    g.add_node("broken", broken)
    g.add_edge(START, "broken")
    g.add_edge("broken", END)

    # run_graph must never swallow a real crash into a quietly-wrong Trajectory --
    # a bug in the graph under test should fail the test loudly, the same way an
    # unhandled exception in any other code under test would.
    with pytest.raises(RuntimeError, match="node itself is broken"):
        run_graph(g.compile(), {"x": 0}, cassette="unused")


def test_direct_tool_call_outside_a_graph_run_is_not_recorded():
    """Sanity check on the interceptor's own "nothing to attribute this to, so
    don't record it" rule (see interceptors.py) -- a bare tool.invoke() with no
    LangGraph config must not crash and must not land anywhere."""
    @tool
    def standalone(x: str) -> str:
        """A tool called with no graph around it at all."""
        return x

    # No exception, and nothing to assert on a Trajectory, because there isn't one --
    # this is just confirming the patched invoke() is still a no-op outside a run.
    assert standalone.invoke({"x": "1"}) == "1"


def test_budget_assertion_against_a_real_recorded_trajectory():
    """One assert_* function, exercised against a real run_graph() result instead
    of only the hand-built fixtures in test_assertions.py -- the thing Week 1's
    goal check is actually about: real recorder output, real assertion API."""
    traj = run_graph(_linear_graph(), {"x": 0}, cassette="unused")
    assert_budget(traj, latency_s=5)
