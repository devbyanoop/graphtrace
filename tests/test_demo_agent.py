"""The Week 1 goal check: run_graph() against a real, realistic-shaped agent
(examples/refund_agent), asserted against with the real, public assert_* API --
not a hand-built Trajectory fixture. This is what proves the recorder/interceptor
layer and the assertion layer actually fit together end to end.

Positive (happy path), negative (rejection / gives-up-after-failure), and edge
cases (retry, interrupt-without-resume) are grouped below, since that's the
coverage asked for.
"""

from __future__ import annotations

import itertools

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from refund_agent.graph import build_refund_graph

from graphtrace import (
    assert_budget,
    assert_interrupted_at,
    assert_path,
    assert_retries,
    assert_state,
    assert_state_at,
    assert_tool_called,
    assert_tool_calls_match,
    assert_visited,
    run_graph,
)


def _fake_model() -> GenericFakeChatModel:
    # itertools.repeat, not a one-shot iter([...]) -- the router node may run more
    # than once across a test (e.g. after a resume), and a GenericFakeChatModel's
    # message iterator doesn't reset itself between calls.
    return GenericFakeChatModel(messages=itertools.repeat(AIMessage(content="ok")))


# -- positive: happy path ---------------------------------------------------------------

def test_refund_flow_happy_path_full_approval():
    graph = build_refund_graph(_fake_model())
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[True])

    assert traj.interrupted is False
    assert_path(traj, ["router", "lookup_order", "human_approval", "refund"])
    assert_tool_called(traj, "get_order", args_match={"order_id": "1234"})
    assert_state(traj, "refunded", True)
    assert_state_at(traj, "router", "intent", "refund")


def test_refund_flow_llm_and_tool_calls_stay_within_budget():
    graph = build_refund_graph(_fake_model())
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[True])
    # No real network call anywhere in this run (fake model, in-process tool), so
    # a generous budget is really checking "did this run at all", not tuning a
    # real threshold -- that tuning is a client's own job, not this test's.
    assert_budget(traj, latency_s=5)


def test_refund_flow_tool_calls_match_reference_strictly():
    graph = build_refund_graph(_fake_model())
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[True])
    assert_tool_calls_match(traj, ["get_order"], mode="strict")


# -- edge cases: retry, interrupt without resume ------------------------------------------

def test_refund_flow_retries_lookup_once_then_succeeds():
    graph = build_refund_graph(_fake_model(), fail_lookup_times=1)
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[True])

    assert traj.interrupted is False
    assert traj.visits("lookup_order") == 2
    assert_retries(traj, "lookup_order", max=1)
    assert_state(traj, "refunded", True)
    # The failed first attempt is real recorded data, not hidden -- its tool call
    # shows the actual error the node's own try/except caught.
    first_attempt = [s for s in traj.steps if s.node == "lookup_order"][0]
    assert first_attempt.tool_calls[0].error == "order service unavailable (attempt 1)"


def test_refund_flow_pauses_for_human_approval_without_resume():
    graph = build_refund_graph(_fake_model())
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused")  # no resume given

    assert traj.interrupted is True
    assert_interrupted_at(traj, "human_approval")
    assert_visited(traj, "lookup_order")
    # The pause happens before refund -- a test asserting the interrupt actually
    # gates the refund, not just that *some* interrupt happened somewhere.
    assert "refund" not in traj.path


# -- negative: rejection and giving up ------------------------------------------------------

def test_refund_flow_rejected_by_human_does_not_refund():
    graph = build_refund_graph(_fake_model())
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[False])

    assert traj.interrupted is False
    assert_path(traj, ["router", "lookup_order", "human_approval", "give_up"])
    assert_state(traj, "approved", False)
    assert_state(traj, "refunded", False)


def test_refund_flow_gives_up_after_too_many_lookup_failures():
    graph = build_refund_graph(_fake_model(), fail_lookup_times=10, max_lookup_retries=2)
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[True])

    assert traj.interrupted is False
    assert_path(traj, ["router", ..., "give_up"])
    assert "human_approval" not in traj.path
    assert_state(traj, "refunded", False)
    assert traj.visits("lookup_order") == 3  # 1 initial + max_lookup_retries(=2) retries


def test_refund_flow_different_thread_ids_dont_cross_contaminate():
    """Two independent run_graph() calls against the *same compiled graph object*
    must not leak state between them -- a real risk given the graph is reused."""
    graph = build_refund_graph(_fake_model())
    traj_1 = run_graph(graph, {"order_id": "AAA"}, cassette="unused", thread_id="t1", resume=[True])
    traj_2 = run_graph(graph, {"order_id": "BBB"}, cassette="unused", thread_id="t2", resume=[True])

    assert_tool_called(traj_1, "get_order", args_match={"order_id": "AAA"})
    assert_tool_called(traj_2, "get_order", args_match={"order_id": "BBB"})


def test_refund_flow_router_llm_call_is_attributed_to_router_node():
    graph = build_refund_graph(_fake_model())
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[True])

    assert len(traj.llm_calls) == 1
    assert traj.llm_calls[0].node == "router"
