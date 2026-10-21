"""Tests for every assert_* function, exercised against one hand-built Trajectory
that mirrors the refund-agent example from Plan/graphtrace.md:

    router -> lookup_order -> human_approval -> refund

Building the Trajectory by hand (instead of running a real graph) is deliberate for
this stage: it proves the assertion logic is correct in isolation, before the
recorder/runner (built next) has to prove it can turn a *real* LangGraph run into
one of these. If an assertion test fails later after the recorder exists, you'll
know immediately whether the bug is in the assertion or in how the recorder built
the Trajectory, because this file never touches the recorder at all.
"""

import pytest

from graphtrace import (
    LLMCall,
    Step,
    ToolCall,
    Trajectory,
    TrajectoryAssertionError,
    any_str,
    assert_budget,
    assert_interrupted_at,
    assert_not_visited,
    assert_path,
    assert_retries,
    assert_state,
    assert_state_at,
    assert_tool_called,
    assert_tool_calls_match,
    assert_tool_not_called,
    assert_visited,
    assert_visits,
)


def _step(index: int, node: str, **kw) -> Step:
    kw.setdefault("state_before", {})
    kw.setdefault("state_after", {})
    kw.setdefault("state_diff", {})
    kw.setdefault("kind", "node")
    return Step(index=index, node=node, **kw)


@pytest.fixture
def refund_trajectory() -> Trajectory:
    router_llm = LLMCall(
        node="router", model="gpt-4o-mini", input_hash="abc123",
        prompt_tokens=412, completion_tokens=9, latency_s=0.61,
        output={"intent": "refund"}, cost_usd=0.0002,
    )
    lookup_tool = ToolCall(
        node="lookup_order", name="get_order", args={"order_id": "1234"},
        result={"status": "delivered", "amount": 49.0}, error=None, latency_s=0.3,
    )
    steps = [
        _step(0, "router", state_after={"intent": "refund"}, llm_calls=[router_llm], latency_s=0.6),
        _step(
            1, "lookup_order",
            state_before={"intent": "refund"},
            state_after={"intent": "refund", "order": {"amount": 49.0}},
            tool_calls=[lookup_tool], latency_s=0.3,
        ),
        _step(
            2, "human_approval",
            state_before={"intent": "refund", "order": {"amount": 49.0}},
            state_after={"intent": "refund", "order": {"amount": 49.0}, "approved": True},
            latency_s=0.1,
        ),
        _step(
            3, "refund",
            state_before={"approved": True},
            state_after={"approved": True, "refunded": True},
            latency_s=0.2,
        ),
    ]
    final_state = {"intent": "refund", "order": {"amount": 49.0}, "approved": True, "refunded": True}
    return Trajectory(steps=steps, final_state=final_state, interrupted=False, mode="replay")


# -- assert_path --------------------------------------------------------------------

def test_assert_path_exact_match_passes(refund_trajectory):
    assert_path(refund_trajectory, ["router", "lookup_order", "human_approval", "refund"])


def test_assert_path_exact_mismatch_fails_with_readable_diff(refund_trajectory):
    with pytest.raises(TrajectoryAssertionError) as exc_info:
        assert_path(refund_trajectory, ["router", "escalate", "refund"])
    message = str(exc_info.value)
    assert "[1] expected='escalate' actual='lookup_order'" in message


def test_assert_path_with_gap_passes(refund_trajectory):
    assert_path(refund_trajectory, ["router", ..., "refund"])


def test_assert_path_with_gap_fails_when_head_or_tail_dont_match(refund_trajectory):
    with pytest.raises(TrajectoryAssertionError):
        assert_path(refund_trajectory, ["escalate", ..., "refund"])


def test_assert_path_rejects_more_than_one_gap(refund_trajectory):
    with pytest.raises(ValueError):
        assert_path(refund_trajectory, ["router", ..., "human_approval", ..., "refund"])


# -- assert_visited / assert_not_visited / assert_visits -----------------------------

def test_assert_visited_passes_and_fails(refund_trajectory):
    assert_visited(refund_trajectory, "human_approval")
    with pytest.raises(TrajectoryAssertionError):
        assert_visited(refund_trajectory, "escalate")


def test_assert_not_visited_passes_and_fails(refund_trajectory):
    assert_not_visited(refund_trajectory, "escalate")
    with pytest.raises(TrajectoryAssertionError):
        assert_not_visited(refund_trajectory, "router")


def test_assert_visits_respects_max():
    traj = Trajectory(
        steps=[_step(0, "retriever"), _step(1, "retriever"), _step(2, "retriever")],
        final_state={}, interrupted=False, mode="replay",
    )
    assert_visits(traj, "retriever", max=3)
    with pytest.raises(TrajectoryAssertionError):
        assert_visits(traj, "retriever", max=2)


# -- assert_state / assert_state_at --------------------------------------------------

def test_assert_state_checks_final_state(refund_trajectory):
    assert_state(refund_trajectory, "intent", "refund")
    with pytest.raises(TrajectoryAssertionError):
        assert_state(refund_trajectory, "intent", "escalation")


def test_assert_state_at_checks_state_after_a_specific_node(refund_trajectory):
    assert_state_at(refund_trajectory, "router", "intent", "refund")
    with pytest.raises(TrajectoryAssertionError):
        assert_state_at(refund_trajectory, "human_approval", "approved", False)


def test_assert_state_at_fails_clearly_for_an_unvisited_node(refund_trajectory):
    with pytest.raises(TrajectoryAssertionError):
        assert_state_at(refund_trajectory, "escalate", "intent", "refund")


def test_assert_state_at_occurrence_picks_first_vs_last_visit():
    traj = Trajectory(
        steps=[
            _step(0, "retriever", state_after={"n": 1}),
            _step(1, "retriever", state_after={"n": 2}),
        ],
        final_state={}, interrupted=False, mode="replay",
    )
    assert_state_at(traj, "retriever", "n", 2)  # default occurrence=-1 -> last visit
    assert_state_at(traj, "retriever", "n", 1, occurrence=0)  # first visit


# -- assert_tool_called / assert_tool_not_called -------------------------------------

def test_assert_tool_called_with_exact_args(refund_trajectory):
    assert_tool_called(refund_trajectory, "get_order", args_match={"order_id": "1234"})
    with pytest.raises(TrajectoryAssertionError):
        assert_tool_called(refund_trajectory, "get_order", args_match={"order_id": "9999"})


def test_assert_tool_called_with_callable_matcher(refund_trajectory):
    assert_tool_called(refund_trajectory, "get_order", args_match={"order_id": lambda v: v.startswith("12")})


def test_assert_tool_called_with_any_str_matcher(refund_trajectory):
    assert_tool_called(refund_trajectory, "get_order", args_match={"order_id": any_str()})


def test_assert_tool_not_called(refund_trajectory):
    assert_tool_not_called(refund_trajectory, "delete_account")
    with pytest.raises(TrajectoryAssertionError):
        assert_tool_not_called(refund_trajectory, "get_order")


# -- assert_tool_calls_match ----------------------------------------------------------

def _tool_call(name: str) -> ToolCall:
    return ToolCall(node="x", name=name, args={}, result=None, error=None, latency_s=0.0)


def _traj_with_tools(*names: str) -> Trajectory:
    return Trajectory(
        steps=[_step(i, "x", tool_calls=[_tool_call(n)]) for i, n in enumerate(names)],
        final_state={}, interrupted=False, mode="replay",
    )


def test_assert_tool_calls_match_strict_requires_exact_order():
    traj = _traj_with_tools("lookup", "charge")
    assert_tool_calls_match(traj, ["lookup", "charge"], mode="strict")
    with pytest.raises(TrajectoryAssertionError):
        assert_tool_calls_match(traj, ["charge", "lookup"], mode="strict")


def test_assert_tool_calls_match_unordered_ignores_order_but_not_multiset():
    traj = _traj_with_tools("lookup", "charge")
    assert_tool_calls_match(traj, ["charge", "lookup"], mode="unordered")
    with pytest.raises(TrajectoryAssertionError):
        assert_tool_calls_match(traj, ["lookup"], mode="unordered")


def test_assert_tool_calls_match_subset_rejects_anything_extra():
    traj = _traj_with_tools("lookup")
    assert_tool_calls_match(traj, ["lookup", "charge"], mode="subset")

    traj_with_extra = _traj_with_tools("lookup", "delete_account")
    with pytest.raises(TrajectoryAssertionError):
        assert_tool_calls_match(traj_with_extra, ["lookup"], mode="subset")


def test_assert_tool_calls_match_superset_requires_every_reference_call():
    traj = _traj_with_tools("lookup", "charge", "notify")
    assert_tool_calls_match(traj, ["lookup", "charge"], mode="superset")
    with pytest.raises(TrajectoryAssertionError):
        assert_tool_calls_match(traj, ["lookup", "refund"], mode="superset")


def test_assert_tool_calls_match_accepts_a_trajectory_as_reference():
    traj = _traj_with_tools("lookup", "charge")
    reference = _traj_with_tools("lookup", "charge")
    assert_tool_calls_match(traj, reference, mode="strict")


def test_assert_tool_calls_match_rejects_unknown_mode():
    traj = _traj_with_tools("lookup")
    with pytest.raises(ValueError):
        assert_tool_calls_match(traj, ["lookup"], mode="fuzzy")


# -- assert_retries -------------------------------------------------------------------

def test_assert_retries_only_counts_reentry_after_a_tool_error():
    failed_call = ToolCall(node="lookup_order", name="get_order", args={}, result=None, error="timeout", latency_s=0.1)
    ok_call = ToolCall(node="lookup_order", name="get_order", args={}, result={"ok": True}, error=None, latency_s=0.1)
    traj = Trajectory(
        steps=[
            _step(0, "lookup_order", tool_calls=[failed_call]),
            _step(1, "lookup_order", tool_calls=[ok_call]),
        ],
        final_state={}, interrupted=False, mode="replay",
    )
    assert_retries(traj, "lookup_order", max=1)
    with pytest.raises(TrajectoryAssertionError):
        assert_retries(traj, "lookup_order", max=0)


def test_assert_retries_does_not_flag_an_ordinary_loop(refund_trajectory):
    # lookup_order only ran once and never errored -- not a retry, nothing to flag.
    assert_retries(refund_trajectory, "lookup_order", max=0)


# -- assert_interrupted_at -------------------------------------------------------------

def test_assert_interrupted_at_passes_and_fails():
    traj = Trajectory(
        steps=[_step(0, "router"), _step(1, "human_approval", kind="interrupt")],
        final_state={}, interrupted=True, mode="replay",
    )
    assert_interrupted_at(traj, "human_approval")
    with pytest.raises(TrajectoryAssertionError):
        assert_interrupted_at(traj, "refund")


def test_assert_interrupted_at_fails_if_run_completed(refund_trajectory):
    with pytest.raises(TrajectoryAssertionError):
        assert_interrupted_at(refund_trajectory, "human_approval")


# -- assert_budget ----------------------------------------------------------------------

def test_assert_budget_passes_within_limits(refund_trajectory):
    assert_budget(refund_trajectory, cost_usd=1.0, latency_s=10, tokens=10_000)


def test_assert_budget_fails_when_cost_exceeded(refund_trajectory):
    with pytest.raises(TrajectoryAssertionError):
        assert_budget(refund_trajectory, cost_usd=0.00001)


def test_assert_budget_fails_when_latency_exceeded(refund_trajectory):
    with pytest.raises(TrajectoryAssertionError):
        assert_budget(refund_trajectory, latency_s=0.01)


def test_assert_budget_fails_when_tokens_exceeded(refund_trajectory):
    with pytest.raises(TrajectoryAssertionError):
        assert_budget(refund_trajectory, tokens=1)
