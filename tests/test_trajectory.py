"""Tests for the Trajectory data model itself -- its derived properties, not the
assert_* functions (those get their own file). If these fail, something's wrong
with the arithmetic graphtrace does over a recorded run, independent of whether any
particular test's expectations are right.
"""

from graphtrace import LLMCall, Step, ToolCall, Trajectory


def _step(index: int, node: str, **kw) -> Step:
    kw.setdefault("state_before", {})
    kw.setdefault("state_after", {})
    kw.setdefault("state_diff", {})
    kw.setdefault("kind", "node")
    return Step(index=index, node=node, **kw)


def test_path_is_node_names_in_order():
    traj = Trajectory(
        steps=[_step(0, "router"), _step(1, "lookup_order"), _step(2, "refund")],
        final_state={},
        interrupted=False,
        mode="replay",
    )
    assert traj.path == ["router", "lookup_order", "refund"]


def test_visits_counts_repeats():
    traj = Trajectory(
        steps=[_step(0, "retriever"), _step(1, "retriever"), _step(2, "answer")],
        final_state={},
        interrupted=False,
        mode="replay",
    )
    assert traj.visits("retriever") == 2
    assert traj.visits("answer") == 1
    assert traj.visits("nonexistent") == 0


def test_tool_calls_and_llm_calls_flatten_across_steps():
    call_a = ToolCall(node="lookup", name="get_order", args={}, result=None, error=None, latency_s=0.1)
    call_b = ToolCall(node="refund", name="refund_api", args={}, result=None, error=None, latency_s=0.1)
    llm = LLMCall(
        node="router", model="gpt-4o-mini", input_hash="x",
        prompt_tokens=10, completion_tokens=5, latency_s=0.2, output="hi",
    )
    traj = Trajectory(
        steps=[
            _step(0, "router", llm_calls=[llm]),
            _step(1, "lookup", tool_calls=[call_a]),
            _step(2, "refund", tool_calls=[call_b]),
        ],
        final_state={},
        interrupted=False,
        mode="replay",
    )
    assert traj.tool_calls == [call_a, call_b]
    assert traj.llm_calls == [llm]


def test_cost_latency_and_tokens_sum_across_llm_calls_and_steps():
    llm_1 = LLMCall(
        node="router", model="gpt-4o-mini", input_hash="a",
        prompt_tokens=100, completion_tokens=20, latency_s=0.5, output="x", cost_usd=0.01,
    )
    llm_2 = LLMCall(
        node="refund", model="gpt-4o-mini", input_hash="b",
        prompt_tokens=50, completion_tokens=10, latency_s=0.3, output="y", cost_usd=0.005,
    )
    traj = Trajectory(
        steps=[
            _step(0, "router", llm_calls=[llm_1], latency_s=0.6),
            _step(1, "refund", llm_calls=[llm_2], latency_s=0.4),
        ],
        final_state={},
        interrupted=False,
        mode="replay",
    )
    assert traj.cost_usd == 0.015
    assert traj.latency_s == 1.0
    assert traj.tokens_total == 100 + 20 + 50 + 10


def test_final_message_reads_last_message_dict():
    traj = Trajectory(
        steps=[],
        final_state={"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}]},
        interrupted=False,
        mode="replay",
    )
    assert traj.final_message == "done"


def test_final_message_reads_last_message_object_with_content_attr():
    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    traj = Trajectory(
        steps=[],
        final_state={"messages": [FakeMessage("hi"), FakeMessage("done")]},
        interrupted=False,
        mode="replay",
    )
    assert traj.final_message == "done"


def test_final_message_is_none_without_a_messages_key():
    traj = Trajectory(steps=[], final_state={"other": 1}, interrupted=False, mode="replay")
    assert traj.final_message is None
