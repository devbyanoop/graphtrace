"""Tests for report/results.py and report/html.py.

Two kinds of trajectory fixture are used deliberately: a hand-built
Trajectory (for the JSON round-trip / _jsonsafe edge cases, where exact
control over shape matters) and a *real* one from the Week 1 refund_agent demo
(for render_report itself) -- per BUILD_PLAN.md §7.5's own verify-by, "rendering
against the Week 1 demo agent's real results, inspected by hand."
"""

from __future__ import annotations

import itertools
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from refund_agent.graph import build_refund_graph

from graphtrace import run_graph
from graphtrace.core.trajectory import LLMCall, Step, ToolCall, Trajectory
from graphtrace.report.html import render_report
from graphtrace.report.results import TestResult, load_results, save_results, trajectory_to_dict


def _fake_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.repeat(AIMessage(content="ok")))


def _hand_built_trajectory() -> Trajectory:
    step = Step(
        index=0,
        node="router",
        kind="node",
        state_before={},
        state_after={"intent": "refund"},
        state_diff={"intent": "refund"},
        llm_calls=[LLMCall(node="router", model="gpt-4o-mini", input_hash="h", prompt_tokens=10, completion_tokens=5, latency_s=0.3, output=AIMessage(content="routing to refund"), cost_usd=0.001)],
        tool_calls=[ToolCall(node="router", name="classify", args={"text": "refund?"}, result={"label": "refund"}, error=None, latency_s=0.1)],
        latency_s=0.4,
    )
    return Trajectory(steps=[step], final_state={"intent": "refund"}, interrupted=False, mode="record")


# -- results.py: trajectory_to_dict / _jsonsafe -------------------------------------------

def test_trajectory_to_dict_converts_an_aimessage_output_to_a_plain_dict():
    """LLMCall.output is a real AIMessage object (confirmed in
    frameworks/langgraph/interceptors.py), not already JSON-safe -- this is the
    one conversion trajectory_to_dict actually has to do by hand."""
    data = trajectory_to_dict(_hand_built_trajectory())
    llm_call = data["steps"][0]["llm_calls"][0]
    assert llm_call["output"] == {"content": "routing to refund", "tool_calls": []}


def test_trajectory_to_dict_passes_through_plain_tool_call_data():
    data = trajectory_to_dict(_hand_built_trajectory())
    tool_call = data["steps"][0]["tool_calls"][0]
    assert tool_call["args"] == {"text": "refund?"}
    assert tool_call["result"] == {"label": "refund"}


def test_trajectory_to_dict_includes_run_level_totals():
    data = trajectory_to_dict(_hand_built_trajectory())
    assert data["cost_usd"] == 0.001
    assert data["tokens_total"] == 15
    assert data["path"] == ["router"]


# -- results.py: TestResult JSON round trip ------------------------------------------------

def test_save_then_load_results_round_trips(tmp_path: Path):
    results = [
        TestResult.from_trajectory("test_a", "passed", _hand_built_trajectory()),
        TestResult(test_id="test_b", outcome="error", error_message="boom"),
    ]
    path = tmp_path / "results.json"
    save_results(results, path)
    loaded = load_results(path)

    assert len(loaded) == 2
    assert loaded[0].test_id == "test_a"
    assert loaded[0].trajectory["path"] == ["router"]
    assert loaded[1].outcome == "error"
    assert loaded[1].error_message == "boom"
    assert loaded[1].trajectory is None


def test_from_trajectory_defaults_judge_verdicts_to_an_empty_list():
    result = TestResult.from_trajectory("t", "passed", _hand_built_trajectory())
    assert result.judge_verdicts == []


# -- html.py: render_report, against hand-built and real trajectories ---------------------

def test_render_report_includes_summary_counts_and_totals():
    results = [
        TestResult.from_trajectory("test_a", "passed", _hand_built_trajectory()),
        TestResult(test_id="test_b", outcome="failed", error_message="path mismatch"),
    ]
    html = render_report(results)
    assert "1 passed" in html
    assert "1 failed" in html
    assert "$0.0010 total cost" in html
    assert "test_a" in html and "test_b" in html
    assert "path mismatch" in html


def test_render_report_against_a_real_refund_agent_trajectory():
    """The actual verify-by: a real run_graph() Trajectory from the Week 1 demo
    agent, not a hand-built stand-in -- proves the whole chain (recorder ->
    Trajectory -> trajectory_to_dict -> template) fits together end to end."""
    graph = build_refund_graph(_fake_model())
    traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[True])
    result = TestResult.from_trajectory("test_refund_flow", "passed", traj)

    html = render_report([result])
    assert "router" in html and "lookup_order" in html and "refund" in html
    assert "get_order" in html  # the real tool call name
    assert "test_refund_flow" in html
    assert "1 passed" in html


def test_render_report_shows_judge_verdicts():
    result = TestResult.from_trajectory(
        "test_a", "passed", _hand_built_trajectory(),
        judge_verdicts=[{"rubric_id": "refund_reply", "passed": True, "criteria": {"confirms_amount": {"pass": True, "reason": "stated $49"}}}],
    )
    html = render_report([result])
    assert "refund_reply" in html
    assert "confirms_amount" in html


# -- html.py: XSS escaping (BUILD_PLAN.md §7.5's explicit requirement) --------------------

def test_render_report_escapes_a_script_tag_in_trajectory_state():
    """Trajectory state can contain arbitrary client data -- confirm a <script>
    tag anywhere in it renders as escaped text, never live markup."""
    step = Step(
        index=0, node="router", kind="node",
        state_before={}, state_after={}, state_diff={"note": "<script>alert(1)</script>"},
        tool_calls=[ToolCall(node="router", name="echo", args={"msg": "<script>alert(2)</script>"}, result="<script>alert(3)</script>", error=None, latency_s=0.0)],
    )
    traj = Trajectory(steps=[step], final_state={"note": "<script>alert(1)</script>"}, interrupted=False, mode="record")
    result = TestResult.from_trajectory("test_xss", "passed", traj)

    html = render_report([result])
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html


def test_render_report_escapes_a_script_tag_in_an_error_message():
    result = TestResult(test_id="test_xss2", outcome="failed", error_message="<script>alert(4)</script>")
    html = render_report([result])
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
