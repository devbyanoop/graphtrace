"""Tests for the pytest11 plugin registration and the `run_graph` fixture.

The entry-point registration test matters because it's exactly the kind of thing
that's easy to get wrong silently: a typo'd entry point in pyproject.toml would
still let every other test in this repo pass (they all import graphtrace.* directly
and never rely on auto-discovery), while a real end user installing graphtrace and
expecting the fixture to "just work" would get a confusing fixture-not-found error.
"""

from __future__ import annotations

from importlib.metadata import entry_points

pytest_plugins = ["pytester"]


def test_pytest11_entry_point_is_registered():
    registered = {ep.name: ep.value for ep in entry_points(group="pytest11")}
    assert registered.get("graphtrace") == "graphtrace.plugin"


def test_run_graph_fixture_is_available_and_prefills_cassette(pytester):
    """A real sub-test-run, via pytest's own `pytester` fixture, proves the
    `run_graph` fixture is discovered the normal way (the editable install's
    entry point), not just importable as a plain Python function -- and that it
    pre-fills `cassette` from the test's own node id.
    """
    pytester.makepyfile(
        """
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict

        class State(TypedDict):
            x: int

        def test_uses_run_graph_fixture(run_graph):
            assert "test_uses_run_graph_fixture" in run_graph.keywords["cassette"]

            def a(state):
                return {"x": state["x"] + 1}

            g = StateGraph(State)
            g.add_node("a", a)
            g.add_edge(START, "a")
            g.add_edge("a", END)

            traj = run_graph(g.compile(), {"x": 0})
            assert traj.final_state == {"x": 1}
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
