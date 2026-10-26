"""Tests for the pytest11 plugin registration and the `run_graph` fixture.

The entry-point registration test matters because it's exactly the kind of thing
that's easy to get wrong silently: a typo'd entry point in pyproject.toml would
still let every other test in this repo pass (they all import graphtrace.* directly
and never rely on auto-discovery), while a real end user installing graphtrace and
expecting the fixture to "just work" would get a confusing fixture-not-found error.

Everything below that exercises the fixture's own behaviour (cassette path,
`--gt-mode`, the auto-budget-check) uses pytest's own `pytester` fixture for a
real sub-process-free sub-test-run -- the only way to actually prove a CLI flag
or a fixture's post-test-call behaviour works, rather than just that the function
that builds it doesn't raise.
"""

from __future__ import annotations

import textwrap
from importlib.metadata import entry_points

pytest_plugins = ["pytester"]


def test_pytest11_entry_point_is_registered():
    registered = {ep.name: ep.value for ep in entry_points(group="pytest11")}
    assert registered.get("graphtrace") == "graphtrace.plugin"


_LINEAR_GRAPH_SNIPPET = textwrap.dedent(
    """
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class State(TypedDict):
        x: int

    @tool
    def _increment(n: int) -> int:
        \"\"\"Add one to n.\"\"\"
        return n + 1

    def _build_graph():
        def a(state, config):
            result = _increment.invoke({"n": state["x"]}, config)
            return {"x": result}
        g = StateGraph(State)
        g.add_node("a", a)
        g.add_edge(START, "a")
        g.add_edge("a", END)
        return g.compile()
    """
)


def _test_file(*bodies: str) -> str:
    return _LINEAR_GRAPH_SNIPPET + "\n" + "\n".join(textwrap.dedent(body) for body in bodies)


def test_run_graph_fixture_is_available_and_writes_a_cassette_at_the_default_path(pytester):
    """Discovered the normal way (the editable install's entry point, no `-p`
    flag) -- and defaults `cassette` to `tests/cassettes/<node id>.yaml` when no
    `graphtrace.yaml` says otherwise."""
    pytester.makepyfile(_test_file(
        """
        def test_uses_run_graph_fixture(run_graph):
            traj = run_graph(_build_graph(), {"x": 0}, mode="record")
            assert traj.final_state == {"x": 1}
        """
    ))
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    assert (pytester.path / "tests" / "cassettes" / "test_uses_run_graph_fixture.yaml").exists()


def test_run_graph_fixture_honors_a_custom_cassette_directory_from_config(pytester):
    pytester.makefile(".yaml", graphtrace="replay:\n  cassettes: my_cassettes/\n")
    pytester.makepyfile(_test_file(
        """
        def test_custom_dir(run_graph):
            run_graph(_build_graph(), {"x": 0}, mode="record")
        """
    ))
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    assert (pytester.path / "my_cassettes" / "test_custom_dir.yaml").exists()


def test_gt_mode_cli_flag_is_read_and_applied(pytester):
    """Record once via one pytest invocation, then replay via a second, driven
    purely by `--gt-mode` -- no `mode=` argument in the test file at all."""
    pytester.makepyfile(_test_file(
        """
        def test_cli_mode(run_graph):
            traj = run_graph(_build_graph(), {"x": 0})
            assert traj.final_state == {"x": 1}
        """
    ))
    recorded = pytester.runpytest("--gt-mode=record")
    recorded.assert_outcomes(passed=1)

    replayed = pytester.runpytest("--gt-mode=replay")
    replayed.assert_outcomes(passed=1)


def test_run_graph_fixture_auto_applies_configured_default_budget(pytester):
    """The exact scenario BUILD_PLAN.md promises for core/budgets.py: a test that
    never calls assert_budget(...) itself still fails when the *configured*
    default is exceeded, because the fixture applies it automatically."""
    pytester.makefile(".yaml", graphtrace="budgets:\n  default: {latency_s: 0.0000001}\n")
    pytester.makepyfile(_test_file(
        """
        def test_exceeds_configured_budget(run_graph):
            run_graph(_build_graph(), {"x": 0}, mode="live")  # no assert_budget call at all
        """
    ))
    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*TrajectoryAssertionError*"])


def test_run_graph_fixture_defaults_to_live_mode_with_no_flag_and_no_config(pytester):
    """Confirms Week 1's behaviour (unchanged unless something opts in) still
    holds through the fixture specifically, not just through run_graph() itself."""
    pytester.makepyfile(_test_file(
        """
        def test_default_mode(run_graph):
            traj = run_graph(_build_graph(), {"x": 0})
            assert traj.mode == "live"
        """
    ))
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


# -- Week 3: --gt-report ------------------------------------------------------------------

def test_gt_report_writes_an_html_report_with_a_passing_and_a_failing_test(pytester):
    pytester.makepyfile(_test_file(
        """
        def test_passes(run_graph):
            traj = run_graph(_build_graph(), {"x": 0}, mode="record")
            assert traj.final_state == {"x": 1}

        def test_fails(run_graph):
            run_graph(_build_graph(), {"x": 0}, mode="record")
            assert False
        """
    ))
    report_path = pytester.path / "report.html"
    result = pytester.runpytest(f"--gt-report={report_path}")
    result.assert_outcomes(passed=1, failed=1)

    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "test_passes" in html and "test_fails" in html
    assert "1 passed" in html and "1 failed" in html


def test_gt_report_includes_the_real_trajectory_from_run_graph(pytester):
    pytester.makepyfile(_test_file(
        """
        def test_increments(run_graph):
            run_graph(_build_graph(), {"x": 0}, mode="record")
        """
    ))
    report_path = pytester.path / "report.html"
    pytester.runpytest(f"--gt-report={report_path}")

    html = report_path.read_text(encoding="utf-8")
    assert "[0] a" in html  # the single node in _LINEAR_GRAPH_SNIPPET's graph


def test_gt_report_reports_a_setup_error_distinctly_without_a_trajectory(pytester):
    """A fixture that blows up before the test body runs at all -- no run_graph
    call happened, so there's no Trajectory, but it must still show up in the
    report as something that needs attention, not be silently dropped."""
    pytester.makepyfile(_test_file(
        """
        import pytest

        @pytest.fixture
        def broken():
            raise RuntimeError("fixture setup exploded")

        def test_with_broken_fixture(broken):
            pass
        """
    ))
    report_path = pytester.path / "report.html"
    result = pytester.runpytest(f"--gt-report={report_path}")
    result.assert_outcomes(errors=1)

    html = report_path.read_text(encoding="utf-8")
    assert "1 errored" in html
    assert "fixture setup exploded" in html


def test_without_gt_report_flag_no_report_file_is_written(pytester):
    pytester.makepyfile(_test_file(
        """
        def test_plain(run_graph):
            run_graph(_build_graph(), {"x": 0}, mode="record")
        """
    ))
    pytester.runpytest()
    assert not (pytester.path / "report.html").exists()
