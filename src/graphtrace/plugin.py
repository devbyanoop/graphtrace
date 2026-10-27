"""graphtrace's pytest11 entry point.

Registered via pyproject.toml's `[project.entry-points.pytest11]`, so installing
graphtrace (`pip install -e .`) makes the fixtures below available in any test file,
with no `-p graphtrace.plugin` flag needed -- confirmed by hand, not assumed, that
a plain `pytest` run in this repo picks the `run_graph` fixture up automatically.

Week 3: `--gt-report=<path>` collects a TestResult (docs/guide/report-and-ci.md's
own documented scope) for every test and renders the HTML report at session end --
no setup beyond the flag itself, the same way `--gt-mode` needed no fixture wiring
in a test body. The `pytest_runtest_makereport` wrapper hook below is the standard
pluggy >=1.1 `wrapper=True` style (confirmed working against the actual installed
pytest/pluggy by hand with a throwaway conftest before writing this for real --
getting this specific hook wrong would silently break every test run in this repo,
since this module loads for all of them via the entry point above).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from graphtrace.config import load_config
from graphtrace.core.budgets import Budgets
from graphtrace.frameworks.langgraph.runner import run_graph as _run_graph
from graphtrace.report.html import render_report
from graphtrace.report.results import TestResult, save_results


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--gt-mode",
        action="store",
        default=None,
        help="graphtrace replay mode: record | replay | record_missing | live. "
        "Falls back to the GRAPHTRACE_MODE environment variable, then 'live', "
        "if not given.",
    )
    parser.addoption(
        "--gt-report",
        action="store",
        default=None,
        help="Write a static HTML regression report (every test's trajectory, "
        "cost/latency totals) to this path at session end.",
    )


def pytest_configure(config: pytest.Config) -> None:
    # A fresh list per Config, not a module-level global -- a module-level global
    # would leak results between separate pytest sessions run in the same process
    # (e.g. this project's own pytester-based plugin tests), since this module is
    # only ever imported once per process.
    config._graphtrace_results = []  # type: ignore[attr-defined]
    # Registered here (not just documented) so using it never prints pytest's
    # own "Unknown marker" warning -- a test author's first use of this should
    # work cleanly, not look like a typo.
    config.addinivalue_line(
        "markers",
        "graphtrace(description=None, severity='normal', category=None): "
        "report metadata for the executive summary (report/results.py's "
        "build_executive_summary) -- a plain-language description of the "
        "scenario, a severity for ranking it among real failures, and a "
        "category for the coverage breakdown. All optional; an untagged "
        "test still reports correctly, just without the extra emphasis.",
    )


def _graphtrace_marker_kwargs(item: pytest.Item) -> dict[str, Any]:
    marker = item.get_closest_marker("graphtrace")
    return dict(marker.kwargs) if marker else {}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> pytest.TestReport:
    report = yield
    results: list[TestResult] | None = getattr(item.config, "_graphtrace_results", None)
    if results is not None:
        marker_kwargs = _graphtrace_marker_kwargs(item)
        if report.when == "call" and not report.skipped:
            traj = getattr(item, "_graphtrace_last_trajectory", None)
            outcome = "passed" if report.passed else "failed"
            error_message = str(report.longrepr) if not report.passed else None
            results.append(TestResult.from_trajectory(item.nodeid, outcome, traj, error_message=error_message, **marker_kwargs))
        elif report.when == "setup" and report.failed:
            # The test body never ran at all (a fixture blew up) -- pytest's own
            # JUnit-XML convention calls this "error", distinct from a "failed"
            # assertion inside the test, and there's no Trajectory to attach.
            results.append(TestResult(test_id=item.nodeid, outcome="error", error_message=str(report.longrepr), **marker_kwargs))
    return report


def pytest_sessionfinish(session: pytest.Session) -> None:
    report_path = session.config.getoption("--gt-report")
    if not report_path:
        return
    results: list[TestResult] = getattr(session.config, "_graphtrace_results", [])
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(results), encoding="utf-8")
    # The JSON alongside the HTML, not just the rendered page -- `graphtrace
    # report` (cli.py) needs a real results.json to re-render from standalone
    # (a second rubric calibrated later, a merge across CI shards, ...), and
    # docs/guide/report-and-ci.md already documents "a pytest hook writes
    # per-test JSON results" as this feature's own scope, not an afterthought.
    save_results(results, path.with_suffix(".json"))


@pytest.fixture
def run_graph(request: pytest.FixtureRequest):
    """graphtrace.run_graph(), with three things filled in automatically so a test
    doesn't have to repeat them by hand:

    - `cassette` defaults to `<config's replay.cassettes dir>/<test node id>.yaml`.
    - `mode` defaults to `--gt-mode` (if the pytest run was given that flag) --
      explicitly passing `mode=` to a call through this fixture still overrides it.
    - After the real call returns, `graphtrace.yaml`'s `budgets.default` (if any)
      is checked against the resulting Trajectory automatically -- a test that
      never calls `assert_budget(...)` at all still fails if the configured
      default is exceeded, the same way a real CI budget gate would work without
      every single test needing to repeat the same numbers.
    - The resulting Trajectory is stashed on the test item, for
      `pytest_runtest_makereport` above to pick up when `--gt-report` is active.
      A test that calls `run_graph` more than once keeps only the *last* call's
      Trajectory -- one test reported as one trajectory is the common case this
      is built for; a test that genuinely runs several distinct graphs isn't
      well served by a single-trajectory report entry regardless of which one
      "wins", so no attempt is made to merge them.
    """
    config = load_config()
    cli_mode = request.config.getoption("--gt-mode")
    cassettes_dir = config.replay.cassettes.rstrip("/")
    cassette_path = f"{cassettes_dir}/{request.node.name}.yaml"
    budgets = Budgets(**config.budgets.default)

    def _call(graph, inputs, **kwargs):
        kwargs.setdefault("cassette", cassette_path)
        kwargs.setdefault("mode", cli_mode)
        traj = _run_graph(graph, inputs, **kwargs)
        request.node._graphtrace_last_trajectory = traj  # stashed before the budget check, so a budget failure still reports the real trajectory, not nothing
        budgets.check(traj)
        return traj

    return _call
