"""graphtrace's pytest11 entry point.

Registered via pyproject.toml's `[project.entry-points.pytest11]`, so installing
graphtrace (`pip install -e .`) makes the fixtures below available in any test file,
with no `-p graphtrace.plugin` flag needed -- confirmed by hand, not assumed, that
a plain `pytest` run in this repo picks the `run_graph` fixture up automatically.

Week 2 scope: the `run_graph` fixture now reads `graphtrace.yaml` (if present) for
its cassette directory and default budgets, and the `--gt-mode` CLI flag. The
`cassette`/`judge` fixtures the architecture plan also names still aren't added --
the judge contract is Week 3, and this project's own rule is not to ship a fixture
that doesn't actually do anything yet.
"""

from __future__ import annotations

import pytest

from graphtrace.config import load_config
from graphtrace.core.budgets import Budgets
from graphtrace.frameworks.langgraph.runner import run_graph as _run_graph


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--gt-mode",
        action="store",
        default=None,
        help="graphtrace replay mode: record | replay | record_missing | live. "
        "Falls back to the GRAPHTRACE_MODE environment variable, then 'live', "
        "if not given.",
    )


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
        budgets.check(traj)
        return traj

    return _call
