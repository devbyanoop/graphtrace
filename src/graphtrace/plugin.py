"""graphtrace's pytest11 entry point.

Registered via pyproject.toml's `[project.entry-points.pytest11]`, so installing
graphtrace (`pip install -e .`) makes the fixtures below available in any test file,
with no `-p graphtrace.plugin` flag needed -- confirmed by hand, not assumed, that
a plain `pytest` run in this repo picks the `run_graph` fixture up automatically.

Scope today: only the `run_graph` fixture, a thin convenience over
`graphtrace.run_graph()`. The plan's `--gt-mode` CLI option and the `cassette`/
`judge` fixtures aren't added yet -- they'd do nothing today, since record/replay
and the judge contract are both still Week 2/3, and this project's own rule is not
to ship a flag or fixture that doesn't actually do anything yet.
"""

from __future__ import annotations

import functools

import pytest

from graphtrace.frameworks.langgraph.runner import run_graph as _run_graph


@pytest.fixture
def run_graph(request: pytest.FixtureRequest):
    """graphtrace.run_graph(), with `cassette` pre-filled from the current test's
    node id -- so a test doesn't have to invent a cassette path by hand.

    `cassette` still isn't used for anything yet (see this module's docstring) --
    pre-filling it now means a test written against this fixture today won't need
    to change at all once record/replay exists.
    """
    cassette_path = f"tests/cassettes/{request.node.name}.yaml"
    return functools.partial(_run_graph, cassette=cassette_path)
