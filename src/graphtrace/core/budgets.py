"""Default cost/latency/token budgets, read from `graphtrace.yaml`'s `budgets:`
section once config.py loads it, and applied automatically by the `run_graph`
pytest fixture -- so a whole suite can share one budget without every test
calling `assert_budget(...)` by hand.

`assert_budget` itself (core/assertions.py, built in Week 1) doesn't change at
all: this is a thin config-driven caller of it, not a new assertion mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphtrace.core.assertions import assert_budget
from graphtrace.core.trajectory import Trajectory


@dataclass
class Budgets:
    cost_usd: float | None = None
    latency_s: float | None = None
    tokens: int | None = None

    def check(self, traj: Trajectory) -> None:
        """Apply this budget to `traj`, the same way a test calling
        assert_budget(traj, ...) by hand would -- raises TrajectoryAssertionError
        on the same terms. A no-op if every field is None (nothing configured)."""
        assert_budget(traj, cost_usd=self.cost_usd, latency_s=self.latency_s, tokens=self.tokens)
