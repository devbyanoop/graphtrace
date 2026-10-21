"""The FrameworkRunner Protocol: the one contract every framework package satisfies.

v0.1 has exactly one implementation (frameworks/langgraph/), built next. This file
exists anyway, built first, because retrofitting a shared contract onto an
already-public `run_graph()` later would mean either a breaking rename or a
confusing two-tier API -- a "real" generic entry point plus a legacy LangGraph-only
one people already depend on. Defining the contract before the first implementation
costs nothing now and means a later CrewAI/AutoGen package (v0.4, see
Plan/graphtrace.md section 20) is purely additive: a new module satisfying this same
Protocol, not a refactor of a shipped one.

This module imports nothing framework-specific -- not even LangGraph. If it ever
does, that's a sign the Protocol leaked a LangGraph assumption into what's supposed
to be the shared contract every framework package implements identically.
"""

from __future__ import annotations

from typing import Any, Protocol

from graphtrace.core.trajectory import Trajectory


class FrameworkRunner(Protocol):
    """Structural contract: run a target through a framework, get a Trajectory back.

    This is a Protocol (structural typing), not an abstract base class on purpose --
    the same reasoning kbguard's own `adapters/base.py` uses for its vector-store
    adapters. A framework implementation doesn't need to inherit from anything here,
    it only needs methods with these shapes. That keeps the coupling one-directional:
    graphtrace depends on the shape of a runner, a runner never has to import or
    subclass anything from graphtrace.frameworks.base to satisfy it.

    `target` and `inputs` are intentionally untyped (`Any`) here -- what a "target"
    even is differs completely per framework (a compiled LangGraph vs. a CrewAI
    `Crew` vs. an AutoGen conversable agent), and this Protocol's job is only to
    promise "whatever it is, you get a Trajectory back," not to describe it.
    """

    def run(
        self,
        target: Any,
        inputs: Any,
        *,
        cassette: str,
        resume: Any = None,
    ) -> Trajectory: ...

    async def arun(
        self,
        target: Any,
        inputs: Any,
        *,
        cassette: str,
        resume: Any = None,
    ) -> Trajectory: ...
