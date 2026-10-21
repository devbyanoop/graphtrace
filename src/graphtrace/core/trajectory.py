"""The Trajectory data model: a recorded account of one run through a graph.

This module is the foundation everything else is built on, and the plan's layering
rule applies strictly here: nothing in `core/` may import LangGraph, an LLM SDK, or
even graphtrace's own `runtime/` package. A Trajectory is just dataclasses and plain
Python values -- lists, dicts, strings, floats. That's deliberate: `runtime/` (built
next) is the part that knows how to turn a live LangGraph run into one of these; once
that conversion happens, everything downstream (assertions, budgets, the HTML report)
only ever touches this framework-agnostic shape. If graphtrace grows a CrewAI or
AutoGen adapter later, it only has to produce a Trajectory too -- nothing in this file
or in assertions.py would need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class LLMCall:
    """One call to a chat model, captured by an interceptor while a node was running."""

    node: str
    model: str
    input_hash: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    output: Any
    # Computed by runtime/pricing.py at record time, not here. Core isn't allowed to
    # know about a model->USD pricing table (that's a runtime concern, and pricing
    # tables go stale and need overriding per the plan's `pricing_overrides` config) --
    # so by the time an LLMCall reaches core, its cost is already a plain float.
    # Defaults to 0.0 so a test can build an LLMCall by hand without supplying it.
    cost_usd: float = 0.0


@dataclass
class ToolCall:
    """One call to a tool, captured by an interceptor while a node was running."""

    node: str
    name: str
    args: dict[str, Any]
    result: Any
    error: str | None
    latency_s: float


@dataclass
class Step:
    """One node's execution: its state transition, plus whatever it called."""

    index: int
    node: str
    kind: Literal["node", "interrupt", "subgraph_enter", "subgraph_exit"]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    state_diff: dict[str, Any]
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    namespace: tuple[str, ...] = ()
    latency_s: float = 0.0


@dataclass
class Trajectory:
    """A full recorded run: every step, the final state, and how it ended."""

    steps: list[Step]
    final_state: dict[str, Any]
    interrupted: bool
    mode: Literal["record", "replay", "record_missing", "live"]

    @property
    def path(self) -> list[str]:
        """The sequence of node names visited, in order -- what assert_path checks."""
        return [step.node for step in self.steps]

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call across every step, flattened and in chronological order."""
        return [call for step in self.steps for call in step.tool_calls]

    @property
    def llm_calls(self) -> list[LLMCall]:
        """Every LLM call across every step, flattened and in chronological order."""
        return [call for step in self.steps for call in step.llm_calls]

    @property
    def final_message(self) -> str | None:
        """The last message's text content, if the graph's state has a `messages` key.

        LangGraph's common convention is a `messages` key reduced with `add_messages`,
        holding either LangChain message objects or plain `{"role": ..., "content": ...}`
        dicts depending on how the graph's state schema was written. We support both
        instead of assuming one -- a graph that doesn't use this convention at all
        just gets None back, which is correct: there's no "final message" to report.
        """
        messages = self.final_state.get("messages")
        if not messages:
            return None
        last = messages[-1]
        if hasattr(last, "content"):
            return last.content
        if isinstance(last, dict):
            return last.get("content")
        return None

    @property
    def cost_usd(self) -> float:
        """Total cost of every LLM call in the run, in US dollars."""
        return sum(call.cost_usd for call in self.llm_calls)

    @property
    def latency_s(self) -> float:
        """Total wall-clock time across all steps, in seconds."""
        return sum(step.latency_s for step in self.steps)

    @property
    def tokens_total(self) -> int:
        """Total prompt + completion tokens across every LLM call in the run."""
        return sum(call.prompt_tokens + call.completion_tokens for call in self.llm_calls)

    def visits(self, node: str) -> int:
        """How many times `node` appears in the path."""
        return self.path.count(node)
