"""graphtrace, pointed at a real external library's real compiled graph --
langchain-ai/langgraph-swarm-py (official langchain-ai org, MIT, ~1.6k stars)
-- not graphtrace's own hand-written demo agent.

Gated with importorskip, not a hard dependency: `langgraph-swarm` is a real,
pinned external app kept purely as a permanent compatibility test (see the
`dev` extra's own comment in pyproject.toml), not something graphtrace itself
needs to run.

FakeChatModel below is copied verbatim from langgraph-swarm-py's own real
test file (tests/test_swarm.py) -- not invented for this exercise. It
overrides BaseChatModel._generate directly (the exact pattern graphtrace's
own Week 1 build log flagged as *not* interceptable at that layer) --
confirming graphtrace's interceptors, which patch BaseChatModel.invoke/
ainvoke instead, correctly see through it regardless of which concrete fake
is used.

Running this real suite against this real external app found and fixed
THREE real gaps in graphtrace itself, none of which graphtrace's own
synthetic demo agent (examples/refund_agent) ever exercised -- see
frameworks/langgraph/interceptors.py and recorder.py's own docstrings for
the full account of each:
  1. LangChain 1.x's newer create_agent() doesn't thread an explicit `config`
     into its model call the way a hand-written node does -- confirmed by
     hand that BaseChatModel.invoke() receives config=None every time here.
     Attribution fixed via LangChain's own ensure_config() ambient-
     contextvar fallback.
  2. A subgraph's synthesized enter/exit step pair had their `index` values
     backwards (exit lower than its own enter) -- fixed.
  3. A tool call returning Command(graph=Command.PARENT, ...) (LangGraph's
     real handoff mechanism) never gets its own task_result/checkpoint at
     all, silently dropping the step -- fixed by absorbing it the instant
     the parent's own task_result resolves. A second, related gap: neither
     the tool's real args (which can carry injected full graph state,
     including raw BaseMessage objects) nor a real Command result were
     YAML-safe, crashing cassette writes -- fixed with a generic, recursive
     serializer that round-trips BaseMessage/Command correctly through replay.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

langchain_agents = pytest.importorskip("langchain.agents")
langgraph_swarm = pytest.importorskip("langgraph_swarm")

from langchain.chat_models import BaseChatModel  # noqa: E402
from langchain.messages import AIMessage  # noqa: E402
from langchain.tools import BaseTool  # noqa: E402
from langchain_core.callbacks.manager import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.messages.base import BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from graphtrace import assert_path, assert_state, assert_tool_called, assert_visited  # noqa: E402


class FakeChatModel(BaseChatModel):
    idx: int = 0
    responses: list[BaseMessage]

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        generation = ChatGeneration(message=self.responses[self.idx])
        self.idx += 1
        return ChatResult(generations=[generation])

    def bind_tools(self, tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool], *, tool_choice: str | None = None, **kwargs: Any) -> "FakeChatModel":
        return self


def _build_swarm_app(model: FakeChatModel):
    create_agent = langchain_agents.create_agent
    create_handoff_tool = langgraph_swarm.create_handoff_tool
    create_swarm = langgraph_swarm.create_swarm

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    alice = create_agent(model, tools=[add, create_handoff_tool(agent_name="Bob")], system_prompt="You are Alice, an addition expert.", name="Alice")
    bob = create_agent(model, tools=[create_handoff_tool(agent_name="Alice", description="Transfer to Alice, she can help with math")], system_prompt="You are Bob, you speak like a pirate.", name="Bob")

    workflow = create_swarm([alice, bob], default_active_agent="Alice")
    return workflow.compile(checkpointer=MemorySaver())


@pytest.mark.graphtrace(
    description="Alice hands the conversation off to Bob when the user asks for him by name.",
    category="handoff",
)
def test_swarm_handoff_to_bob_via_graphtrace(run_graph):
    """Turn 1 of the real test_swarm.py scenario: the user asks for Bob,
    Alice hands off via a real Command(graph=Command.PARENT, ...) tool call."""
    model = FakeChatModel(responses=[
        AIMessage(content="", name="Alice", tool_calls=[{"name": "transfer_to_bob", "args": {}, "id": "call_1"}]),
        AIMessage(content="Ahoy, matey! Bob the pirate be at yer service.", name="Bob"),
    ])
    app = _build_swarm_app(model)

    traj = run_graph(app, {"messages": [{"role": "user", "content": "i'd like to speak to Bob"}]})

    assert_path(traj, ["Alice", "model", "tools", "Alice", "Bob", "model", "Bob"])
    assert_tool_called(traj, "transfer_to_bob")
    assert_state(traj, "active_agent", "Bob")
    assert_visited(traj, "Bob")
    assert traj.final_message == "Ahoy, matey! Bob the pirate be at yer service."

    # The subgraph_enter/exit pair's index values must be correctly ordered
    # (enter earlier than its own exit) -- the real bug this exact scenario
    # found: recorder.py originally gave the *freshly created* enter step a
    # later index than the exit step, which kept the parent's original,
    # earlier one.
    alice_enter = next(s for s in traj.steps if s.node == "Alice" and s.kind == "subgraph_enter")
    alice_exit = next(s for s in traj.steps if s.node == "Alice" and s.kind == "subgraph_exit")
    assert alice_enter.index < alice_exit.index


@pytest.mark.graphtrace(
    description="Bob hands the conversation back to Alice, who then calls the real add tool to answer a math question.",
    category="tool_use",
)
def test_swarm_handoff_back_and_math_tool_call_via_graphtrace(run_graph):
    """Turn 2: Bob hands back to Alice, who then calls the real `add` tool --
    the same two-turn conversation test_swarm.py's own test checks, asserted
    on here with graphtrace's own Trajectory/assert_* API instead."""
    model = FakeChatModel(responses=[
        AIMessage(content="", name="Bob", tool_calls=[{"name": "transfer_to_alice", "args": {}, "id": "call_2"}]),
        AIMessage(content="", name="Alice", tool_calls=[{"name": "add", "args": {"a": 5, "b": 7}, "id": "call_3"}]),
        AIMessage(content="The sum of 5 and 7 is 12.", name="Alice"),
    ])
    app = _build_swarm_app(model)

    traj = run_graph(app, {"messages": [{"role": "user", "content": "what's 5 + 7?"}], "active_agent": "Bob"}, thread_id="turn-2")

    assert_tool_called(traj, "add", args_match={"a": 5, "b": 7})
    assert_state(traj, "active_agent", "Alice")
    assert traj.final_message == "The sum of 5 and 7 is 12."
    # The real tool call that triggers the handoff (transfer_to_alice) must
    # not be silently dropped -- confirmed present with its real Command
    # result, not just the node it happened in.
    transfer_call = next(c for c in traj.tool_calls if c.name == "transfer_to_alice")
    assert transfer_call.result.goto == "Alice"


@pytest.mark.graphtrace(
    description="A three-agent handoff chain: Alice hands off to Bob, who hands off to Carol, who gives the final answer.",
    category="handoff",
)
def test_three_agent_handoff_chain_via_graphtrace(run_graph):
    """A real extension of the swarm API to a longer chain than either of
    this module's other tests uses -- the same create_swarm/
    create_handoff_tool calls, just with a third real agent, confirming
    graphtrace's subgraph handling holds across more than one hop."""
    create_agent = langchain_agents.create_agent
    create_handoff_tool = langgraph_swarm.create_handoff_tool
    create_swarm = langgraph_swarm.create_swarm

    model = FakeChatModel(responses=[
        AIMessage(content="", name="Alice", tool_calls=[{"name": "transfer_to_bob", "args": {}, "id": "call_1"}]),
        AIMessage(content="", name="Bob", tool_calls=[{"name": "transfer_to_carol", "args": {}, "id": "call_2"}]),
        AIMessage(content="Carol here, the final answer is 42.", name="Carol"),
    ])
    alice = create_agent(model, tools=[create_handoff_tool(agent_name="Bob")], system_prompt="You are Alice.", name="Alice")
    bob = create_agent(model, tools=[create_handoff_tool(agent_name="Carol")], system_prompt="You are Bob.", name="Bob")
    carol = create_agent(model, tools=[], system_prompt="You are Carol.", name="Carol")
    app = create_swarm([alice, bob, carol], default_active_agent="Alice").compile(checkpointer=MemorySaver())

    traj = run_graph(app, {"messages": [{"role": "user", "content": "chain to carol"}]}, thread_id="chain-thread")

    assert_tool_called(traj, "transfer_to_bob")
    assert_tool_called(traj, "transfer_to_carol")
    assert_state(traj, "active_agent", "Carol")
    assert traj.final_message == "Carol here, the final answer is 42."


@pytest.mark.graphtrace(
    description="A real tool exception (division by zero) inside a swarm agent must propagate with its real exception type, never swallowed or disguised.",
    severity="high",
    category="error_handling",
)
def test_swarm_real_tool_exception_propagates_with_its_real_type(run_graph):
    """Directly exercises graphtrace's own documented guarantee (replay/
    engine.py: "a real exception from an actual call keeps its exact type via
    a bare raise") against a real external app's real tool error for the
    first time -- not graphtrace's own synthetic demo agent. Confirmed by
    hand first: LangChain's own default tool-error handling re-raises the
    real exception rather than wrapping it into an error ToolMessage, so
    `run_graph` in live mode must let a real ZeroDivisionError through
    completely unmodified, not a generic RuntimeError or something swallowed."""
    create_agent = langchain_agents.create_agent
    create_swarm = langgraph_swarm.create_swarm

    def divide(a: int, b: int) -> float:
        """Divide a by b."""
        return a / b

    model = FakeChatModel(responses=[
        AIMessage(content="", name="Alice", tool_calls=[{"name": "divide", "args": {"a": 10, "b": 0}, "id": "call_1"}]),
    ])
    alice = create_agent(model, tools=[divide], system_prompt="You are Alice.", name="Alice")
    app = create_swarm([alice], default_active_agent="Alice").compile(checkpointer=MemorySaver())

    with pytest.raises(ZeroDivisionError):
        run_graph(app, {"messages": [{"role": "user", "content": "divide 10 by 0"}]}, thread_id="div-error-thread")


@pytest.mark.graphtrace(
    description="The swarm remembers which agent is active across three separate conversation turns, not just two.",
    category="state_persistence",
)
def test_swarm_state_persists_across_three_turns_via_graphtrace(run_graph):
    """A real extension of this module's own two-turn test -- the same
    checkpointer-backed persistence, carried one turn further, to confirm
    `active_agent` genuinely survives more than a single handoff-and-back
    round trip."""
    model = FakeChatModel(responses=[
        AIMessage(content="", name="Alice", tool_calls=[{"name": "transfer_to_bob", "args": {}, "id": "call_1"}]),
        AIMessage(content="Ahoy! Bob here.", name="Bob"),
        AIMessage(content="", name="Bob", tool_calls=[{"name": "transfer_to_alice", "args": {}, "id": "call_2"}]),
        AIMessage(content="Sure, I'm back.", name="Alice"),
        AIMessage(content="", name="Alice", tool_calls=[{"name": "add", "args": {"a": 1, "b": 1}, "id": "call_3"}]),
        AIMessage(content="1 + 1 is 2.", name="Alice"),
    ])
    app = _build_swarm_app(model)

    run_graph(app, {"messages": [{"role": "user", "content": "speak to Bob"}]}, thread_id="three-turn")
    traj_turn_2 = run_graph(app, {"messages": [{"role": "user", "content": "back to Alice"}]}, thread_id="three-turn")
    assert_state(traj_turn_2, "active_agent", "Alice")

    traj_turn_3 = run_graph(app, {"messages": [{"role": "user", "content": "what's 1 + 1?"}]}, thread_id="three-turn")
    assert_tool_called(traj_turn_3, "add", args_match={"a": 1, "b": 1})
    assert traj_turn_3.final_message == "1 + 1 is 2."
