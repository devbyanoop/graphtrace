"""graphtrace, pointed at a second real external library's real compiled
graph -- langchain-ai/deepagents (official langchain-ai org, ~30.6k stars),
the "deep agent" harness whose own README documents "any LangGraph
CompiledStateGraph can be passed in as a sub-agent" -- a *different* way of
using a subgraph than langgraph-swarm-py's (test_langgraph_swarm_compat.py):
there, "Alice"/"Bob" are direct nodes of the swarm's own compiled graph; here,
the sub-agent is invoked *from inside* a plain tool function (`task`), not
added as a graph node at all.

Gated with importorskip, not a hard dependency -- see pyproject.toml's `dev`
extra comment.

GenericFakeChatModel below is copied verbatim from deepagents' own real test
file (tests/unit_tests/chat_model.py) -- not invented for this exercise.
Mirrors the real test pattern in tests/unit_tests/test_subagents.py's
test_subagent_returns_final_message_as_tool_result.

Testing against this real app found a fourth real bug, and confirmed one
real, structural *limitation* worth documenting rather than silently
shipping around -- see frameworks/langgraph/interceptors.py's own docstring
for the first, and the second test below for the second:

4. `ToolRuntime` (LangChain's dependency-injection object, given to a tool
   that declares a `runtime: ToolRuntime` parameter -- deepagents' own
   `task` tool does) carries the *entire* live Pregel execution state as its
   `config` field: per-run object ids, open file/thread handles,
   `functools.partial`s. Confirmed by hand that this made the recorded
   cassette entry for the exact same logical call hash *differently* across
   two separate processes (object reprs embed memory addresses), and bloated
   the cassette into something genuinely unreadable. Fixed by representing
   only `state` and `tool_call_id` -- the only parts of a `ToolRuntime` a
   human reading a cassette, or a hash matching two calls, could plausibly
   care about.

A structural limitation, not a bug: a tool whose *own implementation*
triggers a nested LangGraph execution (deepagents' `task` tool really does
call `compiled_subagent.invoke(...)` inside its own Python body) loses that
nested execution's own step-by-step visibility in replay mode -- confirmed by
hand that record mode shows the sub-agent's own nested `model` step
(`subgraph_enter`/`exit` around it, the same mechanism that already handles
a subgraph *node*), while replay mode shows only the single, opaque, cached
`task` tool call, since replaying it means the real tool body -- including
its own `compiled_subagent.invoke(...)` call -- genuinely never executes.
The parent's own real-world behavior (the final message it produces) is
identical either way; only the fine-grained trace of *how the sub-agent
itself got there* is unrecoverable once that specific call is served from a
cassette. This is structurally different from langgraph-swarm-py's "Alice"/
"Bob": those are direct nodes LangGraph's own Pregel engine always executes
regardless of mode (graphtrace only intercepts the LLM/tool calls *inside*
a node, never node execution itself); a sub-agent invoked from inside a
cached tool call has no such guarantee, because the call that would trigger
it is exactly the thing being skipped.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

import pytest

deepagents_graph = pytest.importorskip("deepagents.graph")
deepagents_subagents = pytest.importorskip("deepagents.middleware.subagents")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models import LanguageModelInput  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from pydantic import Field  # noqa: E402

from graphtrace import assert_budget, assert_tool_called  # noqa: E402
from graphtrace import run_graph as _run_graph  # noqa: E402

create_deep_agent = deepagents_graph.create_deep_agent
CompiledSubAgent = deepagents_subagents.CompiledSubAgent


class GenericFakeChatModel(BaseChatModel):
    messages: Iterator[AIMessage | str] = Field(exclude=True)
    call_history: list[Any] = []
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool] = ()
    stream_delimiter: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def bind_tools(self, tools: Sequence[dict[str, Any] | type | Callable | BaseTool], *, tool_choice: str | None = None, **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = tools
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        self.call_history.append({"messages": messages, "kwargs": {"stop": stop, "run_manager": run_manager, **kwargs}, "tools": self.tools})
        message = next(self.messages)
        message_ = AIMessage(content=message) if isinstance(message, str) else message
        return ChatResult(generations=[ChatGeneration(message=message_)])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        chat_result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = chat_result.generations[0].message
        content = message.content
        tool_calls = message.tool_calls if hasattr(message, "tool_calls") else []
        if content:
            content_chunks = [content] if self.stream_delimiter is None else [c for c in cast(list, re.split(self.stream_delimiter, content)) if c]
            for idx, token in enumerate(content_chunks):
                is_last = idx == len(content_chunks) - 1
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=token, id=message.id, tool_calls=tool_calls if is_last else []))
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        elif tool_calls:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content="", id=message.id, tool_calls=tool_calls))
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"


def _build_deep_agent_with_subagent():
    parent_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "task", "args": {"description": "Calculate the sum of 2 and 3", "subagent_type": "general-purpose"}, "id": "call_calculate_sum", "type": "tool_call"}]),
        AIMessage(content="The calculation has been completed."),
    ]))
    subagent_model = GenericFakeChatModel(messages=iter([AIMessage(content="The sum of 2 and 3 is 5.")]))
    compiled_subagent = create_agent(model=subagent_model)

    return create_deep_agent(
        model=parent_model,
        checkpointer=InMemorySaver(),
        subagents=[CompiledSubAgent(name="general-purpose", description="A general-purpose agent for various tasks.", runnable=compiled_subagent)],
    )


@pytest.mark.graphtrace(
    description="The deep agent delegates a calculation to its general-purpose sub-agent and reports the result back.",
    category="delegation",
)
def test_deep_agent_subagent_delegation_via_graphtrace(run_graph):
    """The real sub-agent-via-tool-call pattern deepagents' own README
    documents. Scoped deliberately to assertions that hold in *both* record
    and live mode, not to the nested sub-agent step detail that record mode
    alone can see -- see the module docstring and the test below."""
    app = _build_deep_agent_with_subagent()

    traj = run_graph(app, {"messages": [HumanMessage(content="What is 2 + 3?")]}, thread_id="calc-thread")

    assert_tool_called(traj, "task", args_match={"subagent_type": "general-purpose"})
    assert traj.final_message == "The calculation has been completed."

    # The real follow-on bug found while generating a report from this exact
    # test: ToolCall.args is read directly by report/html.py (and by anyone
    # else inspecting a Trajectory after the fact), not just by resolve_call
    # -- and only the copy passed to resolve_call was being sanitized. The
    # raw ToolRuntime (object ids, open file/thread handles) was landing in
    # the in-memory Trajectory itself, not just a cassette.
    task_call = next(c for c in traj.tool_calls if c.name == "task")
    assert "object at 0x" not in str(task_call.args)
    assert task_call.args["runtime"]["_type"] == "ToolRuntime"


@pytest.mark.graphtrace(
    description="The deep agent delegates two independent calculations to two different sub-agents in parallel and synthesizes both results.",
    category="parallel_execution",
)
def test_deep_agent_parallel_subagents_via_graphtrace(run_graph):
    """The real parallel-delegation pattern from deepagents' own
    tests/unit_tests/test_subagents.py::test_multiple_subagents_invoked_in_parallel
    -- a single AIMessage with two `task` tool_calls, each routed to a
    *different* specialized sub-agent, confirmed to complete independently
    with their own distinct results, not conflated or dropped."""
    parent_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "task", "args": {"description": "Calculate the sum of 5 and 7", "subagent_type": "math-adder"}, "id": "call_addition", "type": "tool_call"},
            {"name": "task", "args": {"description": "Calculate the product of 4 and 6", "subagent_type": "math-multiplier"}, "id": "call_multiplication", "type": "tool_call"},
        ]),
        AIMessage(content="Both calculations completed successfully."),
    ]))
    addition_model = GenericFakeChatModel(messages=iter([AIMessage(content="The sum of 5 and 7 is 12.")]))
    multiplication_model = GenericFakeChatModel(messages=iter([AIMessage(content="The product of 4 and 6 is 24.")]))
    app = create_deep_agent(
        model=parent_model,
        checkpointer=InMemorySaver(),
        subagents=[
            CompiledSubAgent(name="math-adder", description="Specialized agent for addition.", runnable=create_agent(model=addition_model)),
            CompiledSubAgent(name="math-multiplier", description="Specialized agent for multiplication.", runnable=create_agent(model=multiplication_model)),
        ],
    )

    traj = run_graph(app, {"messages": [HumanMessage(content="What is 5+7 and what is 4*6?")]}, thread_id="parallel-thread")

    assert_tool_called(traj, "task", args_match={"subagent_type": "math-adder"})
    assert_tool_called(traj, "task", args_match={"subagent_type": "math-multiplier"})
    results_by_subagent_type = {c.args["subagent_type"]: c.result.update["messages"][0].content for c in traj.tool_calls if c.name == "task"}
    assert results_by_subagent_type == {
        "math-adder": "The sum of 5 and 7 is 12.",
        "math-multiplier": "The product of 4 and 6 is 24.",
    }
    assert traj.final_message == "Both calculations completed successfully."


@pytest.mark.graphtrace(
    description="Delegating to a sub-agent type that was never configured must fail gracefully -- the agent should be told, and the (nonexistent) work must never silently happen.",
    severity="high",
    category="safety",
)
def test_deep_agent_unknown_subagent_type_fails_gracefully_via_graphtrace(run_graph):
    """A real safety property, not covered by the happy-path delegation test
    above: a `task` call naming a `subagent_type` that was never registered
    must come back as a normal tool result the parent can recover from (not
    a crash, not a silent no-op), and the named sub-agent's own model must
    never actually run -- confirmed directly against the fake model's own
    call history, not inferred from the trajectory alone."""
    parent_model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "task", "args": {"description": "do something", "subagent_type": "nonexistent-agent"}, "id": "call_1", "type": "tool_call"}]),
        AIMessage(content="Sorry, that agent type does not exist."),
    ]))
    subagent_model = GenericFakeChatModel(messages=iter([AIMessage(content="should not run")]))
    app = create_deep_agent(
        model=parent_model,
        checkpointer=InMemorySaver(),
        subagents=[CompiledSubAgent(name="general-purpose", description="A general-purpose agent.", runnable=create_agent(model=subagent_model))],
    )

    traj = run_graph(app, {"messages": [HumanMessage(content="do something")]}, thread_id="unknown-agent-thread")

    assert_tool_called(traj, "task", args_match={"subagent_type": "nonexistent-agent"})
    assert traj.final_message == "Sorry, that agent type does not exist."
    # The real correctness check: the fake subagent model's own call history
    # must stay empty -- proof the deep agent recognized the invalid type and
    # genuinely never dispatched to it, not just that the parent's final
    # message happens to sound like it handled the error.
    assert subagent_model.call_history == []
    # No nested step under any namespace either -- nothing ran "inside" a
    # sub-agent, because no sub-agent was ever actually invoked.
    assert not any(s.namespace for s in traj.steps)


@pytest.mark.graphtrace(
    description="A single sub-agent delegation should complete well within a generous latency budget.",
    category="performance",
)
def test_deep_agent_delegation_stays_within_latency_budget(run_graph):
    """A real `assert_budget` check, not a toy -- fast, free, and local (a
    `GenericFakeChatModel`, no real network at all) specifically so this
    threshold is checking "did the real interception/recording machinery add
    unreasonable overhead", not "is the network slow today"."""
    app = _build_deep_agent_with_subagent()
    traj = run_graph(app, {"messages": [HumanMessage(content="What is 2 + 3?")]}, thread_id="budget-thread")
    assert_budget(traj, latency_s=5)


def test_deep_agent_record_replay_round_trip_and_the_nested_visibility_limitation(tmp_path):
    """Record once, replay in a *separate* call (simulating a separate
    process -- the real concern the ToolRuntime fix addresses: hashing had
    been non-deterministic across genuinely separate runs). Also documents,
    with a real assertion rather than just prose, the one structural
    limitation this app surfaced: the sub-agent's own nested `model` step is
    visible in the recorded trajectory but not the replayed one, while the
    parent's real-world outcome (the final message) is identical either way.
    """
    cassette = str(tmp_path / "deep_agent.yaml")
    inputs = {"messages": [HumanMessage(content="What is 2 + 3?")]}

    recorded = _run_graph(_build_deep_agent_with_subagent(), inputs, cassette=cassette, mode="record", thread_id="calc-thread")
    replayed = _run_graph(_build_deep_agent_with_subagent(), inputs, cassette=cassette, mode="replay", thread_id="calc-thread")

    assert recorded.final_message == replayed.final_message == "The calculation has been completed."
    assert_tool_called(replayed, "task", args_match={"subagent_type": "general-purpose"})

    recorded_nested = [s for s in recorded.steps if s.namespace]
    replayed_nested = [s for s in replayed.steps if s.namespace]
    assert len(recorded_nested) > 0, "record mode should see the sub-agent's own nested model step"
    assert len(replayed_nested) == 0, "replay mode cannot -- the cached tool call's real body, including its own compiled_subagent.invoke(...), never executes"

    # The cassette itself must stay deterministic and readable -- the real
    # bug found here: a ToolRuntime's `config` field embeds per-run object
    # ids/memory addresses that changed this request's hash on every run.
    cassette_text = (tmp_path / "deep_agent.yaml").read_text(encoding="utf-8")
    assert "object at 0x" not in cassette_text
    assert "_type: ToolRuntime" in cassette_text
