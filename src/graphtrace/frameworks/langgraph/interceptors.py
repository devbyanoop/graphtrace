"""Capture LLM and tool calls made while a LangGraph node is running, and attribute
each one to the exact task (node execution) that made it -- without changing the
graph's own code at all.

How attribution actually works (verified against real langgraph 1.2 / langchain-core
1.6, not assumed from memory or from the original architecture plan):

The plan's own original design called for a `contextvars.ContextVar` set by the
recorder to "the current step" while a node runs, read back by these interceptors.
That design doesn't actually work: LangGraph's Pregel engine runs node functions
(including ones running in parallel, "fan-out") on a ThreadPoolExecutor worker
thread, and `ThreadPoolExecutor.submit()` does **not** propagate the calling
thread's contextvars into the worker thread by default -- a ContextVar set in the
thread that's consuming `graph.stream(...)` would read back as empty inside the
node function that actually makes the LLM/tool call, confirmed by hand with a
`threading.current_thread()` probe inside a fan-out node before trusting this design
at all.

What actually works, and is used instead: LangGraph tags every LLM and tool call's
`config["metadata"]` with `langgraph_checkpoint_ns`, a string shaped exactly
`"{node_name}:{task_id}"` -- confirmed by hand that the `{task_id}` half is
byte-for-byte the same `id` the debug event stream (recorder.py) uses for that task,
including across repeated visits to the same node in a retry loop (each visit gets
its own id). This is a strictly better correlation key than a ContextVar would have
been even if threading had cooperated: it travels with the call itself, through
whatever thread actually executes it, with no side channel at all.

Where this module patches, and why -- also verified by hand, not assumed:
`BaseChatModel._generate`/`_agenerate` (the architecture plan's original patch
target) turned out to be the wrong place entirely: every concrete chat model
(OpenAI's, Anthropic's, LangChain's own fakes -- confirmed with
`GenericFakeChatModel`) defines its *own* `_generate`, overriding the base class's.
Patching `BaseChatModel._generate` therefore patches a method real calls never
actually go through -- Python's method resolution finds the subclass's own
`_generate` first, every time. The fix mirrors what already had to be true for
tools: patch `BaseChatModel.invoke`/`ainvoke` instead, the same way
`BaseTool.invoke`/`ainvoke` are patched below -- neither is something a concrete
subclass overrides (confirmed by hand for `GenericFakeChatModel`), and both carry
the same `config["metadata"]` correlation key straight through.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from graphtrace.core.trajectory import LLMCall, ToolCall
from graphtrace.frameworks.pricing import cost_for_call


def _task_id_from_checkpoint_ns(checkpoint_ns: str | None) -> str | None:
    """Pull the task id back out of a "{node_name}:{task_id}" checkpoint namespace.

    Splitting on the *last* ':' is safe even though it looks fragile: LangGraph
    itself uses ':' as this separator, which structurally means a node name can't
    contain one either (if it could, LangGraph's own parsing of this same string
    would already be ambiguous, not just ours).
    """
    if not checkpoint_ns:
        return None
    return checkpoint_ns.rsplit(":", 1)[-1]


def _hash_messages(messages: Any) -> str:
    """A stable-enough id for an LLM call's input, for Trajectory.LLMCall.input_hash.

    Plain str() of a list of LangChain message objects is deterministic given the
    same content (their __repr__ includes field values), which is all this needs to
    be useful for -- matching "was this the same request" later (replay/matcher.py,
    Week 2), not cryptographic security.
    """
    return hashlib.sha256(str(messages).encode("utf-8")).hexdigest()


class CallBuffer:
    """Where intercepted calls land, keyed by task id, until the recorder claims them.

    Guarded by a lock because real concurrency is the actual reason this class
    exists at all (see this module's docstring) -- two different tasks' calls can
    genuinely race to insert into this shared structure from different threads.
    Different tasks always write to different keys (each task id is unique, even
    across repeated visits to the same node), so the lock is only ever protecting
    the dict's own internal bookkeeping during concurrent inserts, not fighting over
    a single task's data.
    """

    def __init__(self) -> None:
        self._llm_calls: dict[str, list[LLMCall]] = defaultdict(list)
        self._tool_calls: dict[str, list[ToolCall]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_llm_call(self, task_id: str, call: LLMCall) -> None:
        with self._lock:
            self._llm_calls[task_id].append(call)

    def add_tool_call(self, task_id: str, call: ToolCall) -> None:
        with self._lock:
            self._tool_calls[task_id].append(call)

    def pop(self, task_id: str) -> tuple[list[LLMCall], list[ToolCall]]:
        """Claim (and remove) everything recorded for one task -- the recorder calls
        this exactly once per task, when that task's task_result event arrives."""
        with self._lock:
            return self._llm_calls.pop(task_id, []), self._tool_calls.pop(task_id, [])


@contextmanager
def intercept(buffer: CallBuffer) -> Iterator[None]:
    """Patch LangChain's chat-model and tool call surfaces for the duration of one
    run_graph()/arun_graph() call, routing every real call into `buffer`.

    Patching is undone in `finally` even if the graph run itself raises -- a test
    that never calls run_graph() must never see LangChain's model/tool classes
    behave any differently, and a crash mid-run must not leave them patched for
    whatever test runs next.
    """
    orig_chat_invoke = BaseChatModel.invoke
    orig_chat_ainvoke = BaseChatModel.ainvoke
    orig_invoke = BaseTool.invoke
    orig_ainvoke = BaseTool.ainvoke

    def patched_chat_invoke(self: BaseChatModel, input, config=None, **kwargs):
        start = time.perf_counter()
        result = orig_chat_invoke(self, input, config, **kwargs)
        _record_llm_call(self, input, config, result, time.perf_counter() - start, buffer)
        return result

    async def patched_chat_ainvoke(self: BaseChatModel, input, config=None, **kwargs):
        start = time.perf_counter()
        result = await orig_chat_ainvoke(self, input, config, **kwargs)
        _record_llm_call(self, input, config, result, time.perf_counter() - start, buffer)
        return result

    def patched_invoke(self: BaseTool, input, config=None, **kwargs):
        start = time.perf_counter()
        try:
            result = orig_invoke(self, input, config, **kwargs)
        except Exception as exc:
            _record_tool_call(self, input, config, result=None, error=str(exc), latency_s=time.perf_counter() - start, buffer=buffer)
            raise
        _record_tool_call(self, input, config, result=result, error=None, latency_s=time.perf_counter() - start, buffer=buffer)
        return result

    async def patched_ainvoke(self: BaseTool, input, config=None, **kwargs):
        start = time.perf_counter()
        try:
            result = await orig_ainvoke(self, input, config, **kwargs)
        except Exception as exc:
            _record_tool_call(self, input, config, result=None, error=str(exc), latency_s=time.perf_counter() - start, buffer=buffer)
            raise
        _record_tool_call(self, input, config, result=result, error=None, latency_s=time.perf_counter() - start, buffer=buffer)
        return result

    BaseChatModel.invoke = patched_chat_invoke
    BaseChatModel.ainvoke = patched_chat_ainvoke
    BaseTool.invoke = patched_invoke
    BaseTool.ainvoke = patched_ainvoke
    try:
        yield
    finally:
        BaseChatModel.invoke = orig_chat_invoke
        BaseChatModel.ainvoke = orig_chat_ainvoke
        BaseTool.invoke = orig_invoke
        BaseTool.ainvoke = orig_ainvoke


def _record_llm_call(model: BaseChatModel, input, config, result, latency_s: float, buffer: CallBuffer) -> None:
    metadata = (config or {}).get("metadata") or {}
    task_id = _task_id_from_checkpoint_ns(metadata.get("langgraph_checkpoint_ns"))
    if task_id is None:
        # Not running inside a recorded graph (e.g. a direct model.invoke() call in
        # a plain unit test) -- nothing to attribute this to, so there's nothing to
        # record. Not an error: graphtrace only records what a graph run produces.
        return

    # BaseChatModel.invoke() returns the AIMessage directly (unlike the lower-level
    # _generate(), which returns a ChatResult wrapping it) -- confirmed by hand.
    usage = getattr(result, "usage_metadata", None) or {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__

    buffer.add_llm_call(
        task_id,
        LLMCall(
            node=metadata.get("langgraph_node", "?"),
            model=model_name,
            input_hash=_hash_messages(input),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            output=result,
            cost_usd=cost_for_call(model_name, prompt_tokens, completion_tokens),
        ),
    )


def _record_tool_call(tool: BaseTool, input, config, *, result: Any, error: str | None, latency_s: float, buffer: CallBuffer) -> None:
    metadata = (config or {}).get("metadata") or {}
    task_id = _task_id_from_checkpoint_ns(metadata.get("langgraph_checkpoint_ns"))
    if task_id is None:
        return

    # LangGraph's ToolNode passes a full ToolCall-shaped dict ({"name", "args",
    # "id", "type"}) as `input`; a plain `tool.invoke({...})` outside a ToolNode
    # passes the raw args dict directly. Handle both rather than assuming one.
    args = input.get("args", input) if isinstance(input, dict) and "args" in input else input

    buffer.add_tool_call(
        task_id,
        ToolCall(
            node=metadata.get("langgraph_node", "?"),
            name=tool.name,
            args=args,
            result=result,
            error=error,
            latency_s=latency_s,
        ),
    )
