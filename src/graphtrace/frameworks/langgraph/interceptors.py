"""Capture LLM and tool calls made while a LangGraph node is running, attribute
each one to the exact task (node execution) that made it, and route the actual
call through the active replay mode (replay/engine.py) -- without changing the
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

Record/replay (Week 2): each patched method wraps its real call in
`replay.engine.resolve_call`, converting its own result to/from a plain,
YAML-safe shape first -- `resolve_call` itself never sees an `AIMessage` or
`ToolMessage` object, only dicts/strings/numbers, which is what keeps it
framework-agnostic (see that module's own docstring).

A fourth real correction (2026-10-26, found testing against a real external
app -- `langchain-ai/langgraph-swarm-py`'s own `create_agent`): the
`config["metadata"]` lookup above assumed every node explicitly threads its
`config` argument into whatever call it makes, the way a hand-written node
(this project's own `examples/refund_agent`) does. LangChain 1.x's newer
`langchain.agents.create_agent()` doesn't -- confirmed by hand that its model
call reaches `BaseChatModel.invoke()` with `config=None` every time, relying
instead on `langchain_core.runnables.config`'s *ambient*, contextvar-based
config propagation (`var_child_runnable_config`), which genuinely does still
carry the real `langgraph_node`/`langgraph_checkpoint_ns` metadata -- just not
through the explicit argument this module originally read. Fixed by resolving
metadata through `ensure_config(config)` (the same public function LangChain's
own runnables use for exactly this "explicit-or-ambient" fallback) instead of
`(config or {})` directly, at every site below. `config` passed to the real
underlying call is left completely untouched either way -- only the metadata
*read* for attribution purposes changes, never what a real call actually sees.

A fifth real correction (2026-10-26, same external-app testing session): a
tool call's `args` -- and, separately, its *result* -- are not guaranteed to
already be YAML-safe the way an LLM call's `str(input)` request is.
`langgraph-swarm-py`'s own handoff tools confirmed two real, not contrived,
cases: a tool invoked with LangGraph's "inject full graph state as an arg"
pattern receives `args` containing raw `BaseMessage` objects nested inside a
plain dict, and a handoff tool's real *result* is a `Command` object (whose
own `update` field, in turn, nests more raw messages) -- not a plain
dict/str/`ToolMessage`, the only shapes this module's serialization
previously handled. `_yaml_safe()`/`_restore_value()` below extend the
existing `ToolMessage` round-trip pattern to `BaseMessage` generally (human/
ai/tool/system) and to `Command`, recursively, so record mode doesn't crash
trying to write one to a cassette, and replay mode hands back a *real*
message/`Command` object afterward -- not a plain dict a real node's own
`isinstance(result, Command)` check would then silently fail. Anything still
unrecognized falls back to `str()` for the write path; there's nothing
sensible to reconstruct from that on replay, so it's left as the plain
string rather than guessed at -- an honest, narrower limitation than crashing
or silently returning the wrong type.
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool
from langgraph.types import Command

from graphtrace.core.trajectory import LLMCall, ToolCall
from graphtrace.frameworks.pricing import cost_for_call
from graphtrace.replay.context import ReplayContext
from graphtrace.replay.engine import resolve_call

_MESSAGE_CLASSES_BY_TYPE: dict[str, type[BaseMessage]] = {"human": HumanMessage, "ai": AIMessage, "tool": ToolMessage, "system": SystemMessage}


def _yaml_safe(value: Any) -> Any:
    """Recursively convert a real tool call's args or result into something
    `ruamel.yaml` can actually write -- see this module's own docstring for
    the real cases (nested `BaseMessage`s, a `Command`) that made this
    necessary. Dict/list recurse; a message or `Command` gets a `_type`-
    tagged dict `_restore_value` below can reverse; anything else already
    YAML-safe passes through; anything still unrecognized falls back to
    `str()` -- a readable cassette entry, never a crash, and never a value
    `_restore_value` will try (and fail) to reconstruct as something it isn't.
    """
    if isinstance(value, BaseMessage):
        data: dict[str, Any] = {"_type": "Message", "type": value.type, "content": value.content}
        if value.name:
            data["name"] = value.name
        if isinstance(value, AIMessage) and value.tool_calls:
            data["tool_calls"] = _yaml_safe(list(value.tool_calls))
        if isinstance(value, ToolMessage):
            data["tool_call_id"] = value.tool_call_id
        return data
    if isinstance(value, Command):
        return {"_type": "Command", "graph": value.graph, "goto": value.goto, "update": _yaml_safe(value.update), "resume": _yaml_safe(value.resume)}
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _restore_value(value: Any) -> Any:
    """The inverse of `_yaml_safe` -- used on a replay hit, so a node that
    checks `isinstance(result, Command)` (LangGraph's own real handoff
    pattern) still sees a real one, not the plain dict a naive replay would
    hand back instead."""
    if isinstance(value, dict):
        tag = value.get("_type")
        if tag == "Message":
            cls = _MESSAGE_CLASSES_BY_TYPE.get(value.get("type"), AIMessage)
            kwargs: dict[str, Any] = {"content": value.get("content", "")}
            if "name" in value:
                kwargs["name"] = value["name"]
            if cls is AIMessage and "tool_calls" in value:
                kwargs["tool_calls"] = _restore_value(value["tool_calls"])
            if cls is ToolMessage:
                kwargs["tool_call_id"] = value.get("tool_call_id", "")
            return cls(**kwargs)
        if tag == "Command":
            return Command(graph=value.get("graph"), goto=value.get("goto"), update=_restore_value(value.get("update")), resume=_restore_value(value.get("resume")))
        return {k: _restore_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_value(v) for v in value]
    return value


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
    same content (their __repr__ includes field values), which is all this needs
    to be useful for -- matching "was this the same request" (replay/matcher.py
    does the same stringification independently), not cryptographic security.
    """
    return hashlib.sha256(str(messages).encode("utf-8")).hexdigest()


def _serialize_llm_response(message: Any) -> dict[str, Any]:
    """AIMessage -> a plain, YAML-safe dict, for `resolve_call`/the cassette."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "content": message.content,
        "tool_calls": list(getattr(message, "tool_calls", None) or []),
        "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)},
    }


def _deserialize_llm_response(data: dict[str, Any]) -> AIMessage:
    """The inverse of `_serialize_llm_response` -- used on a replay hit, so the
    rest of the graph sees a real AIMessage either way, live or replayed."""
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return AIMessage(
        content=data.get("content", ""),
        tool_calls=list(data.get("tool_calls") or []),
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    )


def _serialize_tool_result(result: Any) -> Any:
    """Most tool results are already YAML-safe (a plain dict/str/number) --
    passed through unchanged. `BaseTool.invoke()` called with a ToolCall-shaped
    input (what LangGraph's real ToolNode always passes, confirmed by hand)
    returns a `ToolMessage` instead, which needs its own explicit conversion
    (kept as its own, original, pre-existing case here rather than folded into
    `_yaml_safe`'s newer, more general `Message` handling, so an already-
    recorded cassette's exact `_type: ToolMessage` entries keep working
    unchanged). Anything else -- a `Command` (a real handoff tool's real
    result, confirmed against langgraph-swarm-py) or a message nested inside
    some other structure -- goes through `_yaml_safe`, which didn't exist
    when this function was first written and this result shape wasn't yet
    something this project had seen."""
    if isinstance(result, ToolMessage):
        return {"_type": "ToolMessage", "content": result.content, "tool_call_id": result.tool_call_id, "name": result.name}
    return _yaml_safe(result)


def _deserialize_tool_result(data: Any) -> Any:
    if isinstance(data, dict) and data.get("_type") == "ToolMessage":
        return ToolMessage(content=data["content"], tool_call_id=data.get("tool_call_id", ""), name=data.get("name"))
    return _restore_value(data)


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
def intercept(buffer: CallBuffer, replay_ctx: ReplayContext) -> Iterator[None]:
    """Patch LangChain's chat-model and tool call surfaces for the duration of one
    run_graph()/arun_graph() call, routing every real call into `buffer`, and every
    call's actual execution through `replay_ctx`'s active mode.

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
        metadata = ensure_config(config).get("metadata") or {}
        node = metadata.get("langgraph_node", "?")
        model_name = getattr(self, "model_name", None) or getattr(self, "model", None) or type(self).__name__

        def real_call():
            return _serialize_llm_response(orig_chat_invoke(self, input, config, **kwargs))

        plain_response, latency_s = resolve_call(replay_ctx, node, "llm", str(input), real_call, extra={"model": model_name})
        result = _deserialize_llm_response(plain_response)
        _record_llm_call(self, input, config, result, latency_s, buffer)
        return result

    async def patched_chat_ainvoke(self: BaseChatModel, input, config=None, **kwargs):
        metadata = ensure_config(config).get("metadata") or {}
        node = metadata.get("langgraph_node", "?")
        model_name = getattr(self, "model_name", None) or getattr(self, "model", None) or type(self).__name__

        async def real_call_async():
            return _serialize_llm_response(await orig_chat_ainvoke(self, input, config, **kwargs))

        # resolve_call is sync -- in "live" mode (the only mode an async real call
        # can actually take), this awaits inline before resolve_call ever measures
        # latency around it, so timing is still correct. Replay/record paths never
        # call real_call_async at all.
        if replay_ctx.mode == "live":
            start = time.perf_counter()
            plain_response = await real_call_async()
            latency_s = time.perf_counter() - start
        else:
            plain_response, latency_s = resolve_call(
                replay_ctx, node, "llm", str(input),
                lambda: _serialize_llm_response(orig_chat_invoke(self, input, config, **kwargs)),
                extra={"model": model_name},
            )
        result = _deserialize_llm_response(plain_response)
        _record_llm_call(self, input, config, result, latency_s, buffer)
        return result

    def patched_invoke(self: BaseTool, input, config=None, **kwargs):
        metadata = ensure_config(config).get("metadata") or {}
        node = metadata.get("langgraph_node", "?")
        args = input.get("args", input) if isinstance(input, dict) and "args" in input else input

        def real_call():
            return _serialize_tool_result(orig_invoke(self, input, config, **kwargs))

        start = time.perf_counter()
        try:
            # args is kept raw for _record_tool_call/the real ToolCall object
            # (so a test can assert_tool_called(..., args_match={...}) against
            # the real value) -- only the copy resolve_call sees, for the
            # cassette and its key hash, needs to be YAML-safe.
            plain_response, latency_s = resolve_call(replay_ctx, node, "tool", _yaml_safe(args), real_call, extra={"name": self.name})
        except Exception as exc:
            _record_tool_call(self, input, config, result=None, error=str(exc), latency_s=time.perf_counter() - start, buffer=buffer)
            raise
        result = _deserialize_tool_result(plain_response)
        _record_tool_call(self, input, config, result=result, error=None, latency_s=latency_s, buffer=buffer)
        return result

    async def patched_ainvoke(self: BaseTool, input, config=None, **kwargs):
        metadata = ensure_config(config).get("metadata") or {}
        node = metadata.get("langgraph_node", "?")
        args = input.get("args", input) if isinstance(input, dict) and "args" in input else input

        start = time.perf_counter()
        try:
            if replay_ctx.mode == "live":
                plain_response = _serialize_tool_result(await orig_ainvoke(self, input, config, **kwargs))
                latency_s = time.perf_counter() - start
            else:
                plain_response, latency_s = resolve_call(
                    replay_ctx, node, "tool", _yaml_safe(args),
                    lambda: _serialize_tool_result(orig_invoke(self, input, config, **kwargs)),
                    extra={"name": self.name},
                )
        except Exception as exc:
            _record_tool_call(self, input, config, result=None, error=str(exc), latency_s=time.perf_counter() - start, buffer=buffer)
            raise
        result = _deserialize_tool_result(plain_response)
        _record_tool_call(self, input, config, result=result, error=None, latency_s=latency_s, buffer=buffer)
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
    metadata = ensure_config(config).get("metadata") or {}
    task_id = _task_id_from_checkpoint_ns(metadata.get("langgraph_checkpoint_ns"))
    if task_id is None:
        # Not running inside a recorded graph (e.g. a direct model.invoke() call in
        # a plain unit test) -- nothing to attribute this to, so there's nothing to
        # record. Not an error: graphtrace only records what a graph run produces.
        return

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
    metadata = ensure_config(config).get("metadata") or {}
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
