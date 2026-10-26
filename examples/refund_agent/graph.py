"""A small, real, compiled LangGraph matching the plan's own running example:

    router -> lookup_order -> human_approval -> refund

No real API key needed -- `build_refund_graph` takes any LangChain chat model,
and every test in this repo passes a fake one (`langchain_core`'s own
`GenericFakeChatModel`). This is what makes "10 tests pass in live mode" checkable
without a provider account: "live" here means "real node execution, real object
calls" -- it does not mean "real paid API calls."
"""

from __future__ import annotations

import itertools
from typing import Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class RefundState(TypedDict, total=False):
    order_id: str
    intent: str
    order: dict
    approved: bool
    refunded: bool
    lookup_attempts: int


def _make_get_order_tool(fail_times: int):
    """A tool that fails its first `fail_times` calls, then succeeds -- the knob
    `lookup_order` below retries against, for testing assert_retries for real."""
    attempts = {"count": 0}

    @tool
    def get_order(order_id: str) -> dict:
        """Look up an order by its id."""
        attempts["count"] += 1
        if attempts["count"] <= fail_times:
            raise RuntimeError(f"order service unavailable (attempt {attempts['count']})")
        return {"status": "delivered", "amount": 49.0}

    return get_order


def build_refund_graph(chat_model: Any, *, fail_lookup_times: int = 0, max_lookup_retries: int = 2):
    """Build and compile the demo graph.

    `chat_model` is whatever LangChain chat model the caller wants (a fake one in
    every test here) -- `router` just needs something to call so there's a real
    LLMCall to assert on.

    `lookup_order` calls its tool directly with its own try/except, deliberately
    not through LangGraph's prebuilt ToolNode: confirmed by hand that ToolNode's
    default behaviour on an uncaught tool exception is to let it propagate and
    crash the whole graph.stream() call, not to produce a clean per-call error a
    test could assert on. A real agent that wants graceful retry-on-failure has to
    catch its own tool errors the same way this node does -- this is that pattern,
    not a workaround specific to testing it.
    """
    get_order = _make_get_order_tool(fail_lookup_times)

    def router(state: RefundState, config) -> dict:
        chat_model.invoke(
            [("human", f"What does the customer want for order {state['order_id']}?")],
            config,
        )
        return {"intent": "refund"}

    def lookup_order(state: RefundState, config) -> dict:
        attempts = state.get("lookup_attempts", 0)
        try:
            order = get_order.invoke({"order_id": state["order_id"]}, config)
        except Exception:
            return {"lookup_attempts": attempts + 1}
        return {"order": order, "lookup_attempts": attempts + 1}

    def route_after_lookup(state: RefundState) -> str:
        if state.get("order"):
            return "human_approval"
        if state.get("lookup_attempts", 0) > max_lookup_retries:
            return "give_up"
        return "lookup_order"

    def human_approval(state: RefundState) -> dict:
        approved = interrupt({"question": "approve refund?", "order_id": state["order_id"]})
        return {"approved": approved}

    def route_after_approval(state: RefundState) -> str:
        return "refund" if state.get("approved") else "give_up"

    def refund(state: RefundState) -> dict:
        return {"refunded": True}

    def give_up(state: RefundState) -> dict:
        return {"refunded": False}

    g = StateGraph(RefundState)
    g.add_node("router", router)
    g.add_node("lookup_order", lookup_order)
    g.add_node("human_approval", human_approval)
    g.add_node("refund", refund)
    g.add_node("give_up", give_up)

    g.add_edge(START, "router")
    g.add_edge("router", "lookup_order")
    g.add_conditional_edges(
        "lookup_order",
        route_after_lookup,
        {"human_approval": "human_approval", "lookup_order": "lookup_order", "give_up": "give_up"},
    )
    g.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"refund": "refund", "give_up": "give_up"},
    )
    g.add_edge("refund", END)
    g.add_edge("give_up", END)

    return g.compile()


def build_refund_graph_demo():
    """A zero-argument convenience wrapper around `build_refund_graph`, for
    `graphtrace from-trace`'s own end-to-end demonstration/test (cli.py's
    "--target" needs something real and *importable with no arguments* to
    point at). Not what a real caller should use -- a real one has a real
    chat model and calls `build_refund_graph(that_model)` directly, same as
    every test in test_demo_agent.py does.
    """
    return build_refund_graph(GenericFakeChatModel(messages=itertools.repeat(AIMessage(content="ok"))))


def build_refund_graph_demo_with_retry():
    """Same as `build_refund_graph_demo()`, but `lookup_order`'s first call
    fails once before succeeding -- matches `examples/sample_trace/`'s own
    recorded retry exactly, so that sample's generated test genuinely
    exercises a real retry on replay, not a cassette entry that happens to
    go unused.
    """
    return build_refund_graph(GenericFakeChatModel(messages=itertools.repeat(AIMessage(content="ok"))), fail_lookup_times=1)
