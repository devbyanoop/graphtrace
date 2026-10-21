# Assertions

All 12 functions below are implemented and tested — `src/graphtrace/core/assertions.py`, 37 tests in `tests/test_assertions.py` and `tests/test_trajectory.py`. Every assertion raises `TrajectoryAssertionError` (a subclass of `AssertionError`, so `pytest.raises(AssertionError)` catches it too) with a message built to be read, not just a repr of two values.

```python
from graphtrace import (
    assert_path, assert_visited, assert_not_visited, assert_visits,
    assert_state, assert_state_at, assert_tool_called, assert_tool_not_called,
    assert_retries, assert_interrupted_at, assert_budget, any_str,
)
```

## `assert_path`

```python
assert_path(traj, ["router", "lookup_order", "human_approval", "refund"])
assert_path(traj, ["router", ..., "refund"])  # ... = anything in between
```

Checks the exact sequence of node names visited. Put the literal `...` (Python's `Ellipsis`) at one point to mean "anything can happen here" — useful when a retry loop or a branch makes the middle of a path unpredictable but the start and end are what the test actually cares about.

A plain mismatch (no `...`) produces an index-by-index diff instead of two dumped lists:

```text
path mismatch:
   [0] expected='router' actual='router'
-> [1] expected='escalate' actual='lookup_order'
   [2] expected='refund' actual='<missing>'
```

**Scope limit:** only one `...` gap is supported. A second one raises `ValueError`, not a silently-wrong match — two gaps make "which gap swallowed the mismatch?" genuinely ambiguous to report well, and nothing in v0.1's goals needs it.

## `assert_visited` / `assert_not_visited`

```python
assert_visited(traj, "human_approval")
assert_not_visited(traj, "escalate")
```

Simple membership checks against `.path`.

## `assert_visits`

```python
assert_visits(traj, "retriever", max=2)
```

Caps how many times a node ran. `max` only (no `min`) — that's the direction the use case actually needs (catching an unbounded retry/poll loop); `min` is a one-line addition the day a test needs it.

## `assert_state` / `assert_state_at`

```python
assert_state(traj, "intent", "refund")                       # final state
assert_state_at(traj, "router", "intent", "refund")           # state right after a node ran
assert_state_at(traj, "retriever", "n", 1, occurrence=0)       # ... after its *first* visit, not its last
```

`assert_state_at`'s `occurrence` follows ordinary Python list-indexing: `-1` (the default) means "the most recent visit," `0` means "the first." This matters once a node runs more than once — "the state after the router ran" is ambiguous otherwise, and defaulting to the most recent visit matches how people usually phrase it ("after it finally settled").

## `assert_tool_called` / `assert_tool_not_called`

```python
assert_tool_called(traj, "refund_api", args_match={"order_id": "1234"})
assert_tool_called(traj, "refund_api", args_match={"order_id": lambda v: v.startswith("12")})
assert_tool_called(traj, "refund_api", args_match={"request_id": any_str()})
assert_tool_not_called(traj, "delete_account")
```

`args_match` only has to name the keys you care about — any key not mentioned is ignored, and a call matches if *every* mentioned key matches. If a tool was called several times, matching just one call is enough: this checks "this happened at least once," not "this happened exactly this way every time."

Three kinds of value work in `args_match`:

| Value | Matches when |
| --- | --- |
| A plain value (`"1234"`) | Exactly equal |
| A callable (`lambda v: v > 0`) | Truthy when called with the real value |
| A `Matcher` (`any_str()`) | Its own `.matches()` says yes |

`any_str()` is the one matcher built so far — it matches any string, regardless of content. More (`any_int()`, `any_value()`, …) get added the day a real test needs one, not speculatively.

## `assert_retries`

```python
assert_retries(traj, "lookup_order", max=1)
```

Counts how many times a node was **re-entered after one of its own tool calls errored** — not just how many times it ran. This is narrower than `assert_visits` on purpose: a node that's a normal loop body (pagination, polling) running many times isn't a retry, it's the design. What this counts is specifically "ran, a tool call inside it failed, and it ran again" — the thing you'd actually want a budget on.

## `assert_interrupted_at`

```python
assert_interrupted_at(traj, "human_approval")
```

Asserts the run paused on a human-in-the-loop interrupt at a specific node (checks both `traj.interrupted` and that the last step's `kind == "interrupt"`).

## `assert_budget`

```python
assert_budget(traj, cost_usd=0.05, latency_s=8, tokens=6000)
```

Three independent, optional budgets, each checked against the **total** across the whole run (`Trajectory.cost_usd`, `.latency_s`, `.tokens_total` — see [Core concepts](trajectory-model.md)), not any single step. Pass only the ones a given test cares about.

## A note on `cost_usd`

`assert_budget`'s `cost_usd` check (and `Trajectory.cost_usd` itself) sums a `cost_usd` field that's set per `LLMCall`, not computed by walking a pricing table inside `core`. See [Core concepts: where cost comes from](trajectory-model.md#where-cost-comes-from) for why that split exists.

## `assert_tool_calls_match`

```python
assert_tool_calls_match(traj, ["lookup_order", "refund_api"], mode="strict")
assert_tool_calls_match(traj, other_trajectory, mode="unordered")  # reference can be a Trajectory too
```

Adopted deliberately from `langchain-ai/agentevals`' own trajectory-match evaluator, not invented — it's the official LangGraph ecosystem's validated answer to "compare this run against a known-good reference," which nothing else on this page does (the assertions above check specific, known expectations on *one* trajectory). `reference` is a plain list of tool names, or another `Trajectory` (its tool-call names are pulled from it). Matching is by tool **name only**, not args — `assert_tool_called(..., args_match=...)` already covers the arg side for one specific call.

| Mode | Passes when |
| --- | --- |
| `strict` | Same tool calls as `reference`, same order |
| `unordered` | Same tool calls (same multiset), any order |
| `subset` | `traj` only calls tools that appear in `reference` |
| `superset` | `traj` calls everything in `reference`, extra calls allowed |

An unrecognized `mode` raises `ValueError` immediately, the same "fail loudly on invalid input, don't guess" rule `assert_path`'s multi-gap check follows.
