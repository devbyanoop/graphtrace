"""The assert_* functions a test actually calls.

Every function here does the same three things: look something up on a Trajectory,
compare it to what the caller expected, and raise TrajectoryAssertionError with a
readable message if it doesn't match. Keeping that pattern identical across all of
them is deliberate -- once you've read assert_visited, the rest are not new ideas,
just the same shape applied to a different field.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from graphtrace.core.errors import TrajectoryAssertionError, _first_mismatch
from graphtrace.core.matchers import Matcher
from graphtrace.core.trajectory import Trajectory

_TOOL_CALL_MATCH_MODES = ("strict", "unordered", "subset", "superset")


def _matches(expected: Any, actual: Any) -> bool:
    """Resolve one args_match value against one real value.

    Order matters here: a Matcher is checked first because matcher objects could
    technically also be `callable` (if someone added __call__ to one later) -- we
    want the explicit Matcher protocol to win over the looser "any callable" rule.
    Plain values fall through to ordinary equality.
    """
    if isinstance(expected, Matcher):
        return expected.matches(actual)
    if callable(expected):
        return bool(expected(actual))
    return expected == actual


def assert_path(traj: Trajectory, expected: list) -> None:
    """Assert the exact sequence of node names the graph visited.

    `expected` is normally a plain list of node names, checked for exact equality.
    Put the literal `...` (Python's built-in Ellipsis) at one point in the list to
    mean "anything can happen here" -- useful when a retry loop or a conditional
    branch makes the middle of a path unpredictable but the start and end are what
    the test actually cares about:

        assert_path(traj, ["router", ..., "refund"])

    Only one `...` is supported in v0.1. Supporting several, non-overlapping gaps is
    a small amount of extra matching logic, but nothing in the spec's examples or
    goals needs it yet, and a second gap makes failure messages much harder to make
    readable ("which gap swallowed the mismatch?"). Raising ValueError here is a
    direct, honest failure instead of a matcher that silently picks one interpretation
    out of several equally plausible ones.
    """
    actual = traj.path
    gap_positions = [i for i, item in enumerate(expected) if item is ...]

    if not gap_positions:
        if actual != expected:
            raise TrajectoryAssertionError(
                "path mismatch:\n" + _first_mismatch(expected, actual),
                expected=expected,
                actual=actual,
            )
        return

    if len(gap_positions) > 1:
        raise ValueError("assert_path supports at most one '...' gap in v0.1")

    gap = gap_positions[0]
    head, tail = expected[:gap], expected[gap + 1 :]
    actual_head = actual[: len(head)]
    actual_tail = actual[len(actual) - len(tail) :] if tail else []

    head_ok = actual_head == head
    tail_ok = (not tail) or actual_tail == tail
    enough_room = len(head) + len(tail) <= len(actual)

    if not (head_ok and tail_ok and enough_room):
        raise TrajectoryAssertionError(
            f"expected path to start with {head!r} and end with {tail!r} "
            f"(with anything in between), got {actual!r}",
            expected=expected,
            actual=actual,
        )


def assert_visited(traj: Trajectory, node: str) -> None:
    """Assert `node` appears in the path at least once."""
    if node not in traj.path:
        raise TrajectoryAssertionError(
            f"expected {node!r} to be visited, path was {traj.path!r}",
            expected=node,
            actual=traj.path,
        )


def assert_not_visited(traj: Trajectory, node: str) -> None:
    """Assert `node` never appears in the path."""
    if node in traj.path:
        raise TrajectoryAssertionError(
            f"expected {node!r} not to be visited, but path was {traj.path!r}",
            expected=None,
            actual=traj.path,
        )


def assert_visits(traj: Trajectory, node: str, *, max: int) -> None:
    """Assert `node` was visited at most `max` times.

    `max` only, no `min`, because that's the one direction the spec actually names
    (catching an unbounded retry/poll loop). Add `min` the day a test needs to assert
    a node ran *at least* N times -- it's a one-line addition, not a redesign.
    """
    count = traj.visits(node)
    if count > max:
        raise TrajectoryAssertionError(
            f"expected {node!r} to be visited at most {max} time(s), was visited {count} time(s)",
            expected=max,
            actual=count,
        )


def assert_state(traj: Trajectory, key: str, value: Any) -> None:
    """Assert a key in the trajectory's *final* state equals `value`."""
    actual = traj.final_state.get(key)
    if actual != value:
        raise TrajectoryAssertionError(
            f"expected final_state[{key!r}] == {value!r}, got {actual!r}",
            expected=value,
            actual=actual,
        )


def assert_state_at(traj: Trajectory, node: str, key: str, value: Any, *, occurrence: int = -1) -> None:
    """Assert a key in the state right after `node` ran equals `value`.

    `occurrence` follows ordinary Python list-indexing: -1 (the default) means "the
    most recent visit", 0 means "the first visit". This matters once a node is
    visited more than once (a loop body, a retry target) -- "the state after router
    ran" is ambiguous unless you say which visit you mean, and defaulting to the last
    one matches how people usually phrase this ("after it finally settled").
    """
    matches = [step for step in traj.steps if step.node == node]
    if not matches:
        raise TrajectoryAssertionError(
            f"node {node!r} was never visited", expected=node, actual=traj.path
        )
    try:
        step = matches[occurrence]
    except IndexError:
        raise TrajectoryAssertionError(
            f"node {node!r} was only visited {len(matches)} time(s); "
            f"occurrence={occurrence} is out of range",
            expected=occurrence,
            actual=len(matches),
        ) from None

    actual = step.state_after.get(key)
    if actual != value:
        raise TrajectoryAssertionError(
            f"expected state[{key!r}] == {value!r} after {node!r} "
            f"(occurrence {occurrence}), got {actual!r}",
            expected=value,
            actual=actual,
        )


def assert_tool_called(traj: Trajectory, name: str, *, args_match: dict[str, Any] | None = None) -> None:
    """Assert a tool named `name` was called, optionally matching specific args.

    `args_match` only has to describe the args you care about -- any key not
    mentioned is ignored, and a call "matches" if *all* mentioned keys match it.
    If several calls to this tool happened, matching just one of them is enough:
    this asserts "this happened at least once", not "this happened exactly this way
    every time".
    """
    calls = [c for c in traj.tool_calls if c.name == name]
    if not calls:
        raise TrajectoryAssertionError(
            f"expected tool {name!r} to be called, but it wasn't; "
            f"tools called: {[c.name for c in traj.tool_calls]!r}",
            expected=name,
            actual=[c.name for c in traj.tool_calls],
        )
    if args_match is None:
        return
    for call in calls:
        if all(key in call.args and _matches(matcher, call.args[key]) for key, matcher in args_match.items()):
            return
    raise TrajectoryAssertionError(
        f"tool {name!r} was called, but no call matched args {args_match!r}; "
        f"actual calls: {[c.args for c in calls]!r}",
        expected=args_match,
        actual=[c.args for c in calls],
    )


def assert_tool_not_called(traj: Trajectory, name: str) -> None:
    """Assert a tool named `name` was never called."""
    calls = [c.args for c in traj.tool_calls if c.name == name]
    if calls:
        raise TrajectoryAssertionError(
            f"expected tool {name!r} not to be called, but it was called {len(calls)} time(s)",
            expected=None,
            actual=calls,
        )


def assert_retries(traj: Trajectory, node: str, *, max: int) -> None:
    """Assert `node` was re-entered at most `max` times *after one of its own tool
    calls failed*.

    This is narrower than assert_visits on purpose. A node that's a normal loop body
    (pagination, polling) is expected to run many times -- that's not a retry, it's
    the design. What this counts is specifically "ran, a tool call inside it errored,
    and it ran again" -- an actual retry-after-failure, which is the thing you'd want
    a budget on.
    """
    steps = [s for s in traj.steps if s.node == node]
    retries = sum(
        1
        for prev, _curr in zip(steps, steps[1:])
        if any(call.error is not None for call in prev.tool_calls)
    )
    if retries > max:
        raise TrajectoryAssertionError(
            f"expected {node!r} to retry at most {max} time(s) after a tool error, "
            f"it retried {retries} time(s)",
            expected=max,
            actual=retries,
        )


def assert_interrupted_at(traj: Trajectory, node: str) -> None:
    """Assert the run paused on a human-in-the-loop interrupt at `node`."""
    last = traj.steps[-1] if traj.steps else None
    actual_node = last.node if last else None
    if not traj.interrupted or last is None or last.node != node or last.kind != "interrupt":
        raise TrajectoryAssertionError(
            f"expected the run to interrupt at {node!r}, but it "
            + ("ran to completion" if not traj.interrupted else f"stopped at {actual_node!r}"),
            expected=node,
            actual=actual_node,
        )


def assert_budget(
    traj: Trajectory,
    *,
    cost_usd: float | None = None,
    latency_s: float | None = None,
    tokens: int | None = None,
) -> None:
    """Assert the run stayed within cost, latency and/or token budgets.

    Each budget is independent and optional -- pass only the ones this test cares
    about. All three check the *total* across the whole run (see Trajectory.cost_usd,
    .latency_s and .tokens_total), not any single step.
    """
    if cost_usd is not None and traj.cost_usd > cost_usd:
        raise TrajectoryAssertionError(
            f"cost ${traj.cost_usd:.4f} exceeded budget ${cost_usd:.4f}",
            expected=cost_usd,
            actual=traj.cost_usd,
        )
    if latency_s is not None and traj.latency_s > latency_s:
        raise TrajectoryAssertionError(
            f"latency {traj.latency_s:.2f}s exceeded budget {latency_s:.2f}s",
            expected=latency_s,
            actual=traj.latency_s,
        )
    if tokens is not None and traj.tokens_total > tokens:
        raise TrajectoryAssertionError(
            f"token usage {traj.tokens_total} exceeded budget {tokens}",
            expected=tokens,
            actual=traj.tokens_total,
        )


def assert_tool_calls_match(
    traj: Trajectory,
    reference: Trajectory | Sequence[str],
    *,
    mode: str = "strict",
) -> None:
    """Assert the tool calls in `traj` match a reference, under one of four modes.

    Adopted deliberately from `langchain-ai/agentevals`' own trajectory-match
    evaluator, not invented -- it's the official LangGraph ecosystem's validated
    answer to "compare this run against a known-good reference run," which is a
    different, equally common need from the specific, known expectations
    assert_path/assert_tool_called are for.

    `reference` is either another Trajectory (its tool call *names*, in order, are
    pulled from it) or a plain list of tool names. Matching is by tool name only, not
    args -- agentevals treats sequence-matching and arg-matching as separate
    concerns, and `assert_tool_called(..., args_match=...)` already covers the arg
    side for a specific call.

    Modes (exactly agentevals' own four):
      - "strict":    same tool calls, same order.
      - "unordered": same tool calls (same multiset), any order.
      - "subset":    traj calls only tools that appear in reference -- nothing
                     beyond what reference allows happened.
      - "superset":  traj calls everything in reference -- extra calls are fine, but
                     nothing required is missing.
    """
    if mode not in _TOOL_CALL_MATCH_MODES:
        raise ValueError(f"mode must be one of {_TOOL_CALL_MATCH_MODES!r}, got {mode!r}")

    actual_names = [call.name for call in traj.tool_calls]
    reference_names = (
        [call.name for call in reference.tool_calls]
        if isinstance(reference, Trajectory)
        else list(reference)
    )

    if mode == "strict":
        ok = actual_names == reference_names
    elif mode == "unordered":
        ok = sorted(actual_names) == sorted(reference_names)
    elif mode == "subset":
        ok = set(actual_names) <= set(reference_names)
    else:  # superset
        ok = set(actual_names) >= set(reference_names)

    if not ok:
        raise TrajectoryAssertionError(
            f"tool calls did not {mode}-match reference: expected {reference_names!r}, "
            f"got {actual_names!r}",
            expected=reference_names,
            actual=actual_names,
        )
