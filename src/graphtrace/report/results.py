"""TestResult: the one JSON-serializable record both halves of this feature
share -- whatever collects results (a pytest hook, wired in cli.py) builds
these, and report/html.py only ever reads them back, never a live Trajectory.

Every field is already plain dict/list/str/number/bool/None by construction
(see `_jsonsafe` below), so save/load is a direct `json.dumps`/`json.loads` +
`dataclasses.asdict`/`TestResult(**d)` -- no custom (de)serialization layer,
no intermediate schema version to keep in sync by hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from graphtrace.core.trajectory import Trajectory

Outcome = Literal["passed", "failed", "error"]
Severity = Literal["critical", "high", "normal", "low"]

# Lower index = more severe = sorted first in the executive summary's blocking-
# issues list. "normal" is the default for any test that never opted into the
# @pytest.mark.graphtrace(...) marker at all -- untagged tests still show up,
# just without the extra emphasis a tagged "critical" one gets.
SEVERITY_ORDER: tuple[Severity, ...] = ("critical", "high", "normal", "low")


def _jsonsafe(value: Any) -> Any:
    """Most of a Trajectory is already plain data by core/trajectory.py's own
    layering rule -- the one real exception (confirmed in
    frameworks/langgraph/interceptors.py) is `LLMCall.output` and
    `ToolCall.result`, which hold the real `AIMessage`/`ToolMessage` object,
    not a plain dict. Recurses into dicts/lists since either field could in
    principle nest one (e.g. a tool result containing a message), and falls
    back to `str()` only for whatever's left over that `json.dumps` itself
    would otherwise choke on.
    """
    if hasattr(value, "content"):
        data: dict[str, Any] = {"content": value.content}
        if hasattr(value, "tool_calls"):
            data["tool_calls"] = _jsonsafe(list(value.tool_calls or []))
        return data
    if isinstance(value, dict):
        return {key: _jsonsafe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonsafe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def trajectory_to_dict(traj: Trajectory) -> dict[str, Any]:
    """A Trajectory, flattened into exactly what the report needs to render:
    the step-by-step tree, per-call detail, and the run-level totals
    (`Trajectory`'s own `cost_usd`/`latency_s`/`tokens_total`/`final_message`
    properties) -- computed once here rather than re-derived from raw steps
    every time report/html.py's template touches them.
    """
    return {
        "path": traj.path,
        "interrupted": traj.interrupted,
        "mode": traj.mode,
        "final_state": _jsonsafe(traj.final_state),
        "final_message": traj.final_message,
        "cost_usd": traj.cost_usd,
        "latency_s": traj.latency_s,
        "tokens_total": traj.tokens_total,
        "steps": [
            {
                "index": step.index,
                "node": step.node,
                "kind": step.kind,
                "namespace": list(step.namespace),
                "latency_s": step.latency_s,
                "state_diff": _jsonsafe(step.state_diff),
                "llm_calls": [
                    {
                        "node": c.node,
                        "model": c.model,
                        "prompt_tokens": c.prompt_tokens,
                        "completion_tokens": c.completion_tokens,
                        "latency_s": c.latency_s,
                        "cost_usd": c.cost_usd,
                        "output": _jsonsafe(c.output),
                    }
                    for c in step.llm_calls
                ],
                "tool_calls": [
                    {
                        "node": c.node,
                        "name": c.name,
                        "args": _jsonsafe(c.args),
                        "result": _jsonsafe(c.result),
                        "error": c.error,
                        "latency_s": c.latency_s,
                    }
                    for c in step.tool_calls
                ],
            }
            for step in traj.steps
        ],
    }


@dataclass
class TestResult:
    """One test's outcome, its trajectory (already flattened -- see
    `trajectory_to_dict`), and any judge verdicts graded against it.

    `judge_verdicts` holds plain dicts shaped like
    `dataclasses.asdict(GradeResult(...))` (`judge/client.py`) -- that
    dataclass is already JSON-safe (str/dict/bool fields only), so there's no
    reason to wrap it in a second type here.

    `description`/`severity`/`category` are optional, human-authored metadata
    -- set via `@pytest.mark.graphtrace(description=..., severity=...,
    category=...)` on the test itself (see `plugin.py`), not inferred from
    the test's own assertion/error text. Graphtrace has no way to know on its
    own that "refunded without approval" is more severe than "wording
    changed" -- that's a judgment call only a human can make, and this is
    where they record it. All three default to None/"normal" so an untagged
    test still renders correctly, just without the extra emphasis a tagged
    one gets; the report degrades gracefully rather than requiring universal
    adoption before showing any value at all.
    """

    __test__ = False  # not a pytest test class -- the name just collides with its collection heuristic

    test_id: str
    outcome: Outcome
    trajectory: dict[str, Any] | None = None
    error_message: str | None = None
    judge_verdicts: list[dict[str, Any]] = field(default_factory=list)
    description: str | None = None
    severity: Severity = "normal"
    category: str | None = None

    @classmethod
    def from_trajectory(
        cls,
        test_id: str,
        outcome: Outcome,
        traj: Trajectory | None,
        *,
        error_message: str | None = None,
        judge_verdicts: list[dict[str, Any]] | None = None,
        description: str | None = None,
        severity: Severity = "normal",
        category: str | None = None,
    ) -> "TestResult":
        return cls(
            test_id=test_id,
            outcome=outcome,
            trajectory=trajectory_to_dict(traj) if traj is not None else None,
            error_message=error_message,
            judge_verdicts=judge_verdicts or [],
            description=description,
            severity=severity,
            category=category,
        )


def build_executive_summary(results: list[TestResult]) -> dict[str, Any]:
    """The release-readiness view a QA lead, manager, consultant, or auditor
    actually needs -- a verdict, blocking issues ranked by severity (most
    severe first), and a coverage breakdown by category -- without reading a
    single trajectory. This is computed data, not markup, so it's testable
    on its own and reusable if graphtrace ever grows a second report format.

    Degrades gracefully when no test used the `graphtrace` marker
    (plugin.py) at all: `ready`/the blocking list only ever need `outcome`,
    which every result always has, so they're always meaningful. `categories`
    is empty and `has_categories` is False when nothing was tagged -- the
    report says so honestly instead of fabricating a coverage breakdown from
    nothing.

    Any failed/errored test is "blocking" regardless of its severity tag --
    severity controls *ranking and visual emphasis* among real failures, not
    whether a failure counts. Graphtrace has no basis to decide a failure is
    "safe to ship anyway"; that stays a human call the report surfaces
    clearly, never one the report makes for them.
    """
    blocking = [r for r in results if r.outcome in ("failed", "error")]
    blocking_sorted = sorted(
        blocking,
        key=lambda r: SEVERITY_ORDER.index(r.severity) if r.severity in SEVERITY_ORDER else SEVERITY_ORDER.index("normal"),
    )

    categories: dict[str, int] = {}
    for result in results:
        if result.category:
            categories[result.category] = categories.get(result.category, 0) + 1

    return {
        "ready": len(blocking) == 0,
        "blocking": blocking_sorted,
        "categories": categories,
        "has_categories": bool(categories),
    }


def save_results(results: list[TestResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")


def load_results(path: str | Path) -> list[TestResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TestResult(**entry) for entry in data]
