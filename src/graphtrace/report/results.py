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
    """

    __test__ = False  # not a pytest test class -- the name just collides with its collection heuristic

    test_id: str
    outcome: Outcome
    trajectory: dict[str, Any] | None = None
    error_message: str | None = None
    judge_verdicts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_trajectory(
        cls,
        test_id: str,
        outcome: Outcome,
        traj: Trajectory | None,
        *,
        error_message: str | None = None,
        judge_verdicts: list[dict[str, Any]] | None = None,
    ) -> "TestResult":
        return cls(
            test_id=test_id,
            outcome=outcome,
            trajectory=trajectory_to_dict(traj) if traj is not None else None,
            error_message=error_message,
            judge_verdicts=judge_verdicts or [],
        )


def save_results(results: list[TestResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")


def load_results(path: str | Path) -> list[TestResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TestResult(**entry) for entry in data]
