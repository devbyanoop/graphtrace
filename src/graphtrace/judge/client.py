"""JudgeClient.ask_json() via litellm, and grade() -- the higher-level call that
builds a rubric prompt, asks the judge, and scores the verdict.

`litellm` is imported lazily, inside the one method that actually makes a real
call -- the same pattern kbguard's own judge/client.py and ingest/embeddings.py
use. Constructing a JudgeClient, or even hitting a cassette-replayed judge
verdict, never requires litellm to be installed at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from graphtrace.judge.errors import JudgeError
from graphtrace.judge.rubric import Rubric
from graphtrace.replay.cassette import to_plain_python
from graphtrace.replay.context import build_replay_context, save_if_recording
from graphtrace.replay.engine import resolve_call
from graphtrace.replay.modes import resolve_mode


@dataclass
class JudgeClient:
    """model is pinned explicitly (never inferred or defaulted) -- the whole
    point of the judge contract is that CI always grades against one known,
    calibrated model, not whatever litellm's own default happens to be."""

    model: str

    def ask_json(self, prompt: str) -> dict[str, Any]:
        """One real call to the judge, forcing JSON output. Retries exactly once
        on a malformed response, then raises `JudgeError` -- never a silent
        `None` standing in for "the judge didn't answer properly".
        """
        import litellm  # lazy -- see module docstring

        last_error: Exception | None = None
        for _attempt in range(2):
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError) as exc:
                last_error = exc
                continue
        raise JudgeError(f"judge response wasn't valid JSON after 2 attempts: {last_error}")


def build_prompt(rubric: Rubric, input_text: str, output_text: str, tool_results: Any = None) -> str:
    criteria_lines = "\n".join(f"- {c.id}: {c.description}" for c in rubric.criteria)
    tool_results_section = f"\nTool results: {tool_results}\n" if tool_results is not None else ""
    criteria_keys = ", ".join(f'"{c.id}"' for c in rubric.criteria)
    return (
        "You are grading an AI agent's response against a rubric. "
        "Judge only what the rubric asks -- do not invent additional criteria.\n\n"
        f"Input: {input_text}\n\n"
        f"Output: {output_text}\n"
        f"{tool_results_section}\n"
        f"Criteria:\n{criteria_lines}\n\n"
        "Respond with JSON only, shaped exactly like this (one entry per "
        f"criterion id above: {criteria_keys}):\n"
        '{"criteria": {"<criterion_id>": {"pass": true, "reason": "..."}, ...}}'
    )


@dataclass
class GradeResult:
    rubric_id: str
    criteria: dict[str, dict[str, Any]]
    passed: bool


def grade(
    client: JudgeClient,
    rubric: Rubric,
    input_text: str,
    output_text: str,
    *,
    cassette: str,
    tool_results: Any = None,
    mode: str | None = None,
) -> GradeResult:
    """Grade one (input, output) pair against `rubric`, caching the verdict in
    `cassette` the same way an LLM/tool call is cached -- so a replay run never
    calls the judge at all. Reuses `replay/engine.resolve_call` directly: a
    judge verdict is just another kind of call, keyed by `(rubric.id, "judge",
    request)` the same way an LLM call is keyed by `(node, "llm", messages)`.

    Only `scale: pass_fail` rubrics are graded today -- see `rubric.py`'s own
    note on why `scale: "1-5"` validates but isn't scored yet.
    """
    if rubric.scale != "pass_fail":
        raise NotImplementedError(f"grade() only scores scale='pass_fail' rubrics today, got {rubric.scale!r}")

    resolved_mode = resolve_mode(mode)
    ctx = build_replay_context(cassette, resolved_mode)
    request = {"input": input_text, "output": output_text, "tool_results": tool_results}

    def real_call() -> dict[str, Any]:
        return client.ask_json(build_prompt(rubric, input_text, output_text, tool_results))

    data, _latency_s = resolve_call(ctx, rubric.id, "judge", request, real_call, extra={"model": client.model})
    data = to_plain_python(data)
    save_if_recording(cassette, resolved_mode, ctx)

    criteria_results = data.get("criteria", {}) if isinstance(data, dict) else {}
    passed = rubric.pass_threshold == "all" and all(
        bool(criteria_results.get(c.id, {}).get("pass")) for c in rubric.criteria
    )
    return GradeResult(rubric_id=rubric.id, criteria=criteria_results, passed=passed)
