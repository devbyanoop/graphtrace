"""`graphtrace calibrate`: score a human-labelled calibration set with the judge,
reporting agreement % and Cohen's kappa **per criterion** -- not one overall
number, since different criteria can have genuinely different reliability.

Calibration always calls the real judge, never a cassette -- caching a
calibration run would defeat the entire point of it (confirming the judge's
*current* real behaviour against human judgment), so this module never touches
`replay/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from graphtrace.judge.client import JudgeClient, build_prompt
from graphtrace.judge.errors import JudgeError
from graphtrace.judge.rubric import Rubric


@dataclass
class CriterionCalibration:
    agreement: float
    kappa: float
    n: int


def load_calibration_cases(path: Path | str, rubric_id: str) -> list[dict]:
    """`evals/calibration.jsonl`: one JSON object per line, filtered to the
    rubric being calibrated (the file can hold cases for several rubrics)."""
    path = Path(path)
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        if case.get("rubric") == rubric_id:
            cases.append(case)
    return cases


def calibrate(client: JudgeClient, rubric: Rubric, labels_path: Path | str) -> dict[str, CriterionCalibration]:
    """Returns one `CriterionCalibration` per criterion in `rubric`.

    A case the judge fails to answer (a malformed response after retry) is
    skipped for *that* criterion's tally, not treated as a judge "fail" --
    a `JudgeError` is itself a signal the judge is broken for that input, which
    low agreement should already surface; silently counting it as disagreement
    would double-penalize the same underlying problem.
    """
    cases = load_calibration_cases(labels_path, rubric.id)
    human_labels: dict[str, list[bool]] = {c.id: [] for c in rubric.criteria}
    judge_labels: dict[str, list[bool]] = {c.id: [] for c in rubric.criteria}

    for case in cases:
        prompt = build_prompt(rubric, case["input"], case["output"], case.get("tool_results"))
        try:
            data = client.ask_json(prompt)
        except JudgeError:
            continue
        criteria_results = data.get("criteria", {}) if isinstance(data, dict) else {}
        for criterion in rubric.criteria:
            if criterion.id not in case.get("human", {}):
                continue
            human_labels[criterion.id].append(bool(case["human"][criterion.id]))
            judge_labels[criterion.id].append(bool(criteria_results.get(criterion.id, {}).get("pass")))

    results: dict[str, CriterionCalibration] = {}
    for criterion in rubric.criteria:
        human, judged = human_labels[criterion.id], judge_labels[criterion.id]
        if not human:
            results[criterion.id] = CriterionCalibration(agreement=0.0, kappa=0.0, n=0)
            continue
        agreement = sum(1 for h, j in zip(human, judged) if h == j) / len(human)
        kappa = cohen_kappa_score(human, judged, replace_undefined_by=0.0)
        results[criterion.id] = CriterionCalibration(agreement=agreement, kappa=float(kappa), n=len(human))
    return results
