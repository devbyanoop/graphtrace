# Judge & calibration

A judge call via `litellm` (so any provider works), with the model id **pinned** in `graphtrace.yaml`'s `judge.model` — never left to drift silently. Rubric files are small YAML documents: a list of pass/fail criteria the judge is asked to grade against, nothing invented beyond them.

```yaml
# rubrics/refund_reply.yaml
id: refund_reply
version: 1
criteria:
  - id: confirms_amount
    description: Reply states the refund amount.
  - id: no_promises
    description: Reply does not promise a specific delivery timeline.
scale: pass_fail
pass_threshold: all
```

`judge/client.py`'s `JudgeClient.ask_json()` forces JSON output (`response_format={"type": "json_object"}`), retries exactly once on a malformed response, then raises `JudgeError` — never a silent `None` standing in for "the judge didn't answer properly." `litellm` is imported lazily, inside that one method — constructing a `JudgeClient`, or hitting a cassette-replayed verdict in `replay` mode, never requires `litellm` to be installed at all.

## Judge verdicts are cached like any other call

`judge/client.py`'s `grade()` reuses `replay/engine.py`'s `resolve_call()` directly — the exact same cassette/replay machinery an LLM or tool call goes through. A judge verdict is keyed by `(rubric.id, "judge", request)`, the same way an LLM call is keyed by `(node, "llm", messages)`. The practical effect: once a judge call is recorded, a `replay`-mode test run never calls the judge again — grading stays free and deterministic in CI the same way the rest of the test is.

Only `scale: pass_fail` rubrics are actually scored today. A `scale: "1-5"` rubric loads and validates (so a project can write one ahead of need), but `grade()` raises `NotImplementedError` rather than guessing a pass/fail threshold from a 1–5 scale that was never designed.

## `graphtrace calibrate`

```bash
graphtrace calibrate --rubric refund_reply
```

Scores the judge against `graphtrace.yaml`'s `judge.calibration_set` — a JSONL file, one case per line, each holding `rubric`, `input`, `output`, and a `human` dict of ground-truth labels per criterion:

```json
{"rubric": "refund_reply", "input": "can I get a refund?", "output": "yes, $49.00 refunded", "human": {"confirms_amount": true, "no_promises": true}}
```

`judge/calibrate.py` reports **agreement % and Cohen's kappa per criterion** — not one overall number, since different criteria genuinely have different reliability. Calibration always calls the real judge, never a cassette: caching a calibration run would defeat the entire point of it (confirming the judge's *current* real behaviour against human judgment).

The CLI exits non-zero when any criterion with at least one labelled case falls below `judge.min_agreement` — the gate a CI pipeline would actually check. A criterion with zero labelled cases is reported and skipped, not silently treated as a pass or a failure.

A case the judge fails to answer (a malformed response even after the one retry) is skipped for that criterion's tally, not counted as disagreement — a `JudgeError` is itself a signal the judge is broken for that input, and low agreement should already surface that; double-penalizing the same underlying problem would just make the number harder to read.

## A real correction, found by checking the math by hand

This project's own internal build plan originally claimed "full disagreement scores kappa 0.0." That's wrong for balanced labels, and it was caught before anything shipped: a perfect inversion (every human `True` judged `False` and vice versa, on a balanced 2-2 split) scores **kappa = -1.0**, not 0.0 — a real, measurable anti-correlation, worse than chance, not merely "no better than chance." Verified by hand against the actual formula (κ = (p₀ − pₑ)/(1 − pₑ)) and against the real installed `sklearn.metrics.cohen_kappa_score`, not assumed from the inherited phrasing.

Kappa 0.0 — genuine chance-level agreement — needs a case with real, independent variance on both sides (e.g. human `[True, True, False, False]` vs. judge `[True, False, True, False]`), not a systematic flip. The distinction matters for reading your own calibration numbers: a kappa *near* -1 is a sharper, more specific warning sign ("the judge is anti-correlated with human judgment, maybe the rubric criterion is phrased backwards") than a kappa near 0 ("the judge adds no information").
