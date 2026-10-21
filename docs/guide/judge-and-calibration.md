# Judge & calibration

!!! warning "Planned — not built yet"
    Nothing on this page is implemented. See [Roadmap & status](../project/roadmap.md) for what's actually real today.

## Scope

- A judge call via `litellm` (so any provider works), with the model id **pinned** in config — never left to drift silently.
- Rubric files (YAML): a list of pass/fail criteria, forced JSON output from the judge, one retry on a malformed response before failing loudly (never a silent `None`).
- Judge responses cached inside the same cassette a test's LLM/tool calls live in, so a replay run never calls the judge.
- `graphtrace calibrate`: scores the judge against a human-labelled calibration set (20–50 cases per rubric), reports agreement % and Cohen's kappa per criterion. CI fails when agreement drops below a configured `min_agreement`.

## Why calibration, not just a judge call

A rubric and a pinned model aren't enough on their own to trust a judge's verdict — the calibration set is what proves the judge actually agrees with a human often enough to be worth listening to, and catches it the day a model swap or rubric edit quietly changes that.
