# Report & CI

!!! warning "Planned — not built yet"
    See [Roadmap & status](../project/roadmap.md) for what's actually real today.

## Scope

- A pytest hook writes per-test JSON results; `graphtrace report` renders a static HTML page from them — one row per test, an expandable trajectory tree, failed assertions highlighted, judge verdicts with reasons, cost/latency totals. No server, no hosted dashboard, by design.
- JUnit XML comes for free from pytest — no extra work needed for a CI dashboard to pick results up.
- A GitHub Actions example ships with the project: `pytest --gt-mode=replay --gt-report=reports/graphtrace.html` on every PR, plus a separate nightly workflow running `--gt-mode=live` to catch drift against real models.
