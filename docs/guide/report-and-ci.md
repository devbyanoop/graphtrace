# Report & CI

```bash
pytest --gt-mode=replay --gt-report=reports/graphtrace.html
```

A pytest hook (`plugin.py`'s `pytest_runtest_makereport`/`pytest_sessionfinish`) collects one `TestResult` per test and, at session end, renders a static HTML page from them — one entry per test, an expandable trajectory tree, judge verdicts with their reasons, cost/latency/token totals. No server, no hosted dashboard, by design. A `results.json` is written alongside the HTML (same path, `.json` extension) — `graphtrace report results.json` re-renders the HTML from it standalone, for re-rendering later or merging results across CI shards without re-running anything.

Every test that calls the `run_graph` fixture gets its real `Trajectory` attached to its report entry automatically — no extra code in the test itself. A test that fails before its body even runs (a broken fixture) is still reported, as `"error"` rather than `"failed"`, with no trajectory to show (there wasn't one to capture) but with the real failure message.

## What the report shows

- **Summary row**: total/passed/failed/errored counts, total cost (USD) and total latency across every test.
- **Per test**: outcome, the error message on failure, judge verdicts (pass/fail per criterion, with the judge's own stated reason), and the trajectory tree — every step's node, kind, latency, and each LLM call (model, token counts, cost, output) or tool call (name, args, result or error) inside it.
- Rendered via Jinja2 with `autoescape=True`, passed explicitly and unconditionally — not inferred from the template's filename. Trajectory state, tool args/results, and an LLM's own output can all hold arbitrary client data (a user's refund note, a crafted prompt injection attempt...), the same XSS concern kbguard's own report has. Confirmed with a real test that a `<script>` tag anywhere in that data renders as escaped text, never live markup.

## CI

JUnit XML comes for free from pytest — no extra work needed for a CI dashboard to pick results up alongside (or instead of) the HTML report. `.github/workflows/graphtrace.yml` runs `pytest` (every test in this repo's own suite runs against fake chat models, so no provider key is needed) and `mkdocs build --strict` on every push/PR — graphtrace testing itself with its own tools, the same gate kbguard uses, scoped to what's actually configured here today (no `ruff`/`mypy` step — neither is wired up in this project yet, and a gate with nothing behind it would just be a step that always passes without checking anything).

A separate nightly `--gt-mode=live` workflow (catching real drift against real models) is documented here as the intended pattern but not itself built — it needs a real provider account and key to mean anything, and this project doesn't want to claim a workflow is "real" when it would do nothing without secrets nobody's configured yet.
