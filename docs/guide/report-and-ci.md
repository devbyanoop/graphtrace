# Report & CI

```bash
pytest --gt-mode=replay --gt-report=reports/graphtrace.html
```

A pytest hook (`plugin.py`'s `pytest_runtest_makereport`/`pytest_sessionfinish`) collects one `TestResult` per test and, at session end, renders a static HTML page from them — one entry per test, an expandable trajectory tree, judge verdicts with their reasons, cost/latency/token totals. No server, no hosted dashboard, by design. A `results.json` is written alongside the HTML (same path, `.json` extension) — `graphtrace report results.json` re-renders the HTML from it standalone, for re-rendering later or merging results across CI shards without re-running anything.

Every test that calls the `run_graph` fixture gets its real `Trajectory` attached to its report entry automatically — no extra code in the test itself. A test that fails before its body even runs (a broken fixture) is still reported, as `"error"` rather than `"failed"`, with no trajectory to show (there wasn't one to capture) but with the real failure message.

## What the report shows

The report is laid out in two layers, for two different readers:

- **Executive summary** (always visible, no detail hidden behind a click): a release verdict — **READY** or **NOT READY, N blocking issues** — a ranked list of blocking issues (most severe first), and a coverage breakdown by category. This is the layer a QA lead, manager, consultant, or auditor actually needs to make a go/no-go call, without reading a single trajectory. See [Executive summary metadata](#executive-summary-metadata-pytestmarkgraphtrace) below for how to feed it real descriptions, severities, and categories instead of raw test names.
- **Technical detail** (collapsed by default, under a disclosure triangle): the same per-test trajectory tree this report has always had — every step's node, kind, latency, each LLM call (model, token counts, cost, output), each tool call (name, args, result or error), and judge verdicts with their reasons. This is the engineer's layer, for debugging *why* something failed once the summary says something needs attention.
- **Summary row**: total/passed/failed/errored counts, total cost (USD) and total latency across every test.
- Rendered via Jinja2 with `autoescape=True`, passed explicitly and unconditionally — not inferred from the template's filename. Trajectory state, tool args/results, an LLM's own output, and now a test's own `description`/`category` can all hold arbitrary data (a user's refund note, a crafted prompt injection attempt...), the same XSS concern kbguard's own report has. Confirmed with real tests that a `<script>` tag anywhere in any of these renders as escaped text, never live markup.

### Executive summary metadata: `@pytest.mark.graphtrace(...)`

Without this, the blocking-issues list falls back to showing a bare `test_id` — technically accurate, meaningless to anyone who doesn't already know what that test checks. Tag a test to make its entry readable by someone who doesn't read code:

```python
@pytest.mark.graphtrace(
    description="Customer should not be refunded without human approval",
    severity="critical",  # "critical" | "high" | "normal" (default) | "low"
    category="safety",    # any free-form string -- grouped into the coverage breakdown
)
def test_refund_requires_approval(run_graph):
    ...
```

All three are optional, and the marker is registered (`plugin.py`'s `pytest_configure`) so using it never triggers pytest's own "unknown marker" warning. An untagged test still reports correctly — `severity` defaults to `"normal"`, `description`/`category` fall back to showing the bare test id — so adopting this is additive, not required before the report shows any value at all.

**Severity never decides whether a failure blocks release** — only how it's *ranked and visually emphasized* among real failures. Graphtrace has no basis to decide a "low" severity failure is safe to ship anyway; that's a human call the report surfaces clearly (a red-bordered entry, last in the list) rather than one it ever makes for you. The verdict is simply: any failed or errored test means **NOT READY**, full stop.

**Coverage** is the set of distinct `category` values seen across the whole run, with counts — a quick answer to "what did this actually test," not just "how many things passed." If nothing in the run was tagged with a category, the report says so honestly (`coverage breakdown unavailable`) instead of fabricating one from test names.

**Not yet built, and worth knowing before expecting it**: a trend view across multiple runs ("better or worse than last time") would need a separate persistence design — where run history lives, how many runs to keep, whether it's per-branch — and isn't attempted here. Today's report is a single run's snapshot, full stop; comparing it to a previous one is still a manual, outside-the-report job.

## CI

JUnit XML comes for free from pytest — no extra work needed for a CI dashboard to pick results up alongside (or instead of) the HTML report. `.github/workflows/graphtrace.yml` runs `pytest` (every test in this repo's own suite runs against fake chat models, so no provider key is needed) and `mkdocs build --strict` on every push/PR — graphtrace testing itself with its own tools, the same gate kbguard uses, scoped to what's actually configured here today (no `ruff`/`mypy` step — neither is wired up in this project yet, and a gate with nothing behind it would just be a step that always passes without checking anything).

A separate nightly `--gt-mode=live` workflow (catching real drift against real models) is documented here as the intended pattern but not itself built — it needs a real provider account and key to mean anything, and this project doesn't want to claim a workflow is "real" when it would do nothing without secrets nobody's configured yet.
