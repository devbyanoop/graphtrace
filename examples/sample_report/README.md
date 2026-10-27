# Sample: HTML report

`report.html` is a real, checked-in `graphtrace report` output — not a mockup. Three real `Trajectory` objects (from real `run_graph()` calls against `examples/refund_agent`), one real judge verdict (a real call to `groq/openai/gpt-oss-120b`, the same model this project's own `graphtrace.yaml` uses), and one deliberate, real failure (an assertion that's wrong on purpose, to show what a blocking entry actually looks like — `expected final_state['refunded'] == False, got True`).

All three tests are tagged with `@pytest.mark.graphtrace(description=..., severity=..., category=...)`-equivalent metadata (set directly via `TestResult.from_trajectory(...)` here, since this sample builds results by hand rather than through a real pytest run — see [Report & CI](../../docs/guide/report-and-ci.md) for the real marker) — so the report's **executive summary** (the verdict, the ranked blocking-issues list, the coverage breakdown) shows real content instead of the bare test-id fallback. The full per-test trajectory detail is still there, collapsed under a disclosure triangle below the summary.

`results.json` is the JSON this was rendered from:

```bash
graphtrace report examples/sample_report/results.json \
  --out examples/sample_report/report.html \
  --title "graphtrace sample report"
```

reproduces `report.html` byte-for-byte from it (confirmed with a real `diff`, not assumed).

Open `report.html` directly in a browser to see it rendered.
