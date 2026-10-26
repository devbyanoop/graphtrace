# Sample: HTML report

`report.html` is a real, checked-in `graphtrace report` output — not a mockup. Three real `Trajectory` objects (from real `run_graph()` calls against `examples/refund_agent`), one real judge verdict (a real call to `groq/openai/gpt-oss-120b`, the same model this project's own `graphtrace.yaml` uses), and one deliberate, real failure (an assertion that's wrong on purpose, to show what a failing entry actually looks like — `expected final_state['refunded'] == False, got True`).

`results.json` is the JSON this was rendered from:

```bash
graphtrace report examples/sample_report/results.json \
  --out examples/sample_report/report.html \
  --title "graphtrace sample report"
```

reproduces `report.html` byte-for-byte from it (confirmed with a real `diff`, not assumed).

Open `report.html` directly in a browser to see it rendered.
