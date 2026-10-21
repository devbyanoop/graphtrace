# Configuration

!!! warning "Planned — not built yet"
    There's no `graphtrace.yaml` loader yet (`config.py`, pydantic-validated, per the roadmap). See [Roadmap & status](../project/roadmap.md).

## Intended shape

```yaml
# graphtrace.yaml (project root)
target: my_app.graph:build_graph    # default factory; "target" not "graph" on purpose —
framework: langgraph                # see Multi-framework support (v0.4)
replay:
  mode: replay
  cassettes: tests/cassettes/
  normalize:
    - pattern: '\d{4}-\d{2}-\d{2}T[\d:.]+Z'
      replace: '<ts>'
  redact: [api_key, email, phone]
judge:
  model: <pinned model id>
  rubrics_dir: rubrics/
  calibration_set: evals/calibration.jsonl
  min_agreement: 0.8
budgets:
  default: {cost_usd: 0.10, latency_s: 15}
pricing_overrides: {}
report:
  html: reports/graphtrace.html
```

Nothing here is enforced yet — treat it as a preview of the schema, not a working config file.

`target`/`framework` use the generic names from the start (not `graph:`) even though `langgraph` is the only valid `framework` value until v0.4 — see [Multi-framework support](multi-framework.md) for why.
