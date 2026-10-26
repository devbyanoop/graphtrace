# Configuration

`graphtrace.yaml` (project root) is loaded and validated by `config.py` (pydantic, strict — an unknown field or an invalid value raises `pydantic.ValidationError` naming the actual field, not a silent ignore). A missing file is normal, not an error: every field falls back to its default.

```yaml
target: my_app.graph:build_graph    # default factory; "target" not "graph" on purpose —
framework: langgraph                # see Multi-framework support (v0.4) — the only valid value until then, enforced not just documented
replay:
  mode: replay
  cassettes: tests/cassettes/
  normalize:
    - pattern: '\d{4}-\d{2}-\d{2}T[\d:.]+Z'
      replace: '<ts>'
  redact: [api_key, email, phone]
judge:
  model: groq/openai/gpt-oss-120b     # a real, working litellm model string -- this project's own graphtrace.yaml uses exactly this
  rubrics_dir: rubrics/
  calibration_set: evals/calibration.jsonl
  min_agreement: 0.8
budgets:
  default: {cost_usd: 0.10, latency_s: 15}
pricing_overrides: {}
report:
  html: reports/graphtrace.html
```

## What's actually enforced today

- **`framework`**: `pydantic.Literal["langgraph"]` — any other value (including `crewai`/`autogen`, not built until v0.4) is rejected at load time, not just documented as unsupported.
- **`replay.mode`**: must be one of `record`/`replay`/`record_missing`/`live`.
- **`replay.cassettes`**: the `run_graph` pytest fixture uses this as the directory a test's auto-generated cassette path lives under (`<cassettes>/<test node id>.yaml`).
- **`budgets.default`**: the `run_graph` fixture applies this automatically after every call, via `core/budgets.py`'s `Budgets.check()` — a test that never calls `assert_budget(...)` itself still fails if the configured default is exceeded.
- **`judge`**: requires `model` if the section is present at all; consulted directly by `graphtrace calibrate` (`judge.model`/`judge.rubrics_dir`/`judge.calibration_set`/`judge.min_agreement`) — see [Judge & calibration](judge-and-calibration.md).
- **`report`**: `report.html` is the default output path for `graphtrace report` when `--out` isn't given — see [Report & CI](report-and-ci.md).

## Still planned

!!! warning "Planned — not built yet"
    - `replay.normalize`'s custom rules aren't wired through to `core/normalize.py` yet (the built-in timestamp/UUID rules are, and are used for every cassette match).
    - `replay.redact` isn't enforced — see [Recording & replay](recording-and-replay.md) for the current state of this gap.
    - `pricing_overrides` isn't read by `frameworks/pricing.py` yet.
