# Observability export (v0.2)

!!! warning "Planned for v0.2 — not built yet, and not scheduled for v0.1"
    This is scoped in the project's internal plan but explicitly out of scope for v0.1. See [Roadmap & status](../project/roadmap.md).

## Scope

[Trace-to-test](trace-to-test.md) pulls traces **in** from observability tools to build tests. This closes the loop the other way: pushing graphtrace's own results **out**, so a nightly `live` run shows up as a quality signal next to an app's real production traces, not just as a CI artifact nobody reopens once the build goes green.

- Mapped from the existing per-test JSON result — no new data model: test name/node path becomes a span name/attributes, pass/fail becomes span status, `cost_usd`/`latency_s`/`tokens_total` become span metrics, a judge verdict becomes a span attribute, a budget breach becomes a span event.
- Exporters: OpenTelemetry (OTLP), Langfuse (one trace per test, scored per rubric), or a generic webhook.
- Export is **opt-in and off by default**, best-effort (a failed export logs a warning, never fails the test), and runs through the same redaction hook cassettes use before anything leaves the local machine.
