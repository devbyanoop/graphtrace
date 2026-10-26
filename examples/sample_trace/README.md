# Sample: trace-to-test

A real, checked-in example of `graphtrace from-trace` end to end — not a toy, the actual command and actual output.

**`langfuse_export.json`** — a Langfuse-shaped trace export for a refund request against order `7890`: a `router` LLM call, then `lookup_order` failing once (`order service unavailable (attempt 1)`) before succeeding on retry. Hand-built to match a real production shape, not pulled from a live account (see `docs/guide/trace-to-test.md` for the current, honest state of live-export verification).

**How it was produced**, reproducible from the repo root:

```bash
PYTHONPATH=examples graphtrace from-trace examples/sample_trace/langfuse_export.json \
  --target refund_agent.graph:build_refund_graph_demo_with_retry \
  --test-name refund_sample \
  --out examples/sample_trace/test_refund_sample.py \
  --cassette examples/sample_trace/cassette.yaml
```

`PYTHONPATH=examples` is what makes `refund_agent` importable from the repo root the way `--target` needs — pytest gets this for free from `pyproject.toml`'s own `pythonpath` setting, but a plain CLI invocation doesn't, the same way it wouldn't in your own project unless `refund_agent`-equivalent is already on your path.

**`cassette.yaml`** — the real output: two `lookup_order` entries (the failed attempt and the successful retry), keyed with the `node/kind/any/occurrence` positional convention `graphtrace from-trace` cassettes use (see `docs/guide/trace-to-test.md` for why).

**`test_refund_sample.py`** — the real generated test, **unedited**, checked in exactly as written. Run it:

```bash
PYTHONPATH=examples pytest examples/sample_trace/test_refund_sample.py --gt-mode=replay -v
```

Passes — confirming the generated test genuinely exercises the real retry (both cassette entries get consulted, not just the first), not a cassette entry that happens to go unused.
