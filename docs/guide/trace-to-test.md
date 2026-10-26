# Trace-to-test

```bash
graphtrace from-trace trace.json --target my_app.graph:build_graph
```

Turns a production trace export (Langfuse, OTel/OpenInference, or LangSmith) into:

- a cassette built from the trace's LLM and tool spans, and
- a generated pytest file that replays that run and asserts the nodes it actually visited, with a `# TODO: assert the correct behaviour` marker where you fill in what *should* have happened.

A trace importer only has to implement one small protocol (`can_read`, `to_normalized`) — adding a new source is one file (`importers/<name>.py`), not a change to `codegen/test_writer.py`'s own conversion logic. All three bundled importers converge on one shared `NormalizedTrace`/`NormalizedSpan` shape (`importers/base.py`).

None of the three bundled importers are verified against a live export from a real account — each one's own module docstring says exactly what's confirmed against that platform's documented export format versus representative-but-unverified, rather than overclaiming. Verify field names against a real export from your own project before trusting an importer beyond the bundled test fixtures, especially Langfuse's, whose data model has moved fast.

## Why the generated assertion is "visited", not the exact path

A trace importer only ever sees LLM and tool **calls** — a node that makes neither (a pure routing branch, a human-in-the-loop interrupt point) is invisible to every source platform's own export, not just to graphtrace's reading of it. Asserting the *exact* sequence of every node visited (`assert_path`) from an imported trace would therefore be a false promise: it would claim knowledge the trace never actually contained.

The generated test asserts `assert_visited(trajectory, node)` for each node that made a captured call instead — true, but deliberately looser than what `assert_path` can check. Once you know the real path, tighten the generated test by hand with `assert_path(...)` or `assert_tool_calls_match(...)` — both are regular graphtrace assertions, nothing generated-test-specific about them.

## A real limit, not an oversight

If the trace includes a human-in-the-loop interrupt, the generated test gets a `resume=[...]  # TODO: fill in what the human actually sent` stub instead of a guessed value. A resume payload genuinely isn't recoverable from the trace alone — the generated test is deliberately left needing that fill-in, rather than silently guessing wrong. In practice this field is always unset today: none of Langfuse/OTel/LangSmith's own export formats carry a universal "this run paused for human input" concept to detect it from (`NormalizedTrace.interrupted` defaults to `False`, documented as an open gap in `importers/base.py`). A trace whose *original* run really did pause and resume still generates a usable test either way — replaying it with no resume value simply reproduces the same pause, and the generated assertions (only ever about nodes visited *before* that point) still hold.

## A factory that needs real arguments

A LangGraph builder function commonly takes a chat model (or other required config) as an argument — this project's own demo agent, `build_refund_graph(chat_model, ...)`, is written exactly that way. `from-trace` actually imports your real `--target` and inspects its signature before generating anything: a factory with any required parameter gets

```python
graph = build_graph(...)  # TODO: this factory takes required arguments graphtrace can't know from the trace alone (e.g. a chat model) -- construct it the way your app really does
```

instead of a wrong, silently-broken `build_graph()`. A target that genuinely can't be imported at all (wrong `--target`, module not yet on the path) gets the same cautious placeholder, for the same reason: there's no way to know what it needs, and the generated file's own top-level import line surfaces the real problem either way.

## The cassette this produces doesn't hash-match like other cassettes

Every other cassette in this project matches a call by hashing its exact request (`node/kind/hash(request)/occurrence`) — a real retry against the real graph reproduces the same request, so the hash matches. An imported trace's own `request` field is whatever shape the *source platform* exported it in (Langfuse's JSON, an OTel attribute string, ...), which is never the same shape as the real `str(list[BaseMessage])` the live LangGraph interceptor hashes at replay time. There's no general transformation between the two.

`from-trace`'s cassettes use a position-only fallback instead — a literal `any` hash segment (`node/kind/any/occurrence`), matched by `replay/engine.py` only when the exact-hash lookup already missed, so it can never shadow a real recorded entry from a normal `record`-mode cassette. It trades exact-request verification for trusting the target graph to walk the same node sequence for the same `initial_input` — which is exactly what "replay the observed run" means for an imported trace, no more.
