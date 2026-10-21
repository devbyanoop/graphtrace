# Trace-to-test

!!! warning "Planned — not built yet"
    Nothing on this page is implemented. See [Roadmap & status](../project/roadmap.md) for what's actually real today.

## Scope

`graphtrace from-trace trace.json --target my_app.graph:build_graph` will turn a production trace export (Langfuse, OTel/OpenInference, or LangSmith) into:

- a cassette built from the trace's LLM and tool spans, and
- a generated pytest file that replays that exact run and asserts the **observed** path, with a `# TODO: assert the correct behaviour` marker where a developer fills in what *should* have happened.

A trace importer only has to implement one small protocol (`can_read`, `to_normalized`) — adding a new source is one file, not a change to the core conversion logic.

## A real limit, not an oversight

If the trace includes a human-in-the-loop interrupt, the generated test gets a `resume=[...]  # TODO: fill in what the human actually sent` stub instead of a guessed value. A resume payload genuinely isn't recoverable from the trace alone — the generated test is deliberately left failing until a human fills it in, rather than silently guessing wrong.
