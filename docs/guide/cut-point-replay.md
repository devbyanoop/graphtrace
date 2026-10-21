# Cut-point replay (v0.2)

!!! warning "Planned for v0.2 — not built, and gated on v0.1 shipping first"
    Nothing on this page is implemented. See [Roadmap & status](../project/roadmap.md).

## What it's for

A production incident happens. You record it as a cassette. You fix the code. Now you want to know: **does the fix actually work** — not "does replaying the exact same broken run still look broken," which is all plain `replay` mode can tell you.

`cut` mode answers that: replay the recorded incident up to the point it went wrong, then run the *fixed* code live from there. Everything before the cut point is free and instant (served from the cassette); everything after is real, because that's the part actually being tested.

```python
run_graph(graph, inputs, cassette="incident_2026_11_03.yaml", mode="cut", cut_at="lookup_order")
```

Produces an ordinary `Trajectory` — every existing `assert_*` function works against it unmodified.

## Where this comes from

Adopted deliberately from Chronicle (Microsoft, arXiv 2609.20625), which reports: bit-stable replay across 20 repetitions, zero model calls on the replayed portion, and — in a mutation-testing study — catching every mutant that let a recorded unsafe action through, where naively stubbing every boundary caught none. That last result is the actual evidence this is worth adopting: a cruder "replay everything, assert the final state" approach misses exactly the kind of regression cut-point replay is designed to catch.

## What happens if the recording and the live run disagree *before* the cut point

The run fails loudly — "recorded run diverged before the cut point" — rather than silently serving a stale cassette entry. If a loop now runs a different number of times, or a retry was added or removed, that's a sign the recording no longer describes what's actually happening up to the cut, and trusting it past that point would validate the fix against the wrong starting state.

## Relationship to trace-to-test

[Trace-to-test](trace-to-test.md) already turns a production trace into a cassette and a generated test. `cut` mode is what makes that generated test useful for verifying a *fix*, not just for locking in whatever the agent currently does (which might still be the bug).
