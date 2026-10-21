# Recording & replay

## `run_graph` / `arun_graph` — real, tested

```python
from graphtrace import run_graph

traj = run_graph(graph, inputs, cassette="unused", resume=[True])
```

`run_graph(graph, inputs, *, cassette, config=None, thread_id="gc-test", resume=None) -> Trajectory` (and an async `arun_graph`) drives a compiled LangGraph via `graph.stream(..., stream_mode=["debug"], subgraphs=True)`, feeding every event to a recorder that builds up a [`Trajectory`](trajectory-model.md). A `MemorySaver` checkpointer is attached automatically if the graph doesn't already have one. `resume` is a list of values fed to successive human-in-the-loop interrupts, one per pause, in order — if the graph pauses and there's no next value queued, `run_graph` stops there and returns with `interrupted=True`.

Verified against real compiled graphs, not just hand-built `Trajectory` fixtures: linear paths, parallel fan-out, loops/retries, interrupt-and-resume (including that a resumed node doesn't show up twice), tool-call and LLM-call capture, and a tool error that propagates or that a node catches — 24 tests in `tests/test_runner.py` and `tests/test_demo_agent.py`. **Not yet tested**: subgraphs. `namespace` is threaded through from LangGraph's own `subgraphs=True` output, but nothing in the current test suite actually nests a subgraph inside a node to confirm it comes through correctly — a real gap, not an oversight being hidden.

## `cassette` is accepted but not used yet

```python
run_graph(graph, inputs, cassette="incident_2026_11_03.yaml")
```

The parameter exists today so the signature matches what it will be once replay exists — passing it has **no effect yet**. Every call today runs the same way: real node execution, real LLM/tool calls if the graph makes any. This is Week 1's one honest, temporary scope limit, not a silent gap.

## Still planned — modes, cassettes, matching

!!! warning "Planned — not built yet"
    Nothing below this line is implemented.

- Interceptors patch `BaseChatModel.invoke`/`ainvoke` and `BaseTool.invoke`/`ainvoke` (**corrected from the original design's `_generate`/`_run`** — see the roadmap for why) while a node is active, capturing model/tool calls without the graph's own code changing. This part **is** built (see above) — what's still planned is everything cassette-related below.
- Five modes: `replay` (default in CI — no real LLM/tool calls, cassette entries must already exist), `record` (overwrites a cassette), `record_missing` (appends only unmatched requests), `live` (what every call does today, by default), `cut` (v0.2 — see [Cut-point replay](cut-point-replay.md)).
- Cassettes are YAML, keyed by `node + kind + hash(normalized request) + occurrence index`, with a redaction hook masking secrets/PII before anything is written to disk.

## Why this was the hard part

Turning a *real* LangGraph run into a correct `Trajectory` needed two real corrections to the original design along the way — not assumptions that happened to be wrong in theory, but things confirmed wrong by running real code: the originally-planned `contextvars` call-attribution mechanism doesn't survive LangGraph's actual `ThreadPoolExecutor`-based parallel execution, and the originally-planned `_generate`/`_agenerate` patch target is never actually called by any real chat model. Both are detailed in the project's internal plan doc.

## Statistical live-mode verification (v0.2)

!!! warning "Planned for v0.2 — not built, and not scheduled for v0.1"

A single `live`-mode run is a single run, pass or fail — not strong evidence on its own, since real models are genuinely non-deterministic. `assert_live_stable(target, inputs, trials=50, pass_threshold=0.85, alpha=0.05, check=...)` (adopted from AgentAssay, arXiv 2603.02601) runs the same call repeatedly and reports **Pass / Fail / Inconclusive** from a Wilson-score confidence interval on the pass rate — refusing to force a binary call when there isn't yet enough evidence, rather than trusting a raw percentage that can look fine while actually being too close to call. Scoped to opt-in, scheduled drift checks only (each trial is a real, billed call) — never the default PR path.
