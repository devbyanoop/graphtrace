# Recording & replay

## `run_graph` / `arun_graph` — real, tested

```python
from graphtrace import run_graph

traj = run_graph(graph, inputs, cassette="tests/cassettes/my_test.yaml", mode="record", resume=[True])
```

`run_graph(graph, inputs, *, cassette, config=None, thread_id="gc-test", resume=None, mode=None) -> Trajectory` (and an async `arun_graph`) drives a compiled LangGraph via `graph.stream(..., stream_mode=["debug"], subgraphs=True)`, feeding every event to a recorder that builds up a [`Trajectory`](trajectory-model.md). A `MemorySaver` checkpointer is attached automatically if the graph doesn't already have one. `resume` is a list of values fed to successive human-in-the-loop interrupts, one per pause, in order — if the graph pauses and there's no next value queued, `run_graph` stops there and returns with `interrupted=True`.

`mode` resolves via an explicit argument, then the `GRAPHTRACE_MODE` environment variable, then `"live"` — unchanged Week 1 behaviour unless something actually opts in. Through the `run_graph` **pytest fixture**, it also resolves from `pytest --gt-mode=<mode>`.

**Subgraphs** (a node whose value is itself a compiled graph) are handled transparently, not collapsed into one opaque step. `Step.kind` gets a `subgraph_enter` step the moment the recorder detects a wrapping node (a child event arrives at a namespace that extends it), its own inner nodes become ordinary `"node"` steps in between, and a `subgraph_exit` step closes it out once the wrapping task's own checkpoint finally settles (necessarily last, since a subgraph's own completion can't be known before its children's). A real tool/LLM call made *inside* a subgraph's own node is attributed to that inner node directly, not lost or misattributed to the wrapper — confirmed by hand that `langgraph_checkpoint_ns` chains as `"sub:<id>|inner_a:<id>"`, and the existing `rsplit(':', 1)` attribution logic already extracts the innermost id correctly regardless of nesting depth. A real bug was found and fixed getting here: the recorder's state bookkeeping used to be global, single-namespace state, so a subgraph running *concurrently* with another outer-level node (parallel fan-out) could corrupt that sibling's recorded state with the subgraph's own unrelated partial state — fixed by scoping all of it per namespace. See `tests/test_runner.py`'s two subgraph tests, each confirmed to genuinely fail against the pre-fix recorder.

## The four modes — real, tested

| Mode | LLM/tool calls? | Cassette |
| --- | --- | --- |
| `replay` | No | Must exist; a missing entry raises `CassetteMismatchError` |
| `record` | Yes | Overwritten |
| `record_missing` | Only for unmatched requests | Appended |
| `live` (default) | Yes | Ignored entirely |

```python
# Record once (real calls happen, a cassette gets written):
run_graph(graph, inputs, cassette="tests/cassettes/happy_path.yaml", mode="record")

# Replay it later -- zero real calls, verified by patching the real underlying
# LangChain methods to raise if they're ever actually reached (tests/test_mode_aware_interceptors.py):
run_graph(graph, inputs, cassette="tests/cassettes/happy_path.yaml", mode="replay")
```

A replayed **failure** (a tool call that errored when it was recorded) re-raises as a plain `RuntimeError` carrying the original error message — the original exception's exact type isn't recoverable once serialized to YAML, an honest, documented limitation rather than a bug. A *live* exception (not from a cassette) always keeps its real type, so a node's own `except SomeSpecificError:` still works identically in every mode.

## Cassette format — real

A real cassette, recorded from `examples/refund_agent` (field order exactly as graphtrace itself writes it, not reordered for the doc):

```yaml
version: 1
graphtrace: 0.1.0
recorded_at: '2026-10-21T06:57:12.584162+00:00'
entries:
- key: router/llm/391132dd9ec5/0
  kind: llm
  node: router
  request: "[('human', 'What does the customer want for order 1234?')]"
  latency_s: 0.0021508999925572425
  response:
    content: ok
    tool_calls: []
    usage:
      input_tokens: 0
      output_tokens: 0
  model: GenericFakeChatModel
- key: lookup_order/tool/82eee3ce806e/0
  kind: tool
  node: lookup_order
  request:
    order_id: '1234'
  latency_s: 0.0006061000167392194
  response:
    status: delivered
    amount: 49.0
  name: get_order
```

YAML via `ruamel.yaml`'s **round-trip mode**, loaded and saved through `replay/cassette.py` — proven byte-identical on a load-then-immediately-save round trip (not just that the library loaded without erroring), which is what actually makes `record_missing` growing a cassette produce a clean PR diff: untouched entries stay untouched bytes.

The matching key is `node/kind/hash(normalized request)/occurrence` (`replay/matcher.py`) — `occurrence` is what lets the same node+request pair appear twice in one run (a retry loop) and still match its own two separate recorded entries, in order, confirmed with a real recorded-then-replayed retry sequence.

**A real bug found and fixed along the way**: `ruamel.yaml`'s round-trip mode wraps a loaded float (`49.0`) in a `ScalarFloat` subclass, to preserve its exact on-disk formatting. That's essential for the byte-identical round-trip above — but the *same* value, handed back into a replayed tool call, flows straight into LangGraph's own state and gets `msgpack`-serialized by its checkpointer, which doesn't know what a `ScalarFloat` is and raises `TypeError: Type is not msgpack serializable`. Confirmed by hand, not assumed. Fixed with `replay/cassette.to_plain_python()`, called only on a value about to flow back into real code — never on a value still destined for the cassette file itself, which is what keeps the round-trip guarantee above true at the same time.

## Still planned

!!! warning "Planned — not built yet"

- **Redaction.** The architecture plan's own `redact: [api_key, email, phone]` config field isn't enforced by anything yet — a recorded cassette today contains whatever the real request/response actually held, unmasked. Treat any cassette containing real secrets as sensitive until this exists.
- **Normalization rules beyond the built-in timestamp/UUID pair** (`core/normalize.py` has the mechanism; `graphtrace.yaml`'s `replay.normalize` list of custom rules isn't wired through to it yet).
- `cut` mode (v0.2 — see [Cut-point replay](cut-point-replay.md)).

## Why this was the hard part

Turning a *real* LangGraph run into a correct `Trajectory` needed two real corrections to the original design along the way — not assumptions that happened to be wrong in theory, but things confirmed wrong by running real code: the originally-planned `contextvars` call-attribution mechanism doesn't survive LangGraph's actual `ThreadPoolExecutor`-based parallel execution, and the originally-planned `_generate`/`_agenerate` patch target is never actually called by any real chat model. Both are detailed in the project's internal plan doc, along with the `ScalarFloat` finding above.

## Statistical live-mode verification (v0.2)

!!! warning "Planned for v0.2 — not built, and not scheduled for v0.1"

A single `live`-mode run is a single run, pass or fail — not strong evidence on its own, since real models are genuinely non-deterministic. `assert_live_stable(target, inputs, trials=50, pass_threshold=0.85, alpha=0.05, check=...)` (adopted from AgentAssay, arXiv 2603.02601) runs the same call repeatedly and reports **Pass / Fail / Inconclusive** from a Wilson-score confidence interval on the pass rate — refusing to force a binary call when there isn't yet enough evidence, rather than trusting a raw percentage that can look fine while actually being too close to call. Scoped to opt-in, scheduled drift checks only (each trial is a real, billed call) — never the default PR path.
