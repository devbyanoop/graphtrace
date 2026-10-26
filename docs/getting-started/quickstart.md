# Quickstart

!!! note "Every example below is real and verified"
    This walks through a real `run_graph()` call against `examples/refund_agent`, the project's own demo agent, then the real `graphtrace` CLI against a real trace fixture. Run from a cloned checkout (`git clone https://github.com/devbyanoop/graphtrace.git`) — the commands below assume the repo root as your working directory.

## 1. Install

Follow [Installation](installation.md) first.

## 2. Run the demo agent

`examples/refund_agent` is a small, real, compiled LangGraph matching the architecture plan's own running example: `router -> lookup_order -> human_approval -> refund`. It uses LangChain's own fake chat model, so this needs no API key and makes no real network call.

```python
import sys
sys.path.insert(0, "examples")  # repo root; lets this import examples/refund_agent

import itertools
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from refund_agent.graph import build_refund_graph

from graphtrace import run_graph

model = GenericFakeChatModel(messages=itertools.repeat(AIMessage(content="ok")))
graph = build_refund_graph(model)

# resume=[True] supplies the human-in-the-loop approval the graph pauses for --
# see step 5 below for what happens if you leave it out.
traj = run_graph(graph, {"order_id": "1234"}, cassette="tests/cassettes/quickstart.yaml", mode="record", resume=[True])
print(traj.path)           # ['router', 'lookup_order', 'human_approval', 'refund']
print(traj.final_state)    # {..., 'approved': True, 'refunded': True}
```

## 3. Replay it with zero real calls

Run it again with `mode="replay"` instead of `"record"` — the exact same `assert_*` calls below pass, but the fake chat model and the `get_order` tool are never actually invoked; every response comes straight from `tests/cassettes/quickstart.yaml`. This is what makes CI fast, free, and deterministic once a cassette exists — see [Recording & replay](../guide/recording-and-replay.md).

```python
traj = run_graph(graph, {"order_id": "1234"}, cassette="tests/cassettes/quickstart.yaml", mode="replay", resume=[True])
```

## 4. Assert on it

```python
from graphtrace import assert_path, assert_tool_called, assert_budget

assert_path(traj, ["router", "lookup_order", "human_approval", "refund"])
assert_tool_called(traj, "get_order", args_match={"order_id": "1234"})
assert_budget(traj, latency_s=5)
```

All three pass against the real trajectory above. Change `"1234"` to `"9999"` in the second assertion and it raises `TrajectoryAssertionError` with a readable message telling you which call's args didn't match — try it.

## 5. What happens without `resume`

```python
traj = run_graph(graph, {"order_id": "1234"}, cassette="tests/cassettes/quickstart.yaml", mode="live")  # no resume given

from graphtrace import assert_interrupted_at
assert_interrupted_at(traj, "human_approval")
assert traj.interrupted is True
assert "refund" not in traj.path  # the graph paused before ever reaching it
```

The graph genuinely pauses at `human_approval` (a real LangGraph `interrupt()`), and `run_graph` returns with `interrupted=True` instead of guessing an answer or hanging.

## 6. Build a Trajectory by hand instead

You don't need a real graph to use the assertion API at all — a `Trajectory` is just a list of `Step`s plus a final state. See [Core concepts](../guide/trajectory-model.md) for the shape, or `tests/test_assertions.py` for a worked example used throughout this project's own test suite.

## 7. The CLI: turn a production trace into a test

No real LangGraph run needed for this part — `graphtrace from-trace` converts an *exported* trace (Langfuse/OTel/LangSmith) straight into a cassette plus a generated pytest file:

```bash
graphtrace from-trace my_export.json --target my_app.graph:build_graph
```

Writes `tests/cassettes/from_my_export.yaml` and `tests/test_from_my_export.py` — the generated file replays the observed run and asserts every node that made a captured call was visited, with a `# TODO: assert the correct behaviour` marker for you to tighten by hand. See [Trace-to-test](../guide/trace-to-test.md) for the real, honest limits (a node with no LLM/tool call is invisible to any trace importer, so the generated assertion is deliberately looser than `assert_path`).

## 8. The CLI: grade real outputs with a judge

```bash
graphtrace calibrate --rubric refund_reply
```

Scores a `litellm`-backed judge against a human-labelled calibration set, reporting agreement % and Cohen's kappa per criterion — see [Judge & calibration](../guide/judge-and-calibration.md).

## 9. The CLI: a static HTML report

```bash
pytest --gt-mode=replay --gt-report=reports/graphtrace.html
```

Every test's trajectory, judge verdicts, and cost/latency totals in one page, no server — see [Report & CI](../guide/report-and-ci.md).

## What's next

- Full [Assertions](../guide/assertions.md) reference for every `assert_*` function, including ones not shown above.
- [Configuration](../guide/configuration.md) for `graphtrace.yaml` — the cassette directory, budgets, judge model, and report path all come from there once a project wants to stop repeating them per test.
