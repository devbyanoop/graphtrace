# Quickstart

!!! note "Every example below is real and verified"
    This walks through a real `run_graph()` call against `examples/refund_agent`, the project's own demo agent. Run from a cloned checkout (`git clone https://github.com/devbyanoop/graphtrace.git`) — the commands below assume the repo root as your working directory.

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
# see step 4 below for what happens if you leave it out.
traj = run_graph(graph, {"order_id": "1234"}, cassette="unused", resume=[True])
print(traj.path)           # ['router', 'lookup_order', 'human_approval', 'refund']
print(traj.final_state)    # {..., 'approved': True, 'refunded': True}
```

(`cassette="unused"` because record/replay doesn't exist yet — see [Recording & replay](../guide/recording-and-replay.md). The parameter is required today only to match the signature it will have once it does something.)

## 3. Assert on it

```python
from graphtrace import assert_path, assert_tool_called, assert_budget

assert_path(traj, ["router", "lookup_order", "human_approval", "refund"])
assert_tool_called(traj, "get_order", args_match={"order_id": "1234"})
assert_budget(traj, latency_s=5)
```

All three pass against the real trajectory above. Change `"1234"` to `"9999"` in the second assertion and it raises `TrajectoryAssertionError` with a readable message telling you which call's args didn't match — try it.

## 4. What happens without `resume`

```python
traj = run_graph(graph, {"order_id": "1234"}, cassette="unused")  # no resume given

from graphtrace import assert_interrupted_at
assert_interrupted_at(traj, "human_approval")
assert traj.interrupted is True
assert "refund" not in traj.path  # the graph paused before ever reaching it
```

The graph genuinely pauses at `human_approval` (a real LangGraph `interrupt()`), and `run_graph` returns with `interrupted=True` instead of guessing an answer or hanging.

## 5. Build a Trajectory by hand instead

You don't need a real graph to use the assertion API at all — a `Trajectory` is just a list of `Step`s plus a final state. See [Core concepts](../guide/trajectory-model.md) for the shape, or `tests/test_assertions.py` for a worked example used throughout this project's own test suite.

## What's next

- Record/replay cassettes are the next piece being built — track it on [Roadmap & status](../project/roadmap.md).
- See the full [Assertions](../guide/assertions.md) reference for every `assert_*` function, including ones not shown above.
