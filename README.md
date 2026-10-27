# graphtrace

**Regression testing for LangGraph agents — as a pytest plugin.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: v0.1 complete](https://img.shields.io/badge/status-v0.1%20complete-brightgreen.svg)](docs/project/roadmap.md)

## The problem

A LangGraph agent with branches, retries, and tool calls can look fine on a lucky run while silently changing *how* it got there — a node path that changed, a retry that's now firing on every single call, a tool call that quietly stopped happening, a human-approval gate the agent learned to route around. A test that only checks the final answer text can't catch any of that, because the final answer can still look right while the path that produced it is wrong, expensive, or unsafe.

graphtrace makes that path a first-class, assertable thing — the same way a web framework lets you assert on a response's status code and headers, not just the rendered HTML.

## What it does

- **`run_graph(graph, inputs, ...)`** — drives a real compiled LangGraph and records every node visited, every state change, and every tool/LLM call into a `Trajectory`. No code changes to the graph under test; interception happens entirely through patching, undone automatically when the run ends. Subgraphs are handled transparently (inner nodes show up as their own steps, bracketed by `subgraph_enter`/`subgraph_exit`), not collapsed into one opaque step that would silently drop their LLM/tool calls.
- **A small, readable assertion API, 12 functions** — `assert_path`, `assert_visited`/`assert_visits`, `assert_state`/`assert_state_at`, `assert_tool_called`/`assert_tool_not_called`, `assert_tool_calls_match` (strict/unordered/subset/superset), `assert_retries`, `assert_interrupted_at`, `assert_budget`. Every failure raises with a readable diff, not a repr of two values.
- **Record/replay cassettes** — a run's LLM and tool calls are recorded to YAML once, then replayed with zero real calls on every CI run after that. `record` / `replay` / `record_missing` / `live` modes, set via the `GRAPHTRACE_MODE` env var or `pytest --gt-mode=<mode>`.
- **An LLM-as-judge contract with calibration** — a rubric-graded verdict via `litellm` (any provider), cached in the same cassette mechanism as any LLM/tool call, plus `graphtrace calibrate` scoring the judge against a human-labelled set (agreement % and Cohen's kappa per criterion). Verified against a real, live Groq model, not just a faked one.
- **Trace-to-test** — `graphtrace from-trace export.json --target my_app.graph:build_graph` turns a real production trace export (Langfuse/OTel/LangSmith) into a cassette + a generated pytest file, with an honest `assert_visited`-based assertion (not an overclaimed exact path) and a `# TODO` marker for what should actually happen.
- **A static HTML report with a real go/no-go executive summary** — `pytest --gt-report=report.html` renders a release verdict (**READY** / **NOT READY, N blocking issues**), blocking issues ranked by severity with plain-language descriptions (`@pytest.mark.graphtrace(description=..., severity=..., category=...)`), and a coverage breakdown by category — the layer a QA lead, manager, or auditor actually needs, not just an engineer. The full per-test trajectory tree, judge verdicts, and cost/latency totals are still there, collapsed below. XSS-safe by construction (Jinja2 `autoescape=True`), no server.
- **Cost, latency, and token budgets as assertions** — `assert_budget(traj, cost_usd=0.05, latency_s=8)` fails the same way a path or tool-call assertion does, computed from real recorded usage, never an estimate.
- **pytest-native, plus a real CLI** — `pip install` registers the plugin automatically (confirmed from a real built wheel, not just an editable install) and installs a `graphtrace` console command (`from-trace` / `calibrate` / `report`). A `run_graph` fixture is available in any test with no `-p` flag, and auto-applies a project's configured default budget even if the test itself never calls `assert_budget`.

## Example

Real, runnable code — not pseudocode — against `examples/refund_agent`, the bundled demo agent (`router -> lookup_order -> human_approval -> refund`, using LangChain's own fake chat model, so this needs no API key):

```python
from graphtrace import assert_path, assert_tool_called, run_graph

# First run: records real calls to tests/cassettes/refund_happy_path.yaml
traj = run_graph(
    graph, {"order_id": "1234"},
    cassette="tests/cassettes/refund_happy_path.yaml",
    mode="record",
    resume=[True],  # supplies the human-in-the-loop approval the graph pauses for
)

assert_path(traj, ["router", "lookup_order", "human_approval", "refund"])
assert_tool_called(traj, "get_order", args_match={"order_id": "1234"})
```

Change `mode="record"` to `mode="replay"` on a later run and the exact same assertions pass with **zero** real LLM or tool calls — proven in this repo's own test suite by patching the real underlying LangChain methods to raise if they're ever actually reached.

The CLI, against a real exported trace and a real judge:

```bash
# A production trace export -> a cassette + a generated pytest file
graphtrace from-trace prod_export.json --target my_app.graph:build_graph

# Score an LLM-as-judge against a human-labelled set (real agreement %, real Cohen's kappa)
graphtrace calibrate --rubric refund_reply

# Every test's trajectory, judge verdicts, and cost/latency totals -> one static HTML page
pytest --gt-mode=replay --gt-report=reports/graphtrace.html
```

## Supported frameworks

| Framework | Status |
| --- | --- |
| [LangGraph](https://github.com/langchain-ai/langgraph) | ✅ Built (v0.1) |
| CrewAI | Planned — v0.4 |
| AutoGen | Planned — v0.4 |

The recording layer is built behind a `FrameworkRunner` protocol specifically so CrewAI/AutoGen support (v0.4) is a new package later, not a rewrite of this one — see [Multi-framework support](docs/guide/multi-framework.md).

## Installation

Not yet published to PyPI — install from source:

```bash
git clone https://github.com/devbyanoop/graphtrace.git
cd graphtrace
python -m venv .venv
source .venv/bin/activate   # Windows (Git Bash): source .venv/Scripts/activate

pip install -e ".[dev]"
```

See [Installation](docs/getting-started/installation.md) for the full breakdown, including the `docs` extra for building the documentation site locally.

## Quickstart

The bundled demo agent needs no vector database, API key, or setup of your own:

```bash
pytest tests/test_demo_agent.py -v
```

Full walkthrough — building a `Trajectory` by hand, running it against the real demo agent, and what happens on an unresolved interrupt — in [Quickstart](docs/getting-started/quickstart.md).

Pointing this at your own LangGraph agent is a matter of calling `run_graph(your_compiled_graph, your_inputs, cassette=...)` in your own test, not a framework migration.

## Documentation

Full guide (core concepts, every assertion, recording & replay, judge & calibration, trace-to-test, report & CI, configuration, the CLI, the roadmap) lives under [`docs/`](docs/) — build it locally with `mkdocs serve`, or browse the Markdown directly on GitHub.

- [Getting started](docs/getting-started/installation.md) / [Quickstart](docs/getting-started/quickstart.md)
- [Assertions reference](docs/guide/assertions.md)
- [Recording & replay](docs/guide/recording-and-replay.md)
- [Judge & calibration](docs/guide/judge-and-calibration.md)
- [Trace-to-test](docs/guide/trace-to-test.md)
- [Report & CI](docs/guide/report-and-ci.md)
- [CLI reference](docs/reference/cli.md)
- [Configuration reference](docs/guide/configuration.md)
- [Roadmap & status](docs/project/roadmap.md) — what's actually built vs. planned, updated the same day anything changes
- [Case study](docs/project/case-study.md) — the five real bugs found building this, and what they say about how it was built

## Project status

**v0.1 is complete — 204 tests pass, full regression.** The core `Trajectory` model, the full 12-function assertion API, the real LangGraph recording layer (including subgraphs), record/replay cassettes, the judge contract with calibration (verified against a real live model), trace-to-test from a production export, a real `graphtrace` CLI, the HTML report, and CI are all built and verified — not just against mocks and hand-built fixtures, but against real upstream sources (confirmed real `litellm`/Groq calls, the real `openinference-semantic-conventions` source, a real wheel build in a genuinely isolated clean venv). Several real bugs were found and fixed along the way rather than assumed away — see [Roadmap & status](docs/project/roadmap.md) for the full, honest account of each one. v0.2 (security test pack, observability export, cut-point replay, statistical live-mode verification) and v0.4 (CrewAI/AutoGen support) are designed but not started.

## Contributing

Issues and PRs welcome. See [Contributing](docs/project/contributing.md) for how the test suite and docs policy work.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Anoop Singh.
