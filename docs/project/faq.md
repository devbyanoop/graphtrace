# FAQ

This page grows as real questions come in. It starts with a few that are already answerable from design decisions made so far.

## Will graphtrace ever call a real LLM or tool during a CI run?

Only if you set `mode="live"` or configure `--gt-mode=live`. The intended default for CI is `replay` mode: responses come from a cassette on disk (or, for an imported production trace, from `graphtrace from-trace`'s own generated cassette), and the real LLM/tool are never reached — confirmed by patching the *real* underlying `BaseChatModel.invoke`/`BaseTool.invoke` to raise if they're ever actually reached, not just by trusting the replay logic's own reasoning. `live` mode is meant for a separate, nightly drift check, not the default PR gate.

## Does the judge ever get cached the same way?

Yes — `grade()` reuses the exact same `resolve_call()` cassette machinery an LLM/tool call goes through, keyed by `(rubric.id, "judge", request)`. A replay-mode test run never calls the judge, confirmed against a real recorded cassette and a real Groq model, not just a fake one. See [Judge & calibration](../guide/judge-and-calibration.md).

## Why does `core` refuse to import anything LangGraph-related?

So everything built on top of a `Trajectory` — assertions, budgets, the HTML report — stays reusable no matter what produced that `Trajectory`. See [Core concepts: why core stays framework-agnostic](../guide/trajectory-model.md#why-core-stays-framework-agnostic).

## Why does `assert_path` only support one `...` gap?

Supporting several, non-overlapping gaps is possible, but makes failure messages genuinely ambiguous ("which gap swallowed the mismatch?"), and nothing in the project's own goals needs it yet. See [Assertions: assert_path](../guide/assertions.md#assert_path).

## Why does a generated test from `graphtrace from-trace` assert `assert_visited`, not `assert_path`?

Because a trace importer only ever sees LLM/tool *calls* — a node that makes neither (a routing branch, a human-in-the-loop interrupt point) is invisible to every source platform's own export. Claiming to assert the *exact* path from spans alone would be a false promise, not a reasonable approximation. See [Trace-to-test](../guide/trace-to-test.md).

## How is `assert_retries` different from `assert_visits`?

`assert_visits` just counts how many times a node ran — fine for a node that's a normal loop body. `assert_retries` counts only re-entries that happened *after that node's own tool call failed* — an actual retry, not a designed loop. See [Assertions: assert_retries](../guide/assertions.md#assert_retries).

## Does graphtrace handle subgraphs (a node whose value is itself a compiled graph)?

Yes, transparently — the subgraph's own inner nodes show up as their own steps (so a real LLM/tool call made inside one is never lost), bracketed by `subgraph_enter`/`subgraph_exit` steps marking the boundary. This needed a real bug fix to get right: see [Recording & replay](../guide/recording-and-replay.md) for the full account.

## Will graphtrace ever support CrewAI or AutoGen?

Yes, planned for **v0.4** (gated on v0.1 shipping first) — see [Multi-framework support](../guide/multi-framework.md). That's a real, designed phase, not an open-ended "maybe someday": the `core` data model needs zero changes (see the previous question), and the one real architecture decision this needs — splitting the LangGraph-specific layer into `frameworks/base.py` (a shared contract) + `frameworks/langgraph/` — was already built that way in v0.1, specifically so v0.4 is a new package, not a rename of a shipped one.

## Why does graphtrace borrow designs from other tools instead of inventing its own?

Because a validated idea is worth more than a novel one. `assert_tool_calls_match`'s four modes come straight from `langchain-ai/agentevals` (the official LangGraph ecosystem's own answer, already adopted widely); cut-point replay comes from a Microsoft research paper ([Chronicle, arXiv 2609.20625](../guide/cut-point-replay.md)) with a mutation-testing study backing it; the statistical live-mode verdict comes from another published paper (AgentAssay, arXiv 2603.02601, see [Recording & replay](../guide/recording-and-replay.md#statistical-live-mode-verification-v02)). Reinventing any of these from scratch, worse, to avoid "just copying," would be a worse outcome for anyone actually using graphtrace. See [Roadmap & status](roadmap.md) for where each one stands today.

## Something's missing from this FAQ

Open an issue, or see [Contributing](contributing.md).
