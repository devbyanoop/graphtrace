# FAQ

This page grows as real questions come in. It starts with a few that are already answerable from design decisions made so far.

## Will graphtrace ever call a real LLM or tool during a CI run?

Not once recording/replay exists (it doesn't yet — see [Roadmap & status](roadmap.md)). The intended default for CI is `replay` mode: responses come from a cassette on disk, and the real LLM/tool are never reached. `live` mode (real calls) is meant for a separate, nightly drift check, not the default PR gate.

## Why does `core` refuse to import anything LangGraph-related?

So everything built on top of a `Trajectory` — assertions, budgets, the eventual report — stays reusable no matter what produced that `Trajectory`. See [Core concepts: why core stays framework-agnostic](../guide/trajectory-model.md#why-core-stays-framework-agnostic).

## Why does `assert_path` only support one `...` gap?

Supporting several, non-overlapping gaps is possible, but makes failure messages genuinely ambiguous ("which gap swallowed the mismatch?"), and nothing in the project's own goals needs it yet. See [Assertions: assert_path](../guide/assertions.md#assert_path).

## How is `assert_retries` different from `assert_visits`?

`assert_visits` just counts how many times a node ran — fine for a node that's a normal loop body. `assert_retries` counts only re-entries that happened *after that node's own tool call failed* — an actual retry, not a designed loop. See [Assertions: assert_retries](../guide/assertions.md#assert_retries).

## Will graphtrace ever support CrewAI or AutoGen?

Yes, planned for **v0.4** (gated on v0.1 shipping first) — see [Multi-framework support](../guide/multi-framework.md). That's a real, designed phase now, not an open-ended "maybe someday": the `core` data model needs zero changes (see the previous question), and the one real architecture decision this needs — splitting the LangGraph-specific layer into `frameworks/base.py` (a shared contract) + `frameworks/langgraph/` — is already being built that way in v0.1, specifically so v0.4 is a new package, not a rename of a shipped one.

## Why does graphtrace borrow designs from other tools instead of inventing its own?

Because a validated idea is worth more than a novel one. `assert_tool_calls_match`'s four modes come straight from `langchain-ai/agentevals` (the official LangGraph ecosystem's own answer, already adopted widely); cut-point replay comes from a Microsoft research paper ([Chronicle, arXiv 2609.20625](../guide/cut-point-replay.md)) with a mutation-testing study backing it; the statistical live-mode verdict comes from another published paper (AgentAssay, arXiv 2603.02601, see [Recording & replay](../guide/recording-and-replay.md#statistical-live-mode-verification-v02)). Reinventing any of these from scratch, worse, to avoid "just copying," would be a worse outcome for anyone actually using graphtrace. See [Roadmap & status](roadmap.md) for where each one stands today.

## Something's missing from this FAQ

Open an issue, or see [Contributing](contributing.md).
