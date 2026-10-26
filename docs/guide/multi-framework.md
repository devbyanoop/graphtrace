# Multi-framework support: CrewAI & AutoGen (v0.4)

!!! warning "Planned for v0.4 — not built, and gated on v0.1 shipping first"
    Nothing on this page is implemented. v0.1 is LangGraph-only. See [Roadmap & status](../project/roadmap.md).

## Why v0.1 is LangGraph-only, and what "later" actually means

v0.1's goals are scoped to LangGraph. Non-LangGraph framework support was originally written into the plan as a loose aspiration ("these can come later") with no version and no design — as of 2026-10-21, it's a real, designed phase: **v0.4**, with the one architecture decision it needs already made in v0.1 itself (see below), not deferred.

## What stays unchanged (the part of "framework-agnostic" that was already true)

`core/` — the `Trajectory`/`Step` data model, all 12 `assert_*` functions, budgets, errors, matchers — needs **zero changes** for CrewAI or AutoGen support. Every field is already generic enough (plain `dict[str, Any]` state, a plain `tuple[str, ...]` namespace) to represent a different framework's execution. This is also the real test of the design: if adding a framework ever turns out to require a `core/` change, that's a bug in this claim, not a reason to quietly patch around it.

## What v0.1 changes proactively, ahead of needing it

The LangGraph-specific recording layer is built as `frameworks/base.py` (a shared `FrameworkRunner` Protocol) + `frameworks/langgraph/` (the only real implementation until v0.4) — not as a flat, LangGraph-only `runtime/` package that would need renaming later. Config and CLI already use generic names (`target:`/`framework:` instead of `graph:`; `--target` instead of `--graph`) even though `langgraph` is the only valid `framework` value today.

## What v0.4 actually has to build

- `frameworks/crewai/` and `frameworks/autogen/` — each a full `runner.py`/`recorder.py`/`interceptors.py`, implementing the same Protocol.
- **Interceptors are per-framework, not shared.** CrewAI and AutoGen don't reliably go through the LangChain classes graphtrace's LangGraph interceptor patches — each framework gets its own real patch target, confirmed against the installed library version at build time, not assumed in advance.
- **An open design question, not yet answered:** AutoGen group chats can rotate speakers dynamically, by design. Whether that fits `assert_path`'s current exact-match/single-`...`-gap model, needs a new assertion, or is a documented scope limit for open-ended group chats specifically, is decided against a real example when this gets built — not guessed now.

The full design — the concept-mapping table (how each framework's execution maps onto `Step`/`Trajectory`), the reasoning for per-framework interceptors, and the weekwise build order — lives in the project's own internal plan doc, not published here.
