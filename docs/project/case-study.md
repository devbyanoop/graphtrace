# Case study: building graphtrace

*A short account of what it took to make a regression-testing tool for LangGraph agents actually trustworthy — written for anyone deciding whether to work with the person who built it.*

## The problem it solves

A LangGraph agent with branches, retries, and tool calls can look fine on a lucky run while silently changing *how* it got there — a node path that changed, a retry now firing on every call, a tool call that quietly stopped happening, an approval gate the agent learned to route around. Checking only the final answer text can't catch any of that, because the final answer can still look right while the path that produced it is wrong, expensive, or unsafe. graphtrace records the full trajectory of a run — every node, every state change, every tool/LLM call — and makes that path a first-class, assertable thing, with record/replay cassettes so a CI run never has to make a real, billed LLM call to pass.

It's a complete v0.1: the trajectory model and a 12-function assertion API, real LangGraph recording (including subgraphs), record/replay, an LLM-as-judge contract with calibration, a CLI that turns a production trace export into a generated test, a static HTML report, and CI — 204 tests pass today (four real bugs found testing against two real external apps, plus an executive summary layer for the report -- see the roadmap).

## What this case study is actually about

Not the feature list above — that's in the [README](https://github.com/devbyanoop/graphtrace). This is about the five real bugs that building it surfaced, each one found by deliberately checking a claim against real behavior instead of trusting a design document's description of it, and what each one says about how this kind of work should be done.

### 1. The attribution mechanism in the original design didn't survive real concurrency

The plan called for a `contextvars.ContextVar` set to "the current node" while a graph streamed, read back inside whatever LLM/tool call that node made. It looked right. It wasn't: LangGraph runs parallel nodes on a `ThreadPoolExecutor` worker thread, and Python's `ThreadPoolExecutor.submit()` doesn't propagate the calling thread's contextvars into the worker by default. Confirmed with a `threading.current_thread()` probe inside a real fan-out node before trusting the fix, not after. The real answer was better anyway: LangGraph already tags every call with a `"{node}:{task_id}"` string in its own metadata, which travels with the call through whatever thread actually executes it — no side channel needed.

### 2. The chat-model patch target didn't intercept anything

The original design patched `BaseChatModel._generate`. Every real chat model — OpenAI's, Anthropic's, LangChain's own test doubles — defines its own `_generate`, overriding the base class's. Patching the base class patches a method nothing ever calls. Found by checking `'_generate' in GenericFakeChatModel.__dict__` directly rather than assuming the architecture document's description was current.

### 3. A YAML round-trip feature broke a downstream library in a way neither library's own docs would have predicted

`ruamel.yaml`'s round-trip mode (needed so a growing cassette file diffs cleanly in a PR) wraps a loaded float in a `ScalarFloat` subclass to preserve its exact on-disk formatting. That same value, replayed back into a real LangGraph run, flows into the framework's own state and gets serialized by its checkpointer — which doesn't know what a `ScalarFloat` is, and raises. Two correct libraries, one real incompatibility, found only by running the actual round trip end to end.

### 4. A statistics claim in the project's own build plan was simply wrong

The plan asserted that scoring a human-vs-judge calibration set with Cohen's kappa gives 0.0 for "full disagreement." For balanced labels, a *systematic* inversion actually scores **-1.0** — a real, measurable anti-correlation, worse than chance, not merely "no better than it." Confirmed by deriving the formula by hand and checking it against the real, installed `scikit-learn`, not by re-reading the plan more carefully. The distinction isn't academic: a kappa near -1 is a sharper signal ("the judge may be reading this rubric backwards") than a kappa near 0 ("the judge adds no information"), and conflating them would have shipped a worse diagnostic to anyone using this for real.

### 5. A feature the type system had promised since day one was never actually built — and nothing caught it for weeks

`Step.kind` declared `"subgraph_enter"`/`"subgraph_exit"` as valid values from the very first week. Nothing in the recorder ever produced them, because nothing ever tested a node whose value is itself a compiled graph — a genuinely common pattern in real multi-agent LangGraph apps. Investigating it by hand surfaced a second, more serious problem underneath: the recorder's internal state tracking was global, not scoped per graph-namespace, so a subgraph running *concurrently* with another node could silently corrupt that sibling's recorded state with the subgraph's own unrelated data. Reproduced deliberately, fixed by scoping the bookkeeping correctly, and confirmed with two tests that were each checked to genuinely fail against the pre-fix code — not just pass against the fix.

## Why this is the point, not a footnote

Every one of these was in code that already had passing tests. None of them would have been caught by writing more tests against the same assumptions, or by reading the design documentation more carefully — the documentation was wrong, or silent, in exactly the way that looks fine until someone deliberately runs the real thing and checks. That's the actual discipline behind this project: pin down what a library, a statistic, or a framework's own promise genuinely does, by running it, before building anything on top of the belief that it does what it's supposed to. The same discipline shipped the fix for all five, with the real finding documented next to the fix rather than quietly absorbed into the code.

## Where this leaves graphtrace

[Roadmap & status](roadmap.md) has the complete, dated build history, including a packaging check (a real wheel, installed into a genuinely isolated clean virtual environment — the first attempt at that check used a shell activation that silently fell through to the system interpreter instead, caught before it invalidated the result) and real verification against a live Groq model for the judge contract. v0.2 (a security/red-team test pack, observability export, cut-point replay, statistical live-mode verification) and v0.4 (CrewAI and AutoGen support) are designed, scoped, and not yet started.

If this is the kind of rigor you want applied to testing, auditing, or hardening your own LangGraph or agentic system, [get in touch](https://github.com/devbyanoop).
