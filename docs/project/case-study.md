# Case study: building a regression-testing tool for LangGraph agents

## Why I built this

I kept running into the same problem with agent frameworks like LangGraph: a test checks the final
answer, the answer looks fine, and everyone moves on — while the actual path the agent took to get
there quietly changes underneath. A retry that now fires on every call instead of occasionally. A tool
that silently stopped being called. An approval step the agent learned to route around. None of that
shows up if all you're checking is what came out at the end.

I wanted something that treats the path itself — which nodes ran, in what order, what each one
changed, what it called — as a first-class thing you can assert on, the same way a web framework lets
you check a response's status code and not just the rendered page. And I wanted proof it works on a
real agent, not a demo I built specifically to make my own tool look good — so once the core was
working, I pointed it at two real, official open-source LangGraph projects and let them find what my
own demo agent couldn't.

## What I set out to do

- Record a full run — every node, every state change, every tool/LLM call — without touching the
  agent's own code.
- Give it a small, readable set of assertions, not a pile of raw trace data someone has to dig through
  by hand.
- Let a recorded run replay later with zero real LLM calls, so CI never has to pay for, or wait on, a
  real model just to pass.
- Add an LLM-as-judge for the cases a plain assertion can't cover — but never trust it blindly;
  calibrate it against real human-labelled answers first.
- Prove it against real external agents before calling it done, not just my own test fixtures.

## How it's put together

The core trajectory model and assertion API know nothing about LangGraph — they're built against a
plain data structure (nodes, calls, state), so this isn't wired to one framework forever. Recording
happens by patching the real framework calls underneath for the duration of a run, undone automatically
when it ends, so nothing about the agent under test has to change. A run gets written once to a plain
YAML file and replayed from it afterward — no real network calls, no real bill, on every CI run after
the first recording. Cost and latency numbers come from what was actually recorded, never an estimate.

## Problems I ran into

Most of what I learned building this came from a plan turning out to be wrong once I actually ran it:

- **A concurrency assumption that didn't hold.** My first design tracked "which node is currently
  running" with a thread-local variable, read back later inside whatever LLM call that node made. Made
  sense on paper. In practice, LangGraph runs parallel nodes on worker threads, and Python's thread pool
  doesn't carry that variable into the new thread by default. I caught it by printing the real thread
  name inside an actual parallel run before trusting the fix — and it turned out I didn't need the
  workaround at all. LangGraph already tags every call with its own node/task id in its own metadata; I
  just hadn't noticed it was already there.
- **Patching the wrong method.** My first attempt intercepted chat-model calls by patching a method on
  the base class. Every real chat model — OpenAI's, Anthropic's, even LangChain's own test doubles —
  overrides that method with its own version, so patching the base class caught nothing at all. Found by
  checking directly whether the method actually existed where I'd patched it, instead of assuming the
  architecture made sense just because it read cleanly.
- **A YAML library and LangGraph quietly disagreeing about what a number is.** Recordings use YAML's
  round-trip mode so a growing file diffs cleanly in a pull request. That mode wraps a loaded number in
  a special type to preserve its exact formatting — which then broke the instant that value got replayed
  back into a real LangGraph run and serialized by its own checkpointer. Two libraries, each doing
  something reasonable on its own, incompatible together. I only found it by running the full round trip
  for real, not by reading either library's docs.
- **A statistic I'd simply gotten wrong.** I'd written in my own notes that scoring a judge against
  human labels with Cohen's kappa gives 0.0 for "complete disagreement." Not true — a systematic
  reversal actually scores -1.0, a real negative correlation, which is a much stronger signal that
  something's broken than "the judge adds no information at all." Caught by working the formula out by
  hand and checking it against the real library, not by re-reading my own notes more carefully.
- **A feature I'd designed and then just never built.** The data model had room for "entering/exiting a
  sub-agent" as a valid step from week one. Nothing in the recorder ever actually produced it, because
  nothing had tested a node that is itself another graph — a genuinely common pattern in real
  multi-agent apps. Digging into it surfaced a worse problem underneath: the part of the code tracking
  state was shared globally instead of scoped per sub-agent, so two sub-agents running at the same time
  could silently corrupt each other's recorded data. Fixed by scoping it properly — and before trusting
  the new tests, I made sure each one genuinely failed against the old code first.

## What it found once I pointed it at real agents

Once the core was solid against my own demo agent, I ran it against two real, official open-source
LangGraph projects — [`langgraph-swarm-py`](https://github.com/langchain-ai/langgraph-swarm-py) and
[`deepagents`](https://github.com/langchain-ai/deepagents) — across 25 real scenarios. Real code finds
things a hand-built demo can't:

| Project | What turned up |
| --- | --- |
| `langgraph-swarm-py` | Tool calls were being attributed to the wrong node, because a newer LangChain agent abstraction had stopped passing an explicit config through the way my recorder expected. |
| `langgraph-swarm-py` | Sub-agent enter/exit markers were showing up in the wrong order in the recorded trace. |
| `langgraph-swarm-py` | A real agent-to-agent handoff crashed recording outright — the handoff object wasn't safe to write to a YAML file as-is. |
| `deepagents` | A tool's runtime object leaked internal details (object ids, open file handles) straight into recordings, so the exact same call hashed differently every time you ran it. |

One more thing came out of testing `deepagents` that isn't a bug, just a real limit worth being honest
about: if a tool's own code calls another LangGraph agent internally, replaying that call from a
recording means the nested agent's own step-by-step detail is gone — you still get the right final
answer, just not the blow-by-blow of how the nested agent got there. Fixing that properly is a bigger
feature on its own, so for now it's documented rather than quietly ignored.

## What it looks like now

- 222 tests pass, covering the recording/replay/assertion/judge/report pipeline plus 25 scenarios run
  against the two real external projects above.
- Published on PyPI — `pip install graphtrace` registers it as a pytest plugin automatically, no extra
  setup.
- A CLI that turns a real exported production trace into a generated test, and a static HTML report
  with a plain pass/fail verdict for anyone who isn't the engineer who wrote the test.
- CI runs the full suite, plus a strict documentation build, on every push.

## Using it for your project

If you're building or maintaining a LangGraph agent and want its actual behavior under test — not just
its final answers — graphtrace is open source and ready to point at your own graph, with no changes
needed to the agent itself.

I'm available for freelance work in test automation and AI-agent testing, from framework design to
hardening an existing agent for production. You can reach me through my
[GitHub profile](https://github.com/devbyanoop).
