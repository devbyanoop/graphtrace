# Contributing

graphtrace's v0.1 scope is complete — see [Roadmap & status](roadmap.md) for exactly what's built, including every real bug found and fixed along the way. v0.2 (security test pack, observability export, cut-point replay, statistical live-mode verification) and v0.4 (CrewAI/AutoGen support) are designed but not started; check the roadmap before picking either up.

## Development setup

```bash
git clone https://github.com/devbyanoop/graphtrace.git
cd graphtrace
python -m venv .venv
source .venv/bin/activate   # or .venv/Scripts/activate on Windows (Git Bash)
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in a real provider key if you want `tests/test_judge_live.py`'s 4 live tests to run too (they skip cleanly without one).

## Running checks locally

```bash
pytest                      # 201 tests, full regression
mkdocs build --strict       # this docs site, strict mode (pip install -e ".[docs]" first)
```

Both are what `.github/workflows/graphtrace.yml` runs on every push/PR. Linting (`ruff`) and strict type-checking (`mypy`) aren't configured yet — `src/graphtrace/py.typed` (PEP 561) is already in place for an external type-checker to use against this package, but nothing in this repo enforces it internally yet. If you add either, wire it into this page and into CI in the same change, not as a follow-up.

## Design rules that apply to every change

1. **`core/` never imports LangGraph, an LLM SDK, or anything from `frameworks/`/`replay/`/`judge/`/`importers/`.** A `Trajectory` is plain dataclasses and plain Python values — see [Core concepts](../guide/trajectory-model.md).
2. **`replay/engine.py` never imports LangChain or LangGraph, and never sees a framework object** — only plain dicts/strings/numbers in and out. All framework-specific serialization lives in `frameworks/<name>/interceptors.py`, on either side of the call into `resolve_call()`. This is what keeps record/replay reusable for v0.4's CrewAI/AutoGen interceptors without rewriting it.
3. **A deliberate scope decision gets documented, not silently shipped.** `assert_path`'s single-gap limit, the trace-to-test generated assertion being `assert_visited` rather than `assert_path`, and the judge's `scale: "1-5"` validating-but-not-scoring are all written down on their own guide page and in the [changelog](changelog.md) — a reviewer (or future you) shouldn't have to reverse-engineer *why* from the diff alone.
4. **No invented numbers.** Cost and token budgets come from real recorded usage (`LLMCall.cost_usd`, `.prompt_tokens`/`.completion_tokens`), never an estimate computed on the fly. Judge agreement comes from a real calibration run, never a guessed kappa.
5. **Verify a fast-moving library's real behavior by running real code before trusting a design doc's description of it** — this project's single most repeated lesson (see the roadmap's build history): contextvars attribution, the `_generate` patch target, `ruamel.yaml`'s `ScalarFloat`, Cohen's kappa for a systematic label flip, and subgraph state bookkeeping were all confirmed *wrong* this way, not by re-reading a plan that looked reasonable on paper.

## Documentation

This guide (`docs/`) is written incrementally, alongside the code — a feature's documentation is written and verified the same day the feature itself is built and tested, not ahead of time. A page for a genuinely unbuilt feature (v0.2/v0.4 scope) carries an explicit "Planned" warning with scope only, no invented usage examples. If you're adding a feature, update its doc page in the same change.

Preview the docs site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
