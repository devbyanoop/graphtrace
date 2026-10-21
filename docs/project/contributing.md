# Contributing

graphtrace is in early, active development — see [Roadmap & status](roadmap.md) before picking up anything, since most of the design described elsewhere in this guide isn't implemented yet.

## Development setup

```bash
git clone https://github.com/devbyanoop/graphtrace.git
cd graphtrace
python -m venv .venv
source .venv/bin/activate   # or .venv/Scripts/activate on Windows (Git Bash)
pip install -e ".[dev]"
```

## Running checks locally

```bash
pytest
```

That's the only check wired up today — linting (`ruff`) and strict type-checking (`mypy`) aren't configured yet. If you add either, wire it into this page and into CI (once CI exists — see the roadmap) in the same change, not as a follow-up.

## Design rules that apply to every change

1. **`core/` never imports LangGraph, an LLM SDK, or anything from `runtime/`.** A `Trajectory` is plain dataclasses and plain Python values — see [Core concepts](../guide/trajectory-model.md).
2. **A deliberate scope decision gets documented, not silently shipped.** `assert_path`'s single-gap limit and `assert_retries`' narrower semantics are both written down on [Assertions](../guide/assertions.md) and in the [changelog](changelog.md) — a reviewer (or future you) shouldn't have to reverse-engineer *why* from the diff alone.
3. **No invented numbers.** Cost and token budgets come from real recorded usage (`LLMCall.cost_usd`, `.prompt_tokens`/`.completion_tokens`), never an estimate computed on the fly inside `core`.

## Documentation

This guide (`docs/`) is written incrementally, alongside the code — a feature's documentation is written and verified the same day the feature itself is built and tested, not ahead of time. Pages for unbuilt features carry an explicit "Planned" warning with scope only, no invented usage examples. If you're adding a feature, update its doc page in the same change.

Preview the docs site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
