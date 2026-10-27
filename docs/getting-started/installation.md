# Installation

**Requirements:** Python 3.10 or newer.

```bash
pip install graphtrace
```

That's it for normal use — `graphtrace` is on PyPI (v0.1.0, published 2026-10-27) with no extras required: `langgraph`, `langchain-core`, `ruamel.yaml`, `pydantic`, `litellm`, `scikit-learn`, `jinja2`, and `typer` all come along as core dependencies. This also registers graphtrace as a pytest plugin automatically (no `-p` flag needed) and installs a `graphtrace` console command — confirmed working this way from the real published wheel in a clean, isolated virtualenv, not just an editable dev install.

```bash
graphtrace --help
```

Should show three real commands: `from-trace`, `calibrate`, `report` — see the [CLI reference](../reference/cli.md).

## Install from source (for contributing to graphtrace itself)

This is what the project itself uses for development — needed if you're contributing to graphtrace, not just using it to test your own LangGraph agent.

```bash
git clone https://github.com/devbyanoop/graphtrace.git
cd graphtrace
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (Git Bash):
source .venv/Scripts/activate

pip install -e ".[dev]"
```

- `python -m venv .venv` creates an isolated Python environment just for this project.
- `-e` installs in **editable** mode — changes under `src/graphtrace/` take effect immediately, without reinstalling.
- `[dev]` pulls in `pytest`, `python-dotenv`, and two real, pinned external apps (`langgraph-swarm`, `deepagents`) kept as permanent compatibility tests — not graphtrace's own dependency, gated with `pytest.importorskip` (see [Case study](../project/case-study.md)). `pyproject.toml`'s own core `dependencies` (not an extra) bring in `langgraph`, `langchain-core`, `ruamel.yaml`, `pydantic`, `litellm`, `scikit-learn`, `jinja2`, and `typer` — everything the full feature set below needs, installed by default.

Verify the install:

```bash
pytest
```

**222 tests should pass**, full regression, in well under a minute. Four of those (`tests/test_judge_live.py`) make a real call to a live Groq model and are skipped cleanly if `GROQ_API_KEY` isn't set (see `.env.example`) — everything else runs offline against fake chat models and hand-built fixtures.

## Extras

| Extra | Pulls in | Status |
| --- | --- | --- |
| `dev` | `pytest`, `python-dotenv`, `langgraph-swarm`, `deepagents` | ✅ Available |
| `docs` | `mkdocs`, `mkdocs-material` | ✅ Available |

`deepagents` itself requires Python ≥3.11 (graphtrace's own floor stays ≥3.10) — it's environment-marked in `pyproject.toml` so `pip install -e ".[dev]"` on Python 3.10 simply skips it, and the real compat tests that need it skip cleanly via `pytest.importorskip` rather than failing the install.

Everything else (`langgraph`, `litellm`, `pydantic`, `ruamel.yaml`, `scikit-learn`, `jinja2`, `typer`) is a core dependency, installed by default — not behind an extra.

See [Roadmap & status](../project/roadmap.md) for the full build history.
