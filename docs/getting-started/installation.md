# Installation

!!! warning "Not yet on PyPI"
    graphtrace hasn't published a release yet. Once it does, installation will be `pip install graphtrace`. This page will be updated the day that's true — until then, the section below is the real, verified way to run it today (including from a real built wheel, not just an editable install — see [Roadmap & status](../project/roadmap.md)'s packaging note).

## Install from source (current, verified)

This is what the project itself uses for development, and it works right now against the current state of the repository.

**Requirements:** Python 3.10 or newer.

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
- `[dev]` pulls in `pytest` and `python-dotenv`. `pyproject.toml`'s own core `dependencies` (not an extra) bring in `langgraph`, `langchain-core`, `ruamel.yaml`, `pydantic`, `litellm`, `scikit-learn`, `jinja2`, and `typer` — everything the full feature set below needs, installed by default.

This also installs a `graphtrace` console command and registers graphtrace as a pytest plugin automatically (no `-p` flag needed) — confirmed working the same way from a real built wheel in a clean, isolated virtualenv, not just this editable-install dev loop.

Verify the install:

```bash
pytest
```

**199 tests should pass**, full regression, in well under a minute. Four of those (`tests/test_judge_live.py`) make a real call to a live Groq model and are skipped cleanly if `GROQ_API_KEY` isn't set (see `.env.example`) — everything else runs offline against fake chat models and hand-built fixtures.

```bash
graphtrace --help
```

Should show three real commands: `from-trace`, `calibrate`, `report` — see the [CLI reference](../reference/cli.md).

## Extras

| Extra | Pulls in | Status |
| --- | --- | --- |
| `dev` | `pytest`, `python-dotenv` | ✅ Available |
| `docs` | `mkdocs`, `mkdocs-material` | ✅ Available |

Everything else (`langgraph`, `litellm`, `pydantic`, `ruamel.yaml`, `scikit-learn`, `jinja2`, `typer`) is a core dependency, installed by default — not behind an extra.

See [Roadmap & status](../project/roadmap.md) for the full build history.
