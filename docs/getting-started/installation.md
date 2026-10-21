# Installation

!!! warning "Not yet on PyPI"
    graphtrace hasn't published a release yet. Once it does, installation will be `pip install graphtrace`. This page will be updated the day that's true — until then, the section below is the real, verified way to run it today.

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
- `[dev]` currently only pulls in `pytest`. There's no `langgraph`, `litellm`, or `pydantic` dependency yet because nothing in the code imports them yet — see [Roadmap & status](../project/roadmap.md) for when each one shows up.

Verify the install:

```bash
pytest
```

61 tests should pass — the core data model and assertion API (37, no LangGraph involved, see [Core concepts](../guide/trajectory-model.md)), plus 24 more against real compiled LangGraph runs (the recorder, interceptors, and the `examples/refund_agent` demo agent, see [Recording & replay](../guide/recording-and-replay.md)).

## Planned extras

| Extra | Pulls in | Status |
| --- | --- | --- |
| `dev` | `pytest` | ✅ Available |
| `docs` | `mkdocs`, `mkdocs-material` | ✅ Available |
| (none yet) | `langgraph`, `langchain-core` | Planned — the `runtime/` layer (recorder, interceptors) |
| (none yet) | `ruamel.yaml` | Planned — cassette read/write |
| (none yet) | `litellm` | Planned — judge contract |
| (none yet) | `typer` | Planned — CLI |

See [Roadmap & status](../project/roadmap.md) for the current build state.
