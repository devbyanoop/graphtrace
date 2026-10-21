"""The four replay modes, and how the active one is resolved.

| Mode            | LLM/tool calls? | Cassette               |
|-----------------|------------------|-------------------------|
| `replay`        | No               | Must exist; missing entry raises |
| `record`        | Yes              | Overwritten            |
| `record_missing`| Only unmatched   | Appended               |
| `live`          | Yes              | Ignored                |
"""

from __future__ import annotations

import os
from typing import Literal

Mode = Literal["record", "replay", "record_missing", "live"]

VALID_MODES: tuple[Mode, ...] = ("record", "replay", "record_missing", "live")


def resolve_mode(explicit: str | None = None) -> Mode:
    """An explicit argument wins; otherwise `GRAPHTRACE_MODE`; otherwise `"live"`
    -- Week 1's existing, unchanged default until a caller actually opts into
    something else.
    """
    mode = explicit or os.environ.get("GRAPHTRACE_MODE") or "live"
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES!r}, got {mode!r}")
    return mode  # type: ignore[return-value]
