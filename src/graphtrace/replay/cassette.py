"""Load/save cassette YAML files.

Entries are kept as plain dicts throughout (whatever `ruamel.yaml`'s round-trip
loader hands back), never converted into a dataclass and back -- a load-then-
immediately-save round trip stays byte-identical because the *same* loaded
objects (with their original key order/formatting preserved by round-trip mode)
flow straight through to the dumper, rather than being rebuilt from scratch.
This is what actually makes "cassettes diff well in PRs" true for `record_missing`
(Week 2's growing-a-cassette case): untouched entries are untouched bytes, not
just equivalent content.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

_yaml = YAML(typ="rt")
_yaml.preserve_quotes = True
_yaml.default_flow_style = False


@dataclass
class Cassette:
    entries: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    graphtrace: str = "0.1.0"
    recorded_at: str = ""

    def find(self, key: str) -> dict[str, Any] | None:
        """The entry with this exact key, or None if nothing matches."""
        return next((entry for entry in self.entries if entry.get("key") == key), None)

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)


def load_cassette(path: Path | str) -> Cassette:
    """An empty Cassette if `path` doesn't exist yet -- a missing cassette isn't an
    error on its own; whether that's a problem depends on the active mode
    (replay/modes.py), not on this function."""
    path = Path(path)
    if not path.exists():
        return Cassette()
    data = _yaml.load(path.read_text(encoding="utf-8")) or {}
    return Cassette(
        entries=list(data.get("entries") or []),
        version=data.get("version", 1),
        graphtrace=data.get("graphtrace", "0.1.0"),
        recorded_at=data.get("recorded_at", ""),
    )


def to_plain_python(value: Any) -> Any:
    """Convert ruamel.yaml's round-trip scalar/container subclasses (ScalarFloat,
    CommentedMap, CommentedSeq, ...) back into plain dict/list/float/int/str.

    Round-trip mode's whole point is preserving exact on-disk formatting via
    these subclasses -- exactly what this module's own load-then-save round-trip
    test depends on. But a value that's about to be handed back to *real* code
    (replay/engine.py returning a replayed response into LangGraph's own state)
    has no reason to know these types exist, and genuinely breaks on them:
    confirmed by hand that LangGraph's checkpointer msgpack-serializes state for
    persistence and raises `TypeError: Type is not msgpack serializable:
    ScalarFloat` the moment a replayed `49.0` -- loaded as a `ScalarFloat`, a real
    `float` subclass -- lands in state unconverted. Call this only on a value
    about to flow back into real code, never on a value still destined for
    `save_cassette`'s own round-trip path.
    """
    if isinstance(value, dict):
        return {key: to_plain_python(v) for key, v in value.items()}
    if isinstance(value, list):
        return [to_plain_python(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return str(value)
    return value


def save_cassette(cassette: Cassette, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": cassette.version,
        "graphtrace": cassette.graphtrace,
        "recorded_at": cassette.recorded_at or datetime.now(timezone.utc).isoformat(),
        "entries": cassette.entries,
    }
    buf = io.StringIO()
    _yaml.dump(data, buf)
    path.write_text(buf.getvalue(), encoding="utf-8")
