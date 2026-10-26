"""The TraceImporter protocol and the NormalizedTrace shape every importer
produces, regardless of source format.

NormalizedSpan mirrors a cassette entry's own fields (request/response/error/
latency_s/model/name) almost exactly on purpose -- turning a NormalizedTrace into
a cassette (codegen/test_writer.py) is then close to a direct copy, not a second
translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class NormalizedSpan:
    node: str
    kind: Literal["llm", "tool"]
    request: Any
    response: Any = None
    error: str | None = None
    latency_s: float = 0.0
    model: str | None = None  # llm spans
    name: str | None = None  # tool spans


@dataclass
class NormalizedTrace:
    spans: list[NormalizedSpan] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    # The graph's original input, if the source trace exposes a trace-level
    # input field (Langfuse's own top-level `input`, confirmed by its
    # documented trace shape). OTel/LangSmith have no equally obvious
    # trace-level marker in the generic case -- left `None` there rather than
    # guessed from the first span's own request, which isn't the same thing.
    initial_input: Any = None
    # Whether the traced run paused on a human-in-the-loop interrupt. Always
    # False today -- no importer below actually detects this yet. None of
    # Langfuse/OTel/LangSmith have a universal "this run paused for human input"
    # concept in their own export formats (it's LangGraph application logic, not
    # something these platforms are positioned to observe directly), so this is
    # an honest, acknowledged gap, not a silently wrong guess. The field, and
    # codegen's resume-stub logic that reads it, both exist so a *future*,
    # smarter per-source heuristic has somewhere real to plug in.
    interrupted: bool = False


class TraceImporter(Protocol):
    name: str

    def can_read(self, data: dict[str, Any]) -> bool: ...
    def to_normalized(self, data: dict[str, Any]) -> NormalizedTrace: ...
