"""OpenTelemetry / OpenInference trace importer.

Reads standard OTLP JSON (`resourceSpans[].scopeSpans[].spans[]`) tagged with
OpenInference semantic conventions (`openinference.span.kind` = `LLM`/`TOOL`/...,
`llm.model_name`, `input.value`/`output.value`, `tool.name`) -- these are the
parts of this importer verified against the real, current source
(`openinference-semantic-conventions`'s own `SpanAttributes`/`MessageAttributes`
constants, fetched and quoted directly, not assumed from memory). What's *not*
independently verified here: whether a real exporter you're using actually sets
these the way this importer expects for every span kind -- check against your
own exporter's output if a real trace doesn't parse cleanly.

**A real bug found and fixed while confirming this against the actual source**:
there is no single `"llm.input_messages"` attribute to read -- confirmed via
`openinference-semantic-conventions`'s own `trace/__init__.py`, chat messages
are flattened per-index (`llm.input_messages.0.message.content`,
`llm.input_messages.0.message.role`, `.1....`, ...). An earlier version of this
importer read the un-flattened key directly, which would silently never match
anything in a real export -- always falling through to `input.value` instead,
right up until a trace existed where only the flattened messages were set and
`input.value` wasn't. `_flattened_messages()` below reconstructs the real
convention instead.
"""

from __future__ import annotations

from typing import Any

from graphtrace.importers.base import NormalizedSpan, NormalizedTrace


def _attr_value(value: dict[str, Any] | None) -> Any:
    """OTLP JSON wraps every attribute value in a typed envelope
    ({"stringValue": ...} | {"intValue": ...} | {"doubleValue": ...} |
    {"boolValue": ...}) -- unwrap whichever key is actually present."""
    if not value:
        return None
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    return None


def _attributes_dict(span: dict[str, Any]) -> dict[str, Any]:
    return {attr["key"]: _attr_value(attr.get("value")) for attr in span.get("attributes", [])}


def _flattened_messages(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]] | None:
    """Reconstruct OpenInference's per-index flattened message convention
    (`{prefix}.{i}.message.role`/`.content`) -- confirmed via the real
    `openinference-semantic-conventions` source, not the single un-flattened
    key an earlier version of this importer wrongly looked for. Returns None
    (not an empty list) when nothing is found, so a caller can tell "no
    flattened messages at all" apart from "a message with empty fields" and
    fall back to `input.value`/`output.value` instead.
    """
    messages: list[dict[str, Any]] = []
    index = 0
    while f"{prefix}.{index}.message.role" in attrs or f"{prefix}.{index}.message.content" in attrs:
        messages.append({
            "role": attrs.get(f"{prefix}.{index}.message.role"),
            "content": attrs.get(f"{prefix}.{index}.message.content"),
        })
        index += 1
    return messages or None


class OTelImporter:
    name = "otel"

    def can_read(self, data: dict[str, Any]) -> bool:
        return "resourceSpans" in data

    def to_normalized(self, data: dict[str, Any]) -> NormalizedTrace:
        spans: list[NormalizedSpan] = []
        for resource_span in data.get("resourceSpans", []):
            for scope_span in resource_span.get("scopeSpans", []):
                for raw_span in scope_span.get("spans", []):
                    normalized = self._normalize_span(raw_span)
                    if normalized is not None:
                        spans.append(normalized)
        return NormalizedTrace(spans=spans)

    def _normalize_span(self, raw_span: dict[str, Any]) -> NormalizedSpan | None:
        attrs = _attributes_dict(raw_span)
        span_kind = attrs.get("openinference.span.kind")
        if span_kind == "LLM":
            kind = "llm"
        elif span_kind == "TOOL":
            kind = "tool"
        else:
            # CHAIN/AGENT/RETRIEVER/etc. aren't LLM or tool calls on their own --
            # skipped, not guessed at.
            return None

        status = raw_span.get("status") or {}
        is_error = status.get("code") not in (None, "STATUS_CODE_OK", 0, 1)
        start_ns = int(raw_span.get("startTimeUnixNano", 0) or 0)
        end_ns = int(raw_span.get("endTimeUnixNano", 0) or 0)

        return NormalizedSpan(
            node=raw_span.get("name", "?"),
            kind=kind,
            request=attrs.get("input.value") or _flattened_messages(attrs, "llm.input_messages"),
            response=None if is_error else (attrs.get("output.value") or _flattened_messages(attrs, "llm.output_messages")),
            error=status.get("message") if is_error else None,
            latency_s=(end_ns - start_ns) / 1_000_000_000 if end_ns > start_ns else 0.0,
            model=attrs.get("llm.model_name") if kind == "llm" else None,
            name=attrs.get("tool.name") or raw_span.get("name") if kind == "tool" else None,
        )
