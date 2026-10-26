"""OpenTelemetry / OpenInference trace importer.

Reads standard OTLP JSON (`resourceSpans[].scopeSpans[].spans[]`) tagged with
OpenInference semantic conventions (`openinference.span.kind` = `LLM`/`TOOL`/...,
`llm.model_name`, `llm.input_messages`/`llm.output_messages`, `input.value`/
`output.value`) -- these are the two parts of this importer verified against
real, current documentation (the OTLP JSON wire format itself, and the
OpenInference spec's required `openinference.span.kind` attribute and its
`llm.*` attribute names). What's *not* independently verified here: whether a
real exporter you're using actually sets `input.value`/`output.value` the way
this importer expects for a TOOL span, versus some other attribute name --
check against your own exporter's output if a real trace doesn't parse cleanly.
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
            request=attrs.get("input.value") or attrs.get("llm.input_messages"),
            response=None if is_error else (attrs.get("output.value") or attrs.get("llm.output_messages")),
            error=status.get("message") if is_error else None,
            latency_s=(end_ns - start_ns) / 1_000_000_000 if end_ns > start_ns else 0.0,
            model=attrs.get("llm.model_name") if kind == "llm" else None,
            name=attrs.get("tool.name") or raw_span.get("name") if kind == "tool" else None,
        )
