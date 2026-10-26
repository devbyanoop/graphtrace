"""Tests for importers/{base,langfuse,otel,langsmith}.py.

Fixtures below are hand-built, representative of each platform's documented
export conventions -- not pulled from a real, live export (none of the three
importer modules have been verified against one; see each module's own
docstring for exactly what is and isn't confirmed). The point of these tests
is proving each importer normalizes *its own* documented shape correctly and
consistently with the others, per BUILD_PLAN.md's own verify-by.
"""

from __future__ import annotations

from graphtrace.importers.base import NormalizedTrace
from graphtrace.importers.langfuse import LangfuseImporter
from graphtrace.importers.langsmith import LangSmithImporter
from graphtrace.importers.otel import OTelImporter


# -- can_read: each importer recognizes its own shape, and only its own ------------------

def test_can_read_distinguishes_all_three_formats():
    langfuse_data = {"id": "t1", "observations": []}
    otel_data = {"resourceSpans": []}
    langsmith_data = {"runs": []}

    langfuse, otel, langsmith = LangfuseImporter(), OTelImporter(), LangSmithImporter()

    assert langfuse.can_read(langfuse_data) and not otel.can_read(langfuse_data) and not langsmith.can_read(langfuse_data)
    assert otel.can_read(otel_data) and not langfuse.can_read(otel_data) and not langsmith.can_read(otel_data)
    assert langsmith.can_read(langsmith_data) and not langfuse.can_read(langsmith_data) and not otel.can_read(langsmith_data)


def test_can_read_rejects_an_unrelated_dict():
    assert not LangfuseImporter().can_read({"something": "else"})
    assert not OTelImporter().can_read({"something": "else"})
    assert not LangSmithImporter().can_read({"something": "else"})


# -- Langfuse: positive / negative / edge ------------------------------------------------

def test_langfuse_importer_normalizes_generation_and_span():
    data = {
        "id": "trace-1",
        "observations": [
            {
                "type": "GENERATION",
                "name": "router",
                "input": {"messages": ["hi"]},
                "output": {"content": "ok"},
                "model": "gpt-4o-mini",
                "startTime": "2026-10-20T10:12:00",
                "endTime": "2026-10-20T10:12:00.610000",
            },
            {
                "type": "SPAN",
                "name": "lookup_order",
                "input": {"order_id": "1234"},
                "output": {"status": "delivered"},
                "startTime": "2026-10-20T10:12:01",
                "endTime": "2026-10-20T10:12:01.300000",
            },
        ],
    }
    normalized = LangfuseImporter().to_normalized(data)
    assert isinstance(normalized, NormalizedTrace)
    assert len(normalized.spans) == 2
    llm_span, tool_span = normalized.spans
    assert llm_span.kind == "llm" and llm_span.model == "gpt-4o-mini"
    assert tool_span.kind == "tool" and tool_span.name == "lookup_order"
    assert round(llm_span.latency_s, 2) == 0.61


def test_langfuse_importer_captures_an_errored_generation():
    data = {
        "id": "trace-1",
        "observations": [
            {"type": "GENERATION", "name": "router", "input": {}, "level": "ERROR", "statusMessage": "rate limited"},
        ],
    }
    normalized = LangfuseImporter().to_normalized(data)
    assert normalized.spans[0].error == "rate limited"
    assert normalized.spans[0].response is None


def test_langfuse_importer_skips_event_observations():
    data = {"id": "t", "observations": [{"type": "EVENT", "name": "something"}]}
    assert LangfuseImporter().to_normalized(data).spans == []


# -- OTel/OpenInference: positive / negative / edge -----------------------------------------

def _otel_span(name: str, kind: str, extra_attrs: dict, start_ns: int = 0, end_ns: int = 610_000_000, status_code="STATUS_CODE_OK", status_message=None):
    attrs = [{"key": "openinference.span.kind", "value": {"stringValue": kind}}]
    for key, value in extra_attrs.items():
        attrs.append({"key": key, "value": {"stringValue": value}})
    return {
        "name": name,
        "attributes": attrs,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "status": {"code": status_code, "message": status_message},
    }


def test_otel_importer_normalizes_llm_and_tool_spans():
    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            _otel_span("router", "LLM", {"llm.model_name": "gpt-4o-mini", "input.value": "hi", "output.value": "ok"}),
                            _otel_span("lookup_order", "TOOL", {"tool.name": "get_order", "input.value": "1234", "output.value": "delivered"}),
                        ]
                    }
                ]
            }
        ]
    }
    normalized = OTelImporter().to_normalized(data)
    assert len(normalized.spans) == 2
    llm_span, tool_span = normalized.spans
    assert llm_span.kind == "llm" and llm_span.model == "gpt-4o-mini"
    assert tool_span.kind == "tool" and tool_span.name == "get_order"
    assert round(llm_span.latency_s, 2) == 0.61


def test_otel_importer_reconstructs_flattened_chat_messages_when_input_value_is_absent():
    """Real bug, found and fixed against the actual openinference-semantic-
    conventions source: there is no single "llm.input_messages" attribute to
    read -- messages are flattened per-index as "llm.input_messages.{i}.
    message.role"/".content". An earlier version of this importer read the
    un-flattened key directly, which never matched anything real, silently
    falling through to input.value every time -- invisible until a trace
    exists (like this one) where only the flattened messages are set."""
    data = {
        "resourceSpans": [{"scopeSpans": [{"spans": [
            _otel_span("router", "LLM", {
                "llm.model_name": "gpt-4o-mini",
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.0.message.content": "You are a refund agent.",
                "llm.input_messages.1.message.role": "user",
                "llm.input_messages.1.message.content": "Can I get a refund?",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "Yes, $49.00 refunded.",
            }),
        ]}]}]
    }
    normalized = OTelImporter().to_normalized(data)
    span = normalized.spans[0]
    assert span.request == [
        {"role": "system", "content": "You are a refund agent."},
        {"role": "user", "content": "Can I get a refund?"},
    ]
    assert span.response == [{"role": "assistant", "content": "Yes, $49.00 refunded."}]


def test_otel_importer_prefers_input_value_over_flattened_messages_when_both_are_set():
    data = {
        "resourceSpans": [{"scopeSpans": [{"spans": [
            _otel_span("router", "LLM", {
                "input.value": "the plain summary",
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "ignored",
            }),
        ]}]}]
    }
    normalized = OTelImporter().to_normalized(data)
    assert normalized.spans[0].request == "the plain summary"


def test_otel_importer_captures_an_error_status():
    data = {
        "resourceSpans": [
            {"scopeSpans": [{"spans": [_otel_span("lookup_order", "TOOL", {}, status_code="STATUS_CODE_ERROR", status_message="timeout")]}]}
        ]
    }
    normalized = OTelImporter().to_normalized(data)
    assert normalized.spans[0].error == "timeout"
    assert normalized.spans[0].response is None


def test_otel_importer_skips_non_llm_non_tool_spans():
    data = {"resourceSpans": [{"scopeSpans": [{"spans": [_otel_span("some_chain", "CHAIN", {})]}]}]}
    assert OTelImporter().to_normalized(data).spans == []


# -- LangSmith: positive / negative / edge --------------------------------------------------

def test_langsmith_importer_normalizes_llm_and_tool_runs():
    data = {
        "runs": [
            {
                "name": "router",
                "run_type": "llm",
                "inputs": {"messages": ["hi"]},
                "outputs": {"content": "ok"},
                "start_time": "2026-10-20T10:12:00",
                "end_time": "2026-10-20T10:12:00.610000",
                "extra": {"model": "gpt-4o-mini"},
            },
            {
                "name": "lookup_order",
                "run_type": "tool",
                "inputs": {"order_id": "1234"},
                "outputs": {"status": "delivered"},
                "start_time": "2026-10-20T10:12:01",
                "end_time": "2026-10-20T10:12:01.300000",
            },
        ]
    }
    normalized = LangSmithImporter().to_normalized(data)
    assert len(normalized.spans) == 2
    llm_span, tool_span = normalized.spans
    assert llm_span.kind == "llm" and llm_span.model == "gpt-4o-mini"
    assert tool_span.kind == "tool" and tool_span.name == "lookup_order"


def test_langsmith_importer_captures_a_run_error():
    data = {"runs": [{"name": "lookup_order", "run_type": "tool", "inputs": {}, "error": "ConnectionError"}]}
    normalized = LangSmithImporter().to_normalized(data)
    assert normalized.spans[0].error == "ConnectionError"
    assert normalized.spans[0].response is None


def test_langsmith_importer_skips_chain_runs():
    data = {"runs": [{"name": "agent", "run_type": "chain", "inputs": {}, "outputs": {}}]}
    assert LangSmithImporter().to_normalized(data).spans == []


# -- cross-importer consistency: the whole point of a shared NormalizedTrace shape ----------

def test_all_three_importers_produce_the_same_normalized_shape_for_equivalent_data():
    """Not byte-identical (field names differ per source), but every importer's
    output is a NormalizedTrace of NormalizedSpans with the same attribute set
    -- this is what makes codegen/test_writer.py (next) source-agnostic."""
    langfuse_spans = LangfuseImporter().to_normalized({"id": "t", "observations": [
        {"type": "GENERATION", "name": "router", "input": "hi", "output": "ok", "model": "m"}
    ]}).spans
    otel_spans = OTelImporter().to_normalized({"resourceSpans": [{"scopeSpans": [{"spans": [
        _otel_span("router", "LLM", {"llm.model_name": "m", "input.value": "hi", "output.value": "ok"})
    ]}]}]}).spans
    langsmith_spans = LangSmithImporter().to_normalized({"runs": [
        {"name": "router", "run_type": "llm", "inputs": "hi", "outputs": "ok", "extra": {"model": "m"}}
    ]}).spans

    for spans in (langfuse_spans, otel_spans, langsmith_spans):
        assert len(spans) == 1
        assert spans[0].node == "router"
        assert spans[0].kind == "llm"
        assert spans[0].model == "m"
