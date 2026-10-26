"""Tests for codegen/test_writer.py.

Builds a NormalizedTrace by hand (the importer already has its own tests
proving *that* conversion is correct) and checks what generate_test() does
with it: writes a replay-able cassette, and renders a test file whose
run_graph()/assert_path() calls are both syntactically valid Python and
consistent with the real `run_graph` signature (frameworks/langgraph/runner.py).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from graphtrace.codegen.test_writer import _split_target, generate_test
from graphtrace.importers.base import NormalizedSpan, NormalizedTrace
from graphtrace.replay.cassette import load_cassette


def _trace() -> NormalizedTrace:
    return NormalizedTrace(
        spans=[
            NormalizedSpan(node="router", kind="llm", request={"messages": ["refund?"]}, response={"content": "sure, $49.00"}, model="gpt-4o-mini", latency_s=0.6),
            NormalizedSpan(node="lookup_order", kind="tool", request={"order_id": "1234"}, response={"status": "delivered"}, name="lookup_order", latency_s=0.3),
        ],
        initial_input={"question": "refund?"},
    )


# -- _split_target: positive / negative --------------------------------------------------

def test_split_target_parses_module_and_factory():
    assert _split_target("my_app.graph:build_graph") == ("my_app.graph", "build_graph")


def test_split_target_rejects_a_target_with_no_factory():
    with pytest.raises(ValueError):
        _split_target("my_app.graph")


# -- generate_test: the cassette it writes -----------------------------------------------

def test_generate_test_writes_a_cassette_with_positional_any_keys(tmp_path: Path):
    cassette_path = tmp_path / "from_trace.yaml"
    generate_test(_trace(), target="my_app.graph:build_graph", test_name="refund_flow", cassette_path=cassette_path)

    cassette = load_cassette(cassette_path)
    assert len(cassette.entries) == 2
    assert cassette.entries[0]["key"] == "router/llm/any/0"
    assert cassette.entries[1]["key"] == "lookup_order/tool/any/0"


def test_generate_test_normalizes_an_llm_response_already_shaped_correctly(tmp_path: Path):
    cassette_path = tmp_path / "from_trace.yaml"
    generate_test(_trace(), target="my_app.graph:build_graph", test_name="refund_flow", cassette_path=cassette_path)

    cassette = load_cassette(cassette_path)
    llm_response = cassette.entries[0]["response"]
    assert llm_response["content"] == "sure, $49.00"
    assert llm_response["tool_calls"] == []
    assert llm_response["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_generate_test_normalizes_a_bare_string_llm_response(tmp_path: Path):
    """OTel/OpenInference's output.value attribute is a plain string by
    convention (confirmed: importers/otel.py), not a {"content": ...} dict --
    the generated cassette must still be replay-able."""
    trace = NormalizedTrace(spans=[NormalizedSpan(node="router", kind="llm", request="hi", response="just a plain string", model="m")])
    cassette_path = tmp_path / "from_trace.yaml"
    generate_test(trace, target="my_app.graph:build_graph", test_name="t", cassette_path=cassette_path)

    llm_response = load_cassette(cassette_path).entries[0]["response"]
    assert llm_response == {"content": "just a plain string", "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0}}


def test_generate_test_normalizes_a_list_of_chat_messages_llm_response(tmp_path: Path):
    """importers/otel.py's own flattened-message reconstruction produces a
    list of {"role", "content"} dicts (OpenInference's real convention for
    llm.output_messages) -- confirmed this function would otherwise silently
    stringify the whole list instead of extracting the actual reply text."""
    trace = NormalizedTrace(spans=[NormalizedSpan(
        node="router", kind="llm", request="hi",
        response=[{"role": "assistant", "content": "Yes, $49.00 refunded."}],
        model="m",
    )])
    cassette_path = tmp_path / "from_trace.yaml"
    generate_test(trace, target="my_app.graph:build_graph", test_name="t", cassette_path=cassette_path)

    llm_response = load_cassette(cassette_path).entries[0]["response"]
    assert llm_response == {"content": "Yes, $49.00 refunded.", "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0}}


def test_generate_test_preserves_an_errored_span_as_a_cassette_error_entry(tmp_path: Path):
    trace = NormalizedTrace(spans=[NormalizedSpan(node="lookup_order", kind="tool", request={"order_id": "1234"}, error="order service down")])
    cassette_path = tmp_path / "from_trace.yaml"
    generate_test(trace, target="my_app.graph:build_graph", test_name="t", cassette_path=cassette_path)

    entry = load_cassette(cassette_path).entries[0]
    assert entry["error"] == "order service down"
    assert entry["response"] is None


def test_generate_test_assigns_distinct_occurrences_for_repeated_node_visits(tmp_path: Path):
    """A retry loop re-visiting the same node twice must get two distinct
    cassette entries, not one overwriting the other -- same guarantee
    replay/matcher.py gives the live interceptor path."""
    trace = NormalizedTrace(spans=[
        NormalizedSpan(node="router", kind="llm", request="a", response="first try", model="m"),
        NormalizedSpan(node="router", kind="llm", request="a", response="second try", model="m"),
    ])
    cassette_path = tmp_path / "from_trace.yaml"
    generate_test(trace, target="my_app.graph:build_graph", test_name="t", cassette_path=cassette_path)

    entries = load_cassette(cassette_path).entries
    assert entries[0]["key"] == "router/llm/any/0"
    assert entries[1]["key"] == "router/llm/any/1"


# -- generate_test: the rendered pytest file -----------------------------------------------
#
# Targets below are real, already-importable graphtrace symbols, not a
# fabricated "my_app.graph:build_graph" -- `_factory_needs_manual_construction`
# genuinely imports and inspects whatever `target` names, so a nonexistent
# module would always fall into its own (also tested, separately, below)
# import-failure branch instead of exercising the real zero-arg/needs-args
# logic this is actually meant to prove.
#
# graphtrace.replay.matcher:Matcher -- a dataclass with every field defaulted,
# so Matcher() is a genuinely valid zero-arg call.
# graphtrace.judge.client:JudgeClient -- a dataclass whose one field (`model`)
# has no default, so JudgeClient() would raise TypeError; needs manual
# construction, the same real shape this project's own
# `build_refund_graph(chat_model, ...)` has.

_ZERO_ARG_TARGET = "graphtrace.replay.matcher:Matcher"
_NEEDS_ARGS_TARGET = "graphtrace.judge.client:JudgeClient"


def test_generate_test_renders_syntactically_valid_python(tmp_path: Path):
    source = generate_test(_trace(), target=_ZERO_ARG_TARGET, test_name="refund_flow", cassette_path=tmp_path / "c.yaml")
    ast.parse(source)  # raises SyntaxError if this is ever not valid Python


def test_generate_test_calls_a_zero_arg_factory_directly(tmp_path: Path):
    source = generate_test(_trace(), target=_ZERO_ARG_TARGET, test_name="refund_flow", cassette_path=tmp_path / "c.yaml")
    assert "from graphtrace.replay.matcher import Matcher" in source
    assert "def test_refund_flow():" in source
    assert "graph = Matcher()" in source
    assert 'mode="replay"' in source
    assert "{'question': 'refund?'}" in source  # initial_input, rendered via repr
    assert "assert_visited(trajectory, 'router')" in source
    assert "assert_visited(trajectory, 'lookup_order')" in source
    assert "# TODO: assert the correct behaviour" in source


def test_generate_test_flags_a_factory_that_needs_real_arguments(tmp_path: Path):
    """The honest alternative to guessing: a factory shaped like this project's
    own build_refund_graph(chat_model, ...) can't be called with zero
    arguments -- confirmed by actually importing JudgeClient and inspecting its
    signature, not assumed."""
    source = generate_test(_trace(), target=_NEEDS_ARGS_TARGET, test_name="t", cassette_path=tmp_path / "c.yaml")
    assert "graph = JudgeClient(...)  # TODO:" in source
    assert "takes required arguments" in source


def test_generate_test_treats_an_unimportable_target_as_needing_manual_construction(tmp_path: Path):
    """No way to know what a nonexistent module's factory needs -- the cautious
    default (flag it, don't guess zero-arg) applies here too, even though the
    generated file's own top-level import will fail regardless."""
    source = generate_test(_trace(), target="no_such_module_xyz:build_graph", test_name="t", cassette_path=tmp_path / "c.yaml")
    assert "graph = build_graph(...)  # TODO:" in source


def test_generate_test_omits_resume_stub_when_not_interrupted(tmp_path: Path):
    source = generate_test(_trace(), target=_ZERO_ARG_TARGET, test_name="t", cassette_path=tmp_path / "c.yaml")
    assert "resume=" not in source


def test_generate_test_includes_resume_stub_when_interrupted(tmp_path: Path):
    trace = NormalizedTrace(spans=_trace().spans, interrupted=True)
    source = generate_test(trace, target=_ZERO_ARG_TARGET, test_name="t", cassette_path=tmp_path / "c.yaml")
    assert "resume=[...],  # TODO" in source


def test_generate_test_defaults_initial_input_to_empty_dict_when_unknown(tmp_path: Path):
    """OTel/LangSmith importers leave initial_input as None (base.py's own
    documented, honest gap) -- the generated test must still be valid, with an
    empty dict standing in rather than a literal None breaking run_graph()."""
    trace = NormalizedTrace(spans=_trace().spans, initial_input=None)
    source = generate_test(trace, target=_ZERO_ARG_TARGET, test_name="t", cassette_path=tmp_path / "c.yaml")
    ast.parse(source)
    assert "run_graph(\n        graph,\n        {}," in source


def test_generate_test_deduplicates_observed_nodes_preserving_first_occurrence_order(tmp_path: Path):
    """A node visited twice (a retry loop) must get one assert_visited line, not
    one per visit -- assert_visited already means "at least once"."""
    trace = NormalizedTrace(spans=[
        NormalizedSpan(node="router", kind="llm", request="a", response="x", model="m"),
        NormalizedSpan(node="lookup_order", kind="tool", request="b", response="y", name="lookup_order"),
        NormalizedSpan(node="router", kind="llm", request="c", response="z", model="m"),
    ])
    source = generate_test(trace, target=_ZERO_ARG_TARGET, test_name="t", cassette_path=tmp_path / "c.yaml")
    assert source.count("assert_visited(trajectory, 'router')") == 1
    assert source.index("assert_visited(trajectory, 'router')") < source.index("assert_visited(trajectory, 'lookup_order')")
