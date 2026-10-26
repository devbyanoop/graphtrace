"""Tests for replay/cassette.py, replay/matcher.py, replay/modes.py in isolation --
before they get wired into the interceptors (test_mode_aware_interceptors.py covers
that integration). Positive, negative, and edge cases, as asked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from graphtrace.replay.cassette import Cassette, load_cassette, save_cassette
from graphtrace.replay.context import ReplayContext
from graphtrace.replay.engine import resolve_call
from graphtrace.replay.matcher import CassetteMismatchError, Matcher, hash_request
from graphtrace.replay.modes import resolve_mode

SAMPLE_CASSETTE = """version: 1
graphtrace: 0.1.0
recorded_at: '2026-10-20T10:12:00Z'
entries:
- key: router/llm/7f3a1b2c3d4e/0
  kind: llm
  node: router
  model: gpt-4o-mini
  request:
    messages:
    - hi
  response:
    content: hello
  latency_s: 0.61
"""


# -- cassette.py: positive -------------------------------------------------------------

def test_load_cassette_parses_entries(tmp_path: Path):
    path = tmp_path / "sample.yaml"
    path.write_text(SAMPLE_CASSETTE, encoding="utf-8")

    cassette = load_cassette(path)
    assert cassette.version == 1
    assert len(cassette.entries) == 1
    assert cassette.entries[0]["node"] == "router"


def test_cassette_find_returns_matching_entry_or_none(tmp_path: Path):
    cassette = load_cassette(tmp_path / "sample.yaml")  # doesn't exist -> empty
    cassette.append({"key": "a/llm/x/0", "kind": "llm"})

    assert cassette.find("a/llm/x/0") is not None
    assert cassette.find("nope") is None


def test_load_then_save_is_byte_identical(tmp_path: Path):
    """The actual point of ruamel.yaml's round-trip mode: proves it's genuinely
    configured, not just imported -- a plain PyYAML dump would not reliably
    reproduce this exact formatting (key order, quoting) on rewrite."""
    original = tmp_path / "original.yaml"
    original.write_text(SAMPLE_CASSETTE, encoding="utf-8")

    cassette = load_cassette(original)
    roundtripped = tmp_path / "roundtripped.yaml"
    save_cassette(cassette, roundtripped)

    assert roundtripped.read_text(encoding="utf-8") == original.read_text(encoding="utf-8")


# -- cassette.py: negative / edge -------------------------------------------------------

def test_load_cassette_missing_file_returns_empty_cassette(tmp_path: Path):
    cassette = load_cassette(tmp_path / "does_not_exist.yaml")
    assert cassette.entries == []
    assert cassette.version == 1


def test_save_cassette_creates_parent_directories(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c.yaml"
    save_cassette(Cassette(), nested)
    assert nested.exists()


# -- matcher.py: positive ---------------------------------------------------------------

def test_hash_request_is_stable_for_identical_input():
    assert hash_request({"order_id": "1234"}) == hash_request({"order_id": "1234"})


def test_hash_request_differs_for_different_input():
    assert hash_request({"order_id": "1234"}) != hash_request({"order_id": "9999"})


def test_hash_request_normalizes_timestamps_and_uuids():
    a = hash_request("called at 2026-10-20T10:12:00Z with id 7f3a1b2c-3d4e-5f60-7182-93a4b5c6d7e8")
    b = hash_request("called at 2026-10-21T09:00:00Z with id aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert a == b  # both normalize to the same "<ts>"/"<uuid>" placeholders


def test_matcher_assigns_distinct_occurrence_for_repeated_identical_calls():
    matcher = Matcher()
    key_1 = matcher.next_key("lookup_order", "tool", {"order_id": "1234"})
    key_2 = matcher.next_key("lookup_order", "tool", {"order_id": "1234"})

    assert key_1 != key_2
    assert key_1.endswith("/0")
    assert key_2.endswith("/1")


def test_matcher_keys_are_independent_per_node_and_kind():
    matcher = Matcher()
    key_tool = matcher.next_key("lookup_order", "tool", {"order_id": "1234"})
    key_llm = matcher.next_key("lookup_order", "llm", {"order_id": "1234"})
    assert key_tool.split("/")[1] == "tool"
    assert key_llm.split("/")[1] == "llm"
    assert key_tool.endswith("/0")
    assert key_llm.endswith("/0")  # different kind -> its own occurrence counter


# -- modes.py: positive / negative / edge ------------------------------------------------

def test_resolve_mode_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("GRAPHTRACE_MODE", "record")
    assert resolve_mode("replay") == "replay"


def test_resolve_mode_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("GRAPHTRACE_MODE", "record_missing")
    assert resolve_mode(None) == "record_missing"


def test_resolve_mode_defaults_to_live_with_nothing_set(monkeypatch):
    monkeypatch.delenv("GRAPHTRACE_MODE", raising=False)
    assert resolve_mode(None) == "live"


def test_resolve_mode_rejects_an_invalid_mode():
    with pytest.raises(ValueError):
        resolve_mode("fast-forward")


def test_cassette_mismatch_error_is_a_runtime_error():
    assert issubclass(CassetteMismatchError, RuntimeError)


# -- engine.py: resolve_call, in isolation from any real LangGraph call -------------------

def test_resolve_call_live_mode_calls_real_and_records_nothing():
    ctx = ReplayContext(mode="live", cassette=Cassette())
    calls = []

    def real_call():
        calls.append(1)
        return "real result"

    result, latency_s = resolve_call(ctx, "node_a", "tool", {"x": 1}, real_call)
    assert result == "real result"
    assert calls == [1]
    assert ctx.new_entries == []  # live mode never writes a cassette


def test_resolve_call_record_mode_calls_real_and_queues_a_new_entry():
    ctx = ReplayContext(mode="record", cassette=Cassette())
    result, latency_s = resolve_call(ctx, "node_a", "tool", {"x": 1}, lambda: "real result", extra={"name": "my_tool"})

    assert result == "real result"
    assert len(ctx.new_entries) == 1
    entry = ctx.new_entries[0]
    assert entry["node"] == "node_a"
    assert entry["kind"] == "tool"
    assert entry["name"] == "my_tool"
    assert entry["response"] == "real result"
    assert entry["request"] == {"x": 1}


def test_resolve_call_record_mode_captures_a_real_exception_then_reraises_original_type():
    ctx = ReplayContext(mode="record", cassette=Cassette())

    def real_call():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        resolve_call(ctx, "node_a", "tool", {"x": 1}, real_call)

    assert ctx.new_entries[0]["error"] == "boom"
    assert ctx.new_entries[0]["response"] is None


def test_resolve_call_replay_mode_returns_recorded_response_without_calling_real():
    cassette = Cassette()
    cassette.append({"key": "node_a/tool/" + hash_request({"x": 1}) + "/0", "kind": "tool", "node": "node_a", "response": "cached result", "latency_s": 0.4})
    ctx = ReplayContext(mode="replay", cassette=cassette)

    def real_call():
        raise AssertionError("must never be called in replay mode")

    result, latency_s = resolve_call(ctx, "node_a", "tool", {"x": 1}, real_call)
    assert result == "cached result"
    assert latency_s == 0.4


def test_resolve_call_replay_mode_raises_cassette_mismatch_when_nothing_matches():
    ctx = ReplayContext(mode="replay", cassette=Cassette())
    with pytest.raises(CassetteMismatchError):
        resolve_call(ctx, "node_a", "tool", {"x": 1}, lambda: "unreachable")


def test_resolve_call_replay_mode_reraises_a_recorded_failure_as_runtime_error():
    cassette = Cassette()
    cassette.append({"key": "node_a/tool/" + hash_request({"x": 1}) + "/0", "kind": "tool", "node": "node_a", "error": "order service down", "response": None})
    ctx = ReplayContext(mode="replay", cassette=cassette)

    with pytest.raises(RuntimeError, match="order service down"):
        resolve_call(ctx, "node_a", "tool", {"x": 1}, lambda: "unreachable")


def test_resolve_call_replay_mode_falls_back_to_positional_any_match():
    """codegen/test_writer.py's cassettes use the literal hash segment "any"
    (see engine.py's own comment) because an imported trace's recorded request
    can never hash-match the real LangChain input at replay time. Confirms the
    fallback actually works, keyed only on node/kind/occurrence."""
    cassette = Cassette()
    cassette.append({"key": "node_a/tool/any/0", "kind": "tool", "node": "node_a", "response": "from imported trace", "latency_s": 0.2})
    ctx = ReplayContext(mode="replay", cassette=cassette)

    result, latency_s = resolve_call(ctx, "node_a", "tool", {"this request": "does not match anything recorded"}, lambda: (_ for _ in ()).throw(AssertionError("must never be called")))
    assert result == "from imported trace"


def test_resolve_call_replay_mode_prefers_exact_hash_match_over_any_fallback():
    """A real hash match must win even when an "any" entry also exists for the
    same node/kind/occurrence -- the fallback is only ever a last resort, never
    allowed to shadow a legitimate recorded entry."""
    cassette = Cassette()
    cassette.append({"key": "node_a/tool/any/0", "kind": "tool", "node": "node_a", "response": "wrong one"})
    cassette.append({"key": "node_a/tool/" + hash_request({"x": 1}) + "/0", "kind": "tool", "node": "node_a", "response": "correct one"})
    ctx = ReplayContext(mode="replay", cassette=cassette)

    result, latency_s = resolve_call(ctx, "node_a", "tool", {"x": 1}, lambda: None)
    assert result == "correct one"


def test_resolve_call_record_missing_mode_replays_an_existing_match_but_records_a_new_one():
    cassette = Cassette()
    cassette.append({"key": "node_a/tool/" + hash_request({"x": 1}) + "/0", "kind": "tool", "node": "node_a", "response": "cached", "latency_s": 0.1})
    ctx = ReplayContext(mode="record_missing", cassette=cassette)

    # First call matches the existing entry -- no real call, nothing new queued.
    result_1, _ = resolve_call(ctx, "node_a", "tool", {"x": 1}, lambda: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert result_1 == "cached"
    assert ctx.new_entries == []

    # A different request has no match -> real call happens, and it's queued.
    result_2, _ = resolve_call(ctx, "node_a", "tool", {"x": 2}, lambda: "fresh result")
    assert result_2 == "fresh result"
    assert len(ctx.new_entries) == 1


def test_resolve_call_repeated_identical_requests_match_in_recorded_order():
    """The exact scenario occurrence indexing exists for: a retry loop calling
    the same tool with the same args twice must replay its own two recorded
    responses in order, not both matching the first."""
    cassette = Cassette()
    base_key = "node_a/tool/" + hash_request({"x": 1})
    cassette.append({"key": base_key + "/0", "kind": "tool", "node": "node_a", "error": "first attempt failed", "response": None})
    cassette.append({"key": base_key + "/1", "kind": "tool", "node": "node_a", "response": "second attempt succeeded", "latency_s": 0.1})
    ctx = ReplayContext(mode="replay", cassette=cassette)

    with pytest.raises(RuntimeError, match="first attempt failed"):
        resolve_call(ctx, "node_a", "tool", {"x": 1}, lambda: "unreachable")

    result, _ = resolve_call(ctx, "node_a", "tool", {"x": 1}, lambda: "unreachable")
    assert result == "second attempt succeeded"
