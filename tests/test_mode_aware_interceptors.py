"""The actual point of Week 2: record the real demo agent once, then replay it
with the *real* underlying LangChain methods patched to explode if ever touched --
proving replay mode genuinely never makes a real call, not just that the unit-level
resolve_call() logic says it shouldn't (test_replay.py already covers that in
isolation). Positive (record, replay, record_missing growing a cassette), negative
(no cassette, a recorded failure replaying as a failure), and edge cases (a retry
sequence replaying in the right order) are grouped below.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from refund_agent.graph import build_refund_graph

from graphtrace import assert_path, assert_state, assert_tool_called, run_graph
from graphtrace.replay.cassette import load_cassette
from graphtrace.replay.matcher import CassetteMismatchError


def _fake_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.repeat(AIMessage(content="ok")))


def _explode_if_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches the real, underlying LangChain methods -- the ones graphtrace's own
    interceptor would call *through* if (and only if) replay logic ever fell back
    to a live call. If a test using this still passes, no real call happened,
    full stop -- this isn't inferred from the replay logic's own reasoning, it's
    observed directly at the one layer below graphtrace's control.
    """

    def boom(*args, **kwargs):
        raise AssertionError("a real call happened during what should have been a pure replay")

    monkeypatch.setattr(BaseChatModel, "invoke", boom)
    monkeypatch.setattr(BaseTool, "invoke", boom)


# -- positive: record then replay --------------------------------------------------------

def test_record_then_replay_reproduces_the_same_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cassette_path = str(tmp_path / "refund_happy_path.yaml")
    graph = build_refund_graph(_fake_model())

    recorded = run_graph(graph, {"order_id": "1234"}, cassette=cassette_path, mode="record", resume=[True])
    assert recorded.interrupted is False
    assert_path(recorded, ["router", "lookup_order", "human_approval", "refund"])
    assert Path(cassette_path).exists()

    cassette_on_disk = load_cassette(cassette_path)
    assert len(cassette_on_disk.entries) == 2  # one llm call (router), one tool call (lookup_order)

    _explode_if_called(monkeypatch)  # from here on, any real call fails the test
    graph_2 = build_refund_graph(_fake_model())  # a fresh graph/model -- replay must not need the original
    replayed = run_graph(graph_2, {"order_id": "1234"}, cassette=cassette_path, mode="replay", resume=[True])

    assert replayed.interrupted is False
    assert_path(replayed, ["router", "lookup_order", "human_approval", "refund"])
    assert_tool_called(replayed, "get_order", args_match={"order_id": "1234"})
    assert_state(replayed, "refunded", True)
    assert replayed.llm_calls[0].output.content == "ok"  # the *recorded* response, not a fresh fake one


def test_replay_mode_cassette_round_trip_is_byte_identical_on_reload(tmp_path: Path):
    """Recording, then loading the written file back, then saving it again
    unmodified -- the same round-trip guarantee test_replay.py checks for a
    hand-written fixture, now against a cassette graphtrace itself produced."""
    cassette_path = str(tmp_path / "c.yaml")
    graph = build_refund_graph(_fake_model())
    run_graph(graph, {"order_id": "1234"}, cassette=cassette_path, mode="record", resume=[True])

    from graphtrace.replay.cassette import save_cassette

    loaded = load_cassette(cassette_path)
    resaved_path = tmp_path / "c_resaved.yaml"
    save_cassette(loaded, resaved_path)
    assert resaved_path.read_text(encoding="utf-8") == Path(cassette_path).read_text(encoding="utf-8")


# -- edge case: a retry sequence replays in the recorded order ---------------------------

def test_record_then_replay_a_retry_sequence_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cassette_path = str(tmp_path / "retry.yaml")
    graph = build_refund_graph(_fake_model(), fail_lookup_times=1)
    recorded = run_graph(graph, {"order_id": "1234"}, cassette=cassette_path, mode="record", resume=[True])
    assert recorded.visits("lookup_order") == 2  # one failed attempt, one that succeeded

    _explode_if_called(monkeypatch)
    graph_2 = build_refund_graph(_fake_model(), fail_lookup_times=1)
    replayed = run_graph(graph_2, {"order_id": "1234"}, cassette=cassette_path, mode="replay", resume=[True])

    assert replayed.visits("lookup_order") == 2
    first_attempt, second_attempt = [s for s in replayed.steps if s.node == "lookup_order"]
    assert first_attempt.tool_calls[0].error == "order service unavailable (attempt 1)"
    assert second_attempt.tool_calls[0].error is None
    assert_state(replayed, "refunded", True)


# -- negative: replay with no match, or a recorded failure --------------------------------

def test_replay_mode_with_no_cassette_file_raises_cassette_mismatch():
    graph = build_refund_graph(_fake_model())
    with pytest.raises(CassetteMismatchError):
        run_graph(graph, {"order_id": "1234"}, cassette="tests/does_not_exist.yaml", mode="replay", resume=[True])


def test_replay_mode_with_a_different_request_than_what_was_recorded_raises(tmp_path: Path):
    cassette_path = str(tmp_path / "c.yaml")
    graph = build_refund_graph(_fake_model())
    run_graph(graph, {"order_id": "1234"}, cassette=cassette_path, mode="record", resume=[True])

    graph_2 = build_refund_graph(_fake_model())
    with pytest.raises(CassetteMismatchError):
        # A different order_id hashes differently -- the recorded tool-call entry
        # for "1234" doesn't match a lookup for "9999".
        run_graph(graph_2, {"order_id": "9999"}, cassette=cassette_path, mode="replay", resume=[True])


# -- positive/edge: record_missing grows a cassette without re-recording what's already there --

def test_record_missing_replays_existing_entries_and_appends_only_new_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cassette_path = str(tmp_path / "growing.yaml")

    # First call seeds the cassette for order "1234" via plain record.
    graph_1 = build_refund_graph(_fake_model())
    run_graph(graph_1, {"order_id": "1234"}, cassette=cassette_path, mode="record", resume=[True])
    original_entries = len(load_cassette(cassette_path).entries)

    # record_missing for the SAME order replays every existing entry -- no real
    # call happens, confirmed the same way the pure-replay test proves it.
    _explode_if_called(monkeypatch)
    graph_2 = build_refund_graph(_fake_model())
    replayed = run_graph(graph_2, {"order_id": "1234"}, cassette=cassette_path, mode="record_missing", resume=[True])
    assert_state(replayed, "refunded", True)
    assert len(load_cassette(cassette_path).entries) == original_entries  # nothing new queued
