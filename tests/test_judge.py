"""Tests for judge/rubric.py, judge/client.py, judge/calibrate.py.

Uses a fake `litellm` module injected via `sys.modules` for `JudgeClient`
tests -- no real network call, no API key needed, and (unlike litellm's own
`mock_response`, which is one static string) full control over a *sequence* of
responses, needed to test the retry-once-on-malformed-JSON path precisely.
litellm's real response shape (`response.choices[0].message.content`) was
confirmed by hand against the real installed package via `mock_response`
before writing this fake to match it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphtrace.judge.calibrate import calibrate, load_calibration_cases
from graphtrace.judge.client import GradeResult, JudgeClient, build_prompt, grade
from graphtrace.judge.errors import JudgeError
from graphtrace.judge.rubric import Rubric, load_rubric


# -- rubric.py ----------------------------------------------------------------------------

def test_load_rubric_parses_a_valid_file(tmp_path: Path):
    path = tmp_path / "refund_reply.yaml"
    path.write_text(
        """
id: refund_reply
version: 3
criteria:
  - id: confirms_amount
    description: Reply states the refund amount.
  - id: no_promises
    description: Reply does not promise a timeline.
scale: pass_fail
pass_threshold: all
""",
        encoding="utf-8",
    )
    rubric = load_rubric(path)
    assert rubric.id == "refund_reply"
    assert len(rubric.criteria) == 2
    assert rubric.criteria[0].id == "confirms_amount"


def test_rubric_rejects_empty_criteria():
    with pytest.raises(ValidationError):
        Rubric(id="x", version=1, criteria=[])


def test_rubric_rejects_an_unknown_field():
    with pytest.raises(ValidationError):
        Rubric(id="x", version=1, criteria=[{"id": "a", "description": "d"}], typo_field=True)


# -- judge/client.py: a fake litellm module -----------------------------------------------

def _install_fake_litellm(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> list[dict]:
    """Each call to completion() pops the next canned response off `responses`
    and returns a fake ModelResponse-shaped object exposing
    `.choices[0].message.content`, exactly matching litellm's real shape
    (confirmed by hand against the real package before writing this)."""
    calls: list[dict] = []
    remaining = list(responses)

    def fake_completion(**kwargs):
        calls.append(kwargs)
        content = remaining.pop(0)
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    fake_module = types.ModuleType("litellm")
    fake_module.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return calls


def test_ask_json_succeeds_on_first_valid_response(monkeypatch):
    calls = _install_fake_litellm(monkeypatch, ['{"criteria": {"a": {"pass": true}}}'])
    client = JudgeClient(model="fake-model")
    result = client.ask_json("grade this")
    assert result == {"criteria": {"a": {"pass": True}}}
    assert len(calls) == 1
    assert calls[0]["model"] == "fake-model"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_ask_json_retries_once_on_malformed_json_then_succeeds(monkeypatch):
    calls = _install_fake_litellm(monkeypatch, ["not json at all", '{"criteria": {}}'])
    client = JudgeClient(model="fake-model")
    result = client.ask_json("grade this")
    assert result == {"criteria": {}}
    assert len(calls) == 2  # the retry really happened


def test_ask_json_raises_judge_error_after_two_malformed_responses(monkeypatch):
    _install_fake_litellm(monkeypatch, ["not json", "still not json"])
    client = JudgeClient(model="fake-model")
    with pytest.raises(JudgeError):
        client.ask_json("grade this")


def test_judge_client_construction_never_imports_litellm():
    """Confirms the lazy-import discipline the module docstring claims --
    constructing a JudgeClient must not require litellm to be importable."""
    sys.modules.pop("litellm", None)
    JudgeClient(model="whatever")  # no ImportError, because nothing imports it here


# -- judge/client.py: grade() + cassette caching --------------------------------------------

def _rubric() -> Rubric:
    return Rubric(
        id="refund_reply",
        version=1,
        criteria=[{"id": "confirms_amount", "description": "States the amount."}],
    )


def test_grade_record_mode_calls_the_judge_and_writes_a_cassette(tmp_path, monkeypatch):
    cassette_path = str(tmp_path / "judge.yaml")
    _install_fake_litellm(monkeypatch, ['{"criteria": {"confirms_amount": {"pass": true, "reason": "ok"}}}'])
    client = JudgeClient(model="fake-model")

    result = grade(client, _rubric(), "refund?", "yes, $49.00 refunded", cassette=cassette_path, mode="record")

    assert isinstance(result, GradeResult)
    assert result.passed is True
    assert Path(cassette_path).exists()


def test_grade_replay_mode_returns_cached_verdict_without_calling_the_judge(tmp_path, monkeypatch):
    cassette_path = str(tmp_path / "judge.yaml")
    _install_fake_litellm(monkeypatch, ['{"criteria": {"confirms_amount": {"pass": false, "reason": "no amount stated"}}}'])
    client = JudgeClient(model="fake-model")
    grade(client, _rubric(), "refund?", "sorry, no refund", cassette=cassette_path, mode="record")

    def boom(*args, **kwargs):
        raise AssertionError("the judge must not be called again in replay mode")
    monkeypatch.setattr(JudgeClient, "ask_json", boom)

    replayed = grade(client, _rubric(), "refund?", "sorry, no refund", cassette=cassette_path, mode="replay")
    assert replayed.passed is False
    assert replayed.criteria["confirms_amount"]["reason"] == "no amount stated"


def test_grade_rejects_a_1_to_5_scale_rubric():
    rubric = Rubric(id="x", version=1, criteria=[{"id": "a", "description": "d"}], scale="1-5")
    client = JudgeClient(model="fake-model")
    with pytest.raises(NotImplementedError):
        grade(client, rubric, "in", "out", cassette="unused", mode="live")


def test_build_prompt_includes_every_criterion_and_tool_results():
    rubric = _rubric()
    prompt = build_prompt(rubric, "the input", "the output", tool_results={"amount": 49.0})
    assert "confirms_amount" in prompt
    assert "the input" in prompt
    assert "the output" in prompt
    assert "49.0" in prompt


# -- judge/calibrate.py ----------------------------------------------------------------------

def test_load_calibration_cases_filters_by_rubric_id(tmp_path):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        '{"rubric": "refund_reply", "input": "a", "output": "b", "human": {"confirms_amount": true}}\n'
        '{"rubric": "other_rubric", "input": "x", "output": "y", "human": {}}\n',
        encoding="utf-8",
    )
    cases = load_calibration_cases(path, "refund_reply")
    assert len(cases) == 1
    assert cases[0]["input"] == "a"


def test_calibrate_perfect_agreement_scores_kappa_one(tmp_path, monkeypatch):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        "\n".join(
            f'{{"rubric": "refund_reply", "input": "in{i}", "output": "out{i}", "human": {{"confirms_amount": {str(label).lower()}}}}}'
            for i, label in enumerate([True, True, False, False])
        ),
        encoding="utf-8",
    )
    responses = [
        f'{{"criteria": {{"confirms_amount": {{"pass": {str(label).lower()}}}}}}}'
        for label in [True, True, False, False]
    ]
    _install_fake_litellm(monkeypatch, responses)
    client = JudgeClient(model="fake-model")

    results = calibrate(client, _rubric(), path)
    assert results["confirms_amount"].kappa == 1.0
    assert results["confirms_amount"].agreement == 1.0
    assert results["confirms_amount"].n == 4


def test_calibrate_chance_level_agreement_scores_kappa_zero(tmp_path, monkeypatch):
    """Not "every label disagrees" -- that's actually worse than chance (kappa
    goes *negative*, confirmed by hand: cohen_kappa_score([1,1,0,0],[0,0,1,1])
    is -1.0, not 0.0, since systematically inverting a balanced label is a
    real, measurable anti-correlation). Kappa 0.0 means the judge agrees with
    human exactly as often as random guessing would -- this specific
    human/judge pair (confirmed by hand before trusting it in a test) is the
    real example of that."""
    human = [True, True, False, False]
    judge = [True, False, True, False]
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        "\n".join(
            f'{{"rubric": "refund_reply", "input": "in{i}", "output": "out{i}", "human": {{"confirms_amount": {str(h).lower()}}}}}'
            for i, h in enumerate(human)
        ),
        encoding="utf-8",
    )
    responses = [f'{{"criteria": {{"confirms_amount": {{"pass": {str(j).lower()}}}}}}}' for j in judge]
    _install_fake_litellm(monkeypatch, responses)
    client = JudgeClient(model="fake-model")

    results = calibrate(client, _rubric(), path)
    assert results["confirms_amount"].kappa == 0.0


def test_calibrate_skips_cases_the_judge_fails_to_answer(tmp_path, monkeypatch):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        '{"rubric": "refund_reply", "input": "a", "output": "b", "human": {"confirms_amount": true}}\n',
        encoding="utf-8",
    )
    _install_fake_litellm(monkeypatch, ["not json", "still not json"])  # JudgeError both attempts
    client = JudgeClient(model="fake-model")

    results = calibrate(client, _rubric(), path)
    assert results["confirms_amount"].n == 0  # skipped, not counted as disagreement
