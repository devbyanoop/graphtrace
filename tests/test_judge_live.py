"""Real integration tests against a live Groq model via litellm -- the
live-provider counterpart to tests/test_judge.py's fake-litellm unit tests.

Needs GROQ_API_KEY set via .env or the real environment (see .env.example).
Skips cleanly (not an error) when it isn't set, the same way kbguard's
Pinecone live test skips without PINECONE_API_KEY.

The model string (groq/openai/gpt-oss-120b) and the JSON-mode behaviour it's
relied on here were confirmed by hand with a real, throwaway call before any
of this was written -- not assumed from `model_list` or kbguard's own
.env comment alone.

evals/calibration.jsonl is a small, 4-case, hand-authored *demonstration*
set -- proving the real calibrate() pipeline works end to end against a
live model, not a statistically meaningful calibration. A real production
threshold needs the 20-50 cases per rubric the architecture plan itself
calls for; graphtrace.yaml's `min_agreement: 0.5` reflects that honestly
(loose, because n=4 is noisy, not because 0.5 is a real target).
"""

from __future__ import annotations

import os

import pytest

from graphtrace.config import load_config
from graphtrace.judge.calibrate import calibrate
from graphtrace.judge.client import JudgeClient, grade
from graphtrace.judge.rubric import load_rubric

pytestmark = pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set -- see .env.example")


def _config():
    return load_config("graphtrace.yaml")


def _client() -> JudgeClient:
    return JudgeClient(model=_config().judge.model)


def _rubric():
    config = _config()
    return load_rubric(f"{config.judge.rubrics_dir}refund_reply.yaml")


# -- grade(): a real Groq call, real JSON parsed back into a GradeResult -------------------

def test_grade_against_a_real_groq_model_scores_an_obvious_pass(tmp_path):
    result = grade(
        _client(), _rubric(),
        "Can I get a refund for my order?",
        "Yes, $49.00 has been refunded to your account.",
        cassette=str(tmp_path / "judge.yaml"), mode="record",
    )
    assert result.passed is True
    assert result.criteria["confirms_amount"]["pass"] is True
    assert result.criteria["no_promises"]["pass"] is True
    # Confirms the judge gave a real reason, not an empty/placeholder string --
    # the actual content isn't pinned, since that would be asserting on a real
    # model's exact wording, which is exactly the kind of brittle live-model
    # test this project avoids everywhere else (cassettes exist for a reason).
    assert len(result.criteria["confirms_amount"].get("reason", "")) > 0


def test_grade_against_a_real_groq_model_scores_an_obvious_fail(tmp_path):
    result = grade(
        _client(), _rubric(),
        "Can I get a refund for my order?",
        "Sorry, we can't help with that right now.",
        cassette=str(tmp_path / "judge.yaml"), mode="record",
    )
    assert result.passed is False
    assert result.criteria["confirms_amount"]["pass"] is False


def test_grade_cassette_cache_really_avoids_a_second_live_call(tmp_path, monkeypatch):
    """The actual claim docs/guide/judge-and-calibration.md makes -- "a replay
    run never calls the judge" -- proven here against a *real* recorded
    cassette, not the fake-litellm version test_judge.py already covers."""
    cassette_path = str(tmp_path / "judge.yaml")
    client = _client()
    recorded = grade(client, _rubric(), "Can I get a refund?", "Yes, $49.00 refunded.", cassette=cassette_path, mode="record")

    def boom(*args, **kwargs):
        raise AssertionError("a replay-mode grade() call must never hit the real judge")
    monkeypatch.setattr(JudgeClient, "ask_json", boom)

    replayed = grade(client, _rubric(), "Can I get a refund?", "Yes, $49.00 refunded.", cassette=cassette_path, mode="replay")
    assert replayed.passed == recorded.passed
    assert replayed.criteria == recorded.criteria


# -- calibrate(): the real pipeline, against the small demonstration set -------------------

def test_calibrate_against_a_real_groq_model_runs_end_to_end():
    config = _config()
    results = calibrate(_client(), _rubric(), config.judge.calibration_set)

    assert set(results.keys()) == {"confirms_amount", "no_promises"}
    for criterion_id, calibration in results.items():
        assert calibration.n == 4, f"{criterion_id}: expected all 4 demonstration cases to be answered"
        assert -1.0 <= calibration.kappa <= 1.0
        # Loose on purpose (see module docstring): the 4 cases are each
        # clear-cut by design, so a reasonably capable judge should agree with
        # most of them, but this isn't pinned to kappa == 1.0 -- that would be
        # asserting deterministic behaviour from a real, live model.
        assert calibration.agreement >= 0.5, f"{criterion_id}: agreement {calibration.agreement} suspiciously low for clear-cut cases"
