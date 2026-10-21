"""Tests for config.py (graphtrace.yaml loading) and core/budgets.py (the
config-driven default-budget check the run_graph fixture applies automatically).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from graphtrace import TrajectoryAssertionError, Trajectory
from graphtrace.config import GraphTraceConfig, load_config
from graphtrace.core.budgets import Budgets


# -- config.py: positive -----------------------------------------------------------------

def test_load_config_missing_file_returns_all_defaults(tmp_path: Path):
    config = load_config(tmp_path / "does_not_exist.yaml")
    assert config == GraphTraceConfig()
    assert config.framework == "langgraph"
    assert config.replay.mode == "replay"


def test_load_config_parses_a_full_valid_file(tmp_path: Path):
    path = tmp_path / "graphtrace.yaml"
    path.write_text(
        """
target: my_app.graph:build_graph
framework: langgraph
replay:
  mode: record
  cassettes: tests/cassettes/
  normalize:
    - pattern: '\\d{4}-\\d{2}-\\d{2}'
      replace: '<date>'
  redact: [api_key]
budgets:
  default: {cost_usd: 0.1, latency_s: 15}
report:
  html: reports/custom.html
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.target == "my_app.graph:build_graph"
    assert config.replay.mode == "record"
    assert config.replay.normalize[0].replace == "<date>"
    assert config.budgets.default == {"cost_usd": 0.1, "latency_s": 15.0}
    assert config.report.html == "reports/custom.html"


# -- config.py: negative / edge ------------------------------------------------------------

def test_load_config_rejects_an_unknown_top_level_field(tmp_path: Path):
    path = tmp_path / "graphtrace.yaml"
    path.write_text("tagret: typo-of-target\n", encoding="utf-8")  # typo'd key
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_rejects_an_invalid_framework_value(tmp_path: Path):
    path = tmp_path / "graphtrace.yaml"
    path.write_text("framework: crewai\n", encoding="utf-8")  # not valid until v0.4
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_rejects_an_invalid_replay_mode(tmp_path: Path):
    path = tmp_path / "graphtrace.yaml"
    path.write_text("replay:\n  mode: fast-forward\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_judge_config_requires_a_model_when_present(tmp_path: Path):
    path = tmp_path / "graphtrace.yaml"
    path.write_text("judge:\n  min_agreement: 0.9\n", encoding="utf-8")  # no `model`
    with pytest.raises(ValidationError):
        load_config(path)


# -- core/budgets.py: positive / negative --------------------------------------------------

def _traj(cost_usd: float = 0.0, latency_s: float = 0.0) -> Trajectory:
    from graphtrace import LLMCall, Step

    call = LLMCall(node="a", model="x", input_hash="h", prompt_tokens=0, completion_tokens=0, latency_s=0.0, output="", cost_usd=cost_usd)
    step = Step(index=0, node="a", kind="node", state_before={}, state_after={}, state_diff={}, llm_calls=[call], latency_s=latency_s)
    return Trajectory(steps=[step], final_state={}, interrupted=False, mode="live")


def test_budgets_check_passes_within_configured_default():
    budgets = Budgets(cost_usd=0.10, latency_s=15)
    budgets.check(_traj(cost_usd=0.05, latency_s=5))  # no raise


def test_budgets_check_fails_when_exceeding_configured_default():
    budgets = Budgets(cost_usd=0.10)
    with pytest.raises(TrajectoryAssertionError):
        budgets.check(_traj(cost_usd=0.50))


def test_budgets_check_is_a_noop_with_nothing_configured():
    Budgets().check(_traj(cost_usd=999.0, latency_s=999.0))  # no raise -- nothing set, nothing checked
