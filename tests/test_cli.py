"""Tests for cli.py's three commands (`from-trace`, `calibrate`, `report`).

`from_trace_end_to_end` is BUILD_PLAN.md section 7.6's own verify-by, almost
word for word: "graphtrace from-trace -> pytest --gt-mode=replay end to end on
a real trace fixture." It runs the real CLI command against a real,
Langfuse-shaped trace fixture, then actually spawns `pytest` as a subprocess
on the file it generated and confirms that run passes -- not just that
generate_test() produced *some* string (test_codegen.py already covers that
in isolation).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graphtrace.cli import app
from graphtrace.report.results import TestResult, save_results

runner = CliRunner()

_EXAMPLES_DIR = str(Path(__file__).parent.parent / "examples")

_LANGFUSE_TRACE = {
    "id": "trace-1",
    "input": {"order_id": "9999"},
    "observations": [
        {
            "type": "GENERATION",
            "name": "router",
            "input": {"messages": ["What does the customer want for order 9999?"]},
            "output": {"content": "ok"},
            "model": "demo-model",
            "startTime": "2026-10-20T10:12:00",
            "endTime": "2026-10-20T10:12:00.400000",
        },
        {
            "type": "SPAN",
            "name": "lookup_order",
            "input": {"order_id": "9999"},
            "output": {"status": "delivered", "amount": 49.0},
            "startTime": "2026-10-20T10:12:01",
            "endTime": "2026-10-20T10:12:01.200000",
        },
    ],
}


# -- from-trace: real end-to-end (BUILD_PLAN.md 7.6's own verify-by) -----------------------

def test_from_trace_end_to_end_generated_test_passes_under_pytest_replay(tmp_path: Path):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps(_LANGFUSE_TRACE), encoding="utf-8")
    out_file = tmp_path / "test_from_trace_generated.py"
    cassette_file = tmp_path / "cassette.yaml"

    result = runner.invoke(app, [
        "from-trace", str(trace_file),
        "--target", "refund_agent.graph:build_refund_graph_demo",
        "--test-name", "refund_demo",
        "--out", str(out_file),
        "--cassette", str(cassette_file),
    ])

    assert result.exit_code == 0, result.output
    assert "detected source: langfuse" in result.output
    assert out_file.exists() and cassette_file.exists()

    # The real verification: spawn pytest as a subprocess on exactly the file
    # the CLI just wrote, with --gt-mode=replay -- no cassette entry is hit by
    # coincidence here, since the cassette didn't exist until this command
    # just wrote it.
    env = {**__import__("os").environ, "PYTHONPATH": _EXAMPLES_DIR}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(out_file), "--gt-mode=replay", "-q", "-p", "no:cacheprovider"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed" in proc.stdout


def test_from_trace_rejects_a_trace_shape_no_importer_recognizes(tmp_path: Path):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps({"nothing": "recognizable"}), encoding="utf-8")

    result = runner.invoke(app, ["from-trace", str(trace_file), "--target", "refund_agent.graph:build_refund_graph_demo"])
    assert result.exit_code != 0


def test_from_trace_defaults_test_name_to_the_trace_files_stem(tmp_path: Path):
    trace_file = tmp_path / "my_export.json"
    trace_file.write_text(json.dumps(_LANGFUSE_TRACE), encoding="utf-8")

    result = runner.invoke(app, [
        "from-trace", str(trace_file),
        "--target", "refund_agent.graph:build_refund_graph_demo",
        "--out", str(tmp_path / "generated.py"),
        "--cassette", str(tmp_path / "c.yaml"),
    ])
    assert result.exit_code == 0, result.output
    assert "def test_my_export():" in (tmp_path / "generated.py").read_text(encoding="utf-8")


# -- calibrate -------------------------------------------------------------------------------

def _write_calibration_project(tmp_path: Path, *, judge_responses: list[str], min_agreement: float = 0.8) -> Path:
    (tmp_path / "rubrics").mkdir()
    (tmp_path / "rubrics" / "refund_reply.yaml").write_text(
        "id: refund_reply\nversion: 1\ncriteria:\n  - id: confirms_amount\n    description: States the amount.\n",
        encoding="utf-8",
    )
    (tmp_path / "calibration.jsonl").write_text(
        "\n".join(
            f'{{"rubric": "refund_reply", "input": "in{i}", "output": "out{i}", "human": {{"confirms_amount": {str(h).lower()}}}}}'
            for i, h in enumerate([True, True, False, False])
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "graphtrace.yaml"
    config_path.write_text(
        f"judge:\n  model: fake-model\n  rubrics_dir: {tmp_path / 'rubrics'}\n"
        f"  calibration_set: {tmp_path / 'calibration.jsonl'}\n  min_agreement: {min_agreement}\n",
        encoding="utf-8",
    )
    return config_path


def _install_fake_litellm(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> None:
    import types
    remaining = list(responses)

    def fake_completion(**kwargs):
        content = remaining.pop(0)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])

    fake_module = types.ModuleType("litellm")
    fake_module.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_module)


def test_calibrate_exits_zero_on_good_agreement(tmp_path: Path, monkeypatch):
    config_path = _write_calibration_project(tmp_path, judge_responses=[])
    _install_fake_litellm(monkeypatch, [
        '{"criteria": {"confirms_amount": {"pass": true}}}',
        '{"criteria": {"confirms_amount": {"pass": true}}}',
        '{"criteria": {"confirms_amount": {"pass": false}}}',
        '{"criteria": {"confirms_amount": {"pass": false}}}',
    ])
    result = runner.invoke(app, ["calibrate", "--rubric", "refund_reply", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_calibrate_exits_nonzero_when_kappa_is_below_min_agreement(tmp_path: Path, monkeypatch):
    config_path = _write_calibration_project(tmp_path, judge_responses=[], min_agreement=0.99)
    _install_fake_litellm(monkeypatch, [
        '{"criteria": {"confirms_amount": {"pass": true}}}',
        '{"criteria": {"confirms_amount": {"pass": false}}}',  # disagreement
        '{"criteria": {"confirms_amount": {"pass": false}}}',
        '{"criteria": {"confirms_amount": {"pass": false}}}',
    ])
    result = runner.invoke(app, ["calibrate", "--rubric", "refund_reply", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "BELOW THRESHOLD" in result.output


def test_calibrate_errors_clearly_when_no_judge_config_exists(tmp_path: Path):
    config_path = tmp_path / "graphtrace.yaml"
    config_path.write_text("target: null\n", encoding="utf-8")
    result = runner.invoke(app, ["calibrate", "--rubric", "refund_reply", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "no `judge:` section" in result.output


# -- report ------------------------------------------------------------------------------

def test_report_renders_html_from_a_saved_results_json(tmp_path: Path):
    results_path = tmp_path / "results.json"
    save_results([TestResult(test_id="test_a", outcome="passed")], results_path)
    out_path = tmp_path / "out.html"

    result = runner.invoke(app, ["report", str(results_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert "test_a" in out_path.read_text(encoding="utf-8")


def test_report_defaults_output_path_to_config_report_html(tmp_path: Path):
    results_path = tmp_path / "results.json"
    save_results([TestResult(test_id="test_a", outcome="passed")], results_path)
    config_path = tmp_path / "graphtrace.yaml"
    config_path.write_text(f"report:\n  html: {tmp_path / 'custom_report.html'}\n", encoding="utf-8")

    result = runner.invoke(app, ["report", str(results_path), "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "custom_report.html").exists()


def test_report_accepts_a_custom_title(tmp_path: Path):
    """Without this, there was no way to reproduce a previously-rendered
    report byte-for-byte from its own results.json -- the title always fell
    back to the one hardcoded default, confirmed by hand while building
    examples/sample_report/ before fixing it here."""
    results_path = tmp_path / "results.json"
    save_results([TestResult(test_id="test_a", outcome="passed")], results_path)
    out_path = tmp_path / "out.html"

    result = runner.invoke(app, ["report", str(results_path), "--out", str(out_path), "--title", "my custom title"])
    assert result.exit_code == 0, result.output
    html = out_path.read_text(encoding="utf-8")
    assert "<title>my custom title</title>" in html
    assert "<h1>my custom title</h1>" in html
