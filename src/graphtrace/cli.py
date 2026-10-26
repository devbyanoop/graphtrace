"""The `graphtrace` console command -- wires every command actually built so
far (`from-trace`, `calibrate`, `report`) into one typer app.

Deliberately does *not* include `graphtrace record` or `graphtrace redteam` --
both are still listed "Planned" in docs/reference/cli.md, and this project's
own rule (already applied to plugin.py's fixtures) is not to ship a command
that doesn't actually do anything yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from graphtrace.codegen.test_writer import generate_test
from graphtrace.config import load_config
from graphtrace.importers.base import TraceImporter
from graphtrace.importers.langfuse import LangfuseImporter
from graphtrace.importers.langsmith import LangSmithImporter
from graphtrace.importers.otel import OTelImporter
from graphtrace.judge.calibrate import calibrate as _calibrate
from graphtrace.judge.client import JudgeClient
from graphtrace.judge.rubric import load_rubric
from graphtrace.report.html import render_report
from graphtrace.report.results import load_results

app = typer.Typer(no_args_is_help=True, help="Regression testing for LangGraph agents.")

_IMPORTERS: list[TraceImporter] = [LangfuseImporter(), OTelImporter(), LangSmithImporter()]


def _detect_importer(data: dict) -> TraceImporter:
    for importer in _IMPORTERS:
        if importer.can_read(data):
            return importer
    names = ", ".join(i.name for i in _IMPORTERS)
    raise typer.BadParameter(f"none of the known importers ({names}) recognize this trace's shape")


@app.command("from-trace")
def from_trace(
    trace_file: Path = typer.Argument(..., help="An exported trace (Langfuse/OTel/LangSmith JSON)."),
    target: str = typer.Option(..., "--target", help="module:factory for the graph under test, e.g. 'my_app.graph:build_graph'."),
    test_name: Optional[str] = typer.Option(None, "--test-name", help="Defaults to the trace file's stem."),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write the generated test file. Defaults to tests/test_from_<test-name>.py."),
    cassette: Optional[Path] = typer.Option(None, "--cassette", help="Where to write the cassette. Defaults to tests/cassettes/from_<test-name>.yaml."),
) -> None:
    """Convert an exported trace into a cassette + a generated pytest file that
    replays the observed run (see docs/guide/trace-to-test.md)."""
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    importer = _detect_importer(data)
    trace = importer.to_normalized(data)

    name = test_name or trace_file.stem
    cassette_path = cassette or Path("tests/cassettes") / f"from_{name}.yaml"
    out_path = out or Path("tests") / f"test_from_{name}.py"

    source = generate_test(trace, target=target, test_name=name, cassette_path=cassette_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source, encoding="utf-8")

    typer.echo(f"detected source: {importer.name}")
    typer.echo(f"wrote cassette: {cassette_path}")
    typer.echo(f"wrote test: {out_path}")


@app.command("calibrate")
def calibrate_cmd(
    rubric: str = typer.Option(..., "--rubric", help="Rubric id -- loaded from <judge.rubrics_dir>/<rubric>.yaml."),
    config_path: Path = typer.Option(Path("graphtrace.yaml"), "--config", help="Path to graphtrace.yaml."),
) -> None:
    """Score the judge against graphtrace.yaml's judge.calibration_set, reporting
    per-criterion agreement and Cohen's kappa. Exits non-zero if any criterion
    with at least one labelled case falls below judge.min_agreement."""
    config = load_config(config_path)
    if config.judge is None:
        typer.echo("no `judge:` section in graphtrace.yaml -- nothing to calibrate", err=True)
        raise typer.Exit(code=1)

    rubric_path = Path(config.judge.rubrics_dir) / f"{rubric}.yaml"
    loaded_rubric = load_rubric(rubric_path)
    client = JudgeClient(model=config.judge.model)
    results = _calibrate(client, loaded_rubric, config.judge.calibration_set)

    below_threshold = False
    for criterion_id, calibration in results.items():
        if calibration.n == 0:
            typer.echo(f"{criterion_id}: no labelled cases -- skipped")
            continue
        flag = "OK" if calibration.kappa >= config.judge.min_agreement else "BELOW THRESHOLD"
        typer.echo(f"{criterion_id}: agreement={calibration.agreement:.2f} kappa={calibration.kappa:.2f} n={calibration.n} [{flag}]")
        if calibration.kappa < config.judge.min_agreement:
            below_threshold = True

    if below_threshold:
        raise typer.Exit(code=1)


@app.command("report")
def report_cmd(
    results_file: Path = typer.Argument(..., help="A results.json written by `pytest --gt-report=...` or report/results.py's save_results()."),
    out: Optional[Path] = typer.Option(None, "--out", help="Defaults to graphtrace.yaml's report.html."),
    config_path: Path = typer.Option(Path("graphtrace.yaml"), "--config", help="Path to graphtrace.yaml."),
) -> None:
    """Render the static HTML report from a saved results.json."""
    config = load_config(config_path)
    out_path = out or Path(config.report.html)

    results = load_results(results_file)
    html = render_report(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    typer.echo(f"wrote report: {out_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
