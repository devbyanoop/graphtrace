# CLI reference

`pip install -e .` installs a `graphtrace` console command (`[project.scripts]` in `pyproject.toml`) wiring three commands — exactly what's actually built. `graphtrace record` and `graphtrace redteam` (the security test pack, v0.2) are intentionally not here yet; see [Roadmap & status](../project/roadmap.md).

## `graphtrace from-trace`

```bash
graphtrace from-trace trace.json --target my_app.graph:build_graph [--test-name NAME] [--out PATH] [--cassette PATH]
```

Converts an exported trace into a cassette and a generated pytest file. See [Trace-to-test](../guide/trace-to-test.md) for the real importer/codegen scope and the honest limits documented there.

| Option | Default |
| --- | --- |
| `--target` | *(required)* `module:factory` for the graph under test |
| `--test-name` | the trace file's stem |
| `--out` | `tests/test_from_<test-name>.py` |
| `--cassette` | `tests/cassettes/from_<test-name>.yaml` |

## `graphtrace calibrate`

```bash
graphtrace calibrate --rubric refund_reply [--config graphtrace.yaml]
```

Scores the judge against `graphtrace.yaml`'s `judge.calibration_set`, printing per-criterion agreement and Cohen's kappa. Exits `1` if any criterion with at least one labelled case falls below `judge.min_agreement`, or if `graphtrace.yaml` has no `judge:` section at all. See [Judge & calibration](../guide/judge-and-calibration.md).

## `graphtrace report`

```bash
graphtrace report results.json [--out PATH] [--title "..."] [--config graphtrace.yaml]
```

Renders the static HTML report from an existing `results.json` (written by `pytest --gt-report=...`, or by `report/results.py`'s `save_results()` directly). `--out` defaults to `graphtrace.yaml`'s `report.html`; `--title` defaults to `"graphtrace report"` (set it to reproduce a previous render exactly — see `examples/sample_report/`). See [Report & CI](../guide/report-and-ci.md).

## pytest options

| Flag / env var | Controls |
| --- | --- |
| `--gt-mode` / `GRAPHTRACE_MODE` | `replay` \| `record` \| `record_missing` \| `live` |
| `--gt-report` | Write a static HTML report (+ `results.json` alongside it) at session end |
| `--gt-export` | Observability export target (v0.2, not built yet) |

See [Recording & replay](../guide/recording-and-replay.md) for what each mode means.
