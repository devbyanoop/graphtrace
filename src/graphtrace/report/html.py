"""render_report(): TestResult list -> one self-contained HTML string, via
report/templates/report.html.j2.

`autoescape=True` is passed explicitly and unconditionally, not inferred from
the template's filename via `select_autoescape` -- trajectory state, tool
args/results, and an LLM's own output can all hold arbitrary client data (a
user's refund note, a crafted prompt injection attempt...), the same concern
kbguard's own HTML report has. Confirmed by hand (test_html.py's own XSS test)
that a `<script>` anywhere in that data renders as escaped text, not live markup.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from graphtrace.report.results import TestResult, build_executive_summary

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _summarize(results: list[TestResult]) -> dict[str, float | int]:
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r.outcome == "passed"),
        "failed": sum(1 for r in results if r.outcome == "failed"),
        "errored": sum(1 for r in results if r.outcome == "error"),
        "total_cost_usd": sum((r.trajectory or {}).get("cost_usd", 0.0) for r in results),
        "total_latency_s": sum((r.trajectory or {}).get("latency_s", 0.0) for r in results),
    }


def render_report(results: list[TestResult], *, title: str = "graphtrace report") -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=True)
    template = env.get_template("report.html.j2")
    return template.render(title=title, results=results, summary=_summarize(results), exec_summary=build_executive_summary(results))
