"""Load and validate `graphtrace.yaml` -- the single place in the codebase that
reads this file's format; nothing else calls a YAML loader directly for it.

Uses `ruamel.yaml`'s *safe* loader (`typ="safe"`), not the round-trip mode
`replay/cassette.py` uses -- there's no reason to rewrite a project's own config
file programmatically the way cassettes get grown, so there's nothing here that
needs round-trip's formatting-preservation, and safe mode hands back plain
`float`/`int`/`dict` directly (confirmed by hand) rather than the ScalarFloat-style
subclasses round-trip mode produces (see `replay/cassette.py`'s `to_plain_python`
for why that distinction mattered enough to need its own fix elsewhere).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML

_yaml = YAML(typ="safe")


class NormalizeRuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str
    replace: str


class ReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["record", "replay", "record_missing", "live"] = "replay"
    cassettes: str = "tests/cassettes/"
    normalize: list[NormalizeRuleConfig] = []
    redact: list[str] = []


class BudgetsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default: dict[str, float] = {}


class JudgeConfig(BaseModel):
    """Week 3. Present now so a project can write its real judge config ahead of
    `judge/client.py` existing, without a schema change later."""

    model_config = ConfigDict(extra="forbid")
    model: str
    rubrics_dir: str = "rubrics/"
    calibration_set: str = "evals/calibration.jsonl"
    min_agreement: float = 0.8


class ReportConfig(BaseModel):
    """Week 3."""

    model_config = ConfigDict(extra="forbid")
    html: str = "reports/graphtrace.html"


class GraphTraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str | None = None
    # "langgraph" is the only valid value until v0.4 (see docs/guide/multi-framework.md)
    # -- enforced here, not just documented, the same way every other "not built yet"
    # boundary in this project is a real constraint, not a comment.
    framework: Literal["langgraph"] = "langgraph"
    replay: ReplayConfig = ReplayConfig()
    judge: JudgeConfig | None = None
    budgets: BudgetsConfig = BudgetsConfig()
    pricing_overrides: dict[str, tuple[float, float]] = {}
    report: ReportConfig = ReportConfig()


def load_config(path: Path | str = "graphtrace.yaml") -> GraphTraceConfig:
    """All-defaults if `path` doesn't exist -- a missing config file is normal,
    not a required setup step before anything else works."""
    path = Path(path)
    if not path.exists():
        return GraphTraceConfig()
    data = _yaml.load(path.read_text(encoding="utf-8")) or {}
    return GraphTraceConfig(**data)
