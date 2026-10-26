"""Rubric file schema -- the pass/fail criteria a judge grades a response
against, per Plan/graphtrace.md section 9.

Pydantic, strict, same pattern as config.py -- a rubric is a small, user-authored
YAML file, and a typo in it should raise a clear error naming the actual field,
not silently do nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

_yaml = YAML(typ="safe")


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: int
    criteria: list[Criterion] = Field(min_length=1)
    # "1-5" loads and validates (so a project can write one ahead of time), but
    # judge/client.py's grade() only implements pass/fail grading today -- a
    # 1-5 rubric raises a clear NotImplementedError there, not a silent wrong
    # pass/fail guess.
    scale: Literal["pass_fail", "1-5"] = "pass_fail"
    pass_threshold: str = "all"


def load_rubric(path: Path | str) -> Rubric:
    path = Path(path)
    data = _yaml.load(path.read_text(encoding="utf-8")) or {}
    return Rubric(**data)
