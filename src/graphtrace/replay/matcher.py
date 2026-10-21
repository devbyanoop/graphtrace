"""The cassette request-matching key: `node/kind/hash(normalized request)/occurrence`.

"occurrence" is what lets the same node+request pair appear more than once in one
run (a retry loop re-calling the same tool with the same args) and still match its
own separate recorded entry, in order, rather than every repeat matching the first
one recorded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from graphtrace.core.normalize import DEFAULT_RULES, normalize


class CassetteMismatchError(RuntimeError):
    """Raised in `replay` mode when a call has no matching cassette entry."""


def hash_request(request: Any, rules: tuple | list = DEFAULT_RULES) -> str:
    """A short, stable id for a request, after normalizing away volatile fields.

    Truncated to 12 hex characters (48 bits) -- long enough that an accidental
    collision between two genuinely different requests in one test suite is not
    a realistic concern, short enough to keep cassette keys readable, matching
    the architecture plan's own example (`router/llm/7f3a.../0`).
    """
    text = normalize(str(request), rules)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class Matcher:
    """Hands out the next cassette key for a given (node, kind, request), tracking
    how many times that exact combination has already been seen *in this run* so
    repeats get distinct, ordered keys instead of all colliding on the first one.
    """

    rules: tuple | list = DEFAULT_RULES
    _occurrence_counts: dict[str, int] = field(default_factory=dict)

    def next_key(self, node: str, kind: str, request: Any) -> str:
        base = f"{node}/{kind}/{hash_request(request, self.rules)}"
        occurrence = self._occurrence_counts.get(base, 0)
        self._occurrence_counts[base] = occurrence + 1
        return f"{base}/{occurrence}"
