"""Strip volatile fields out of a request before it's hashed for cassette matching.

Lives in `core/`, not `replay/`, on purpose: normalization is pure string/regex
logic with no cassette-format or LangGraph knowledge at all -- keeping it here
means a future importer (Week 3, trace-to-test) or a different framework's
interceptor (v0.4) can reuse the exact same rules without depending on `replay/`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The two volatile-field shapes real requests actually contain: timestamps and
# UUIDs. Request ids are typically just a UUID too, so no separate rule for them.
DEFAULT_RULES: tuple[tuple[str, str], ...] = (
    (r"\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:\d{2})?", "<ts>"),
    (r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "<uuid>"),
)


@dataclass
class NormalizeRule:
    pattern: str
    replace: str


def normalize(text: str, rules: tuple[tuple[str, str], ...] | list[NormalizeRule] = DEFAULT_RULES) -> str:
    """Replace every volatile substring `text` matches with its rule's placeholder.

    Takes a plain string, not a request object, so it works identically whether
    the caller already serialized the request to text for hashing (replay/matcher.py)
    or has some other string to clean up -- this function doesn't know or care what
    "a request" is, only how to scrub text. Accepts either plain (pattern, replace)
    tuples (the built-in defaults) or `NormalizeRule` objects (what a project's own
    `graphtrace.yaml` config, once config.py exists, deserializes into) uniformly.
    """
    for rule in rules:
        pattern, replace = (rule.pattern, rule.replace) if isinstance(rule, NormalizeRule) else rule
        text = re.sub(pattern, replace, text)
    return text
