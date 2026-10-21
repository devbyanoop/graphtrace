"""Flexible value matchers for assert_tool_called(..., args_match=...).

Tool args often contain values a test shouldn't pin down exactly -- a generated
request id, "some non-empty string", "any positive number". `args_match` accepts
three kinds of values for a given key, and _matches() in assertions.py is what
tells them apart:

    args_match={"order_id": "1234"}            # exact equality
    args_match={"amount": lambda v: v > 0}      # a plain callable predicate
    args_match={"request_id": any_str()}        # a Matcher object, from this module

Only one matcher is built for v0.1 -- any_str() -- because it's the only one the
spec actually names. Add more here (any_int(), any_value(), etc.) the day a real
test needs one; a speculative matcher nobody's tests exercise is just surface area
to maintain.
"""

from __future__ import annotations

from typing import Any


class Matcher:
    """Base class for matcher objects. Subclasses implement `matches`."""

    def matches(self, actual: Any) -> bool:
        raise NotImplementedError


class _AnyStr(Matcher):
    def matches(self, actual: Any) -> bool:
        return isinstance(actual, str)

    def __repr__(self) -> str:
        return "any_str()"


def any_str() -> Matcher:
    """Matches any string value, regardless of content."""
    return _AnyStr()
