"""The one exception type every assert_* function raises, plus its diff formatting."""

from __future__ import annotations

from typing import Any


def _first_mismatch(expected: list[Any], actual: list[Any]) -> str:
    """Line up two lists index by index and mark where they first diverge.

    Used by assert_path. Printing `expected != actual` as two repr'd lists forces you
    to eyeball-diff them yourself; walking index by index and marking the row where
    they stop agreeing is what makes the failure readable at a glance.
    """
    lines = []
    for i in range(max(len(expected), len(actual))):
        e = expected[i] if i < len(expected) else "<missing>"
        a = actual[i] if i < len(actual) else "<missing>"
        marker = "  " if e == a else "->"
        lines.append(f"{marker} [{i}] expected={e!r} actual={a!r}")
    return "\n".join(lines)


class TrajectoryAssertionError(AssertionError):
    """Raised by every assert_* function when a trajectory doesn't match expectations.

    This subclasses AssertionError, not Exception. Two reasons that matters:

    1. pytest already special-cases AssertionError in its output and in
       `pytest.raises(AssertionError)` catches, so a bare `assert` and an
       `assert_path(...)` call fail the same way from pytest's point of view.
    2. pytest's assertion-rewrite machinery only instruments literal `assert`
       statements in test files -- it can't rewrite a call to a function graphtrace
       defines. So instead of relying on that rewrite for a useful failure message,
       we build the message ourselves (see _first_mismatch above) and hand it to the
       base class's constructor, which is what actually shows up in the pytest report.

    `expected` / `actual` are kept as plain attributes (not just baked into the message
    string) so a test runner, IDE, or future graphtrace report renderer can pull the
    structured values back out instead of re-parsing the message text.
    """

    def __init__(self, message: str, *, expected: Any = None, actual: Any = None) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(message)
