"""Exceptions the judge contract raises."""

from __future__ import annotations


class JudgeError(RuntimeError):
    """Raised when the judge's response can't be trusted -- malformed JSON after
    a retry, or any other real failure. Never swallowed into a silent `None`:
    a test that depends on a judge verdict should fail loudly if the judge
    itself is broken, not quietly treat "no verdict" as "verdict: fail" or
    "verdict: pass".
    """
