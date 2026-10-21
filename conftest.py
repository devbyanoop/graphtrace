"""Repo-root conftest: load .env (if present) before any test runs.

Same pattern as kbguard's own conftest.py. Nothing in Week 1 actually reads an
environment variable yet (see .env.example) -- this exists now so the day a test
does need one (the judge contract, Week 3), it's picked up the same way regardless
of which shell or IDE run configuration invoked pytest, with no per-test setup.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Pointed at this file's own directory explicitly, rather than relying on
# load_dotenv()'s default upward-search from the CWD -- deterministic regardless
# of which directory pytest is actually invoked from.
load_dotenv(Path(__file__).parent / ".env")
