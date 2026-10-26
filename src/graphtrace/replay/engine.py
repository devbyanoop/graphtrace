"""The generic record/replay decision tree -- independent of what a "request" or
"response" actually are. A framework interceptor (today: frameworks/langgraph/) is
responsible for:

  - handing this a `request` that's already YAML-safe (a plain dict/str/list/number),
  - wrapping its own real call so `real_call()` returns an already-YAML-safe result,
  - and turning whatever this returns back into its own framework's real object
    (an AIMessage, say) on the way back out.

None of that conversion happens here, on purpose -- it's exactly the LangChain-
specific part that keeps this module reusable for a future CrewAI/AutoGen
interceptor (v0.4) with zero changes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from graphtrace.replay.cassette import to_plain_python
from graphtrace.replay.context import ReplayContext
from graphtrace.replay.matcher import CassetteMismatchError


def resolve_call(
    ctx: ReplayContext,
    node: str,
    kind: str,
    request: Any,
    real_call: Callable[[], Any],
    extra: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    """Returns (result, latency_s) for one call under the active mode.

    Raises rather than ever returning a "fake success": `CassetteMismatchError`
    on an unmatched `replay` call, a plain `RuntimeError` carrying the original
    error message on a replayed *failure* (the original exception's exact type
    is genuinely not recoverable once serialized to YAML -- an honest, documented
    limitation, not a bug), or whatever `real_call` itself raises, unmodified,
    on a live/record/record_missing-miss call -- preserving its exact type so a
    node's own `except SomeSpecificError:` still works the same in every mode.
    """
    if ctx.mode == "live":
        start = time.perf_counter()
        result = real_call()
        return result, time.perf_counter() - start

    key = ctx.matcher.next_key(node, kind, request)

    if ctx.mode in ("replay", "record_missing"):
        entry = ctx.cassette.find(key)
        if entry is None:
            # Positional fallback, for cassettes codegen/test_writer.py builds from an
            # *imported* trace. Those entries can never satisfy the exact hash match
            # above: the request field recorded there is whatever shape the original
            # observability platform exported (e.g. Langfuse's own JSON "input"), not
            # the real `str(list[BaseMessage])` the live interceptor hashes at replay
            # time -- there is no transformation that reconciles the two in general.
            # codegen marks its entries with the literal hash segment "any" (never a
            # real value: hash_request always returns 12 hex characters) to opt into
            # this looser, position-only match -- same node/kind/occurrence, content
            # ignored. Entries written by the normal record path keep the real hash,
            # so this fallback can never accidentally match one of those instead of
            # correctly raising CassetteMismatchError on a genuine miss.
            occurrence = key.rsplit("/", 1)[-1]
            entry = ctx.cassette.find(f"{node}/{kind}/any/{occurrence}")
        if entry is not None:
            if entry.get("error") is not None:
                raise RuntimeError(to_plain_python(entry["error"]))
            return to_plain_python(entry.get("response")), to_plain_python(entry.get("latency_s", 0.0))
        if ctx.mode == "replay":
            raise CassetteMismatchError(
                f"no cassette entry for {key!r} (node={node!r}, kind={kind!r}); "
                f"available keys: {[e.get('key') for e in ctx.cassette.entries]!r}"
            )
        # record_missing, no match -> record a new one below, same as "record".

    start = time.perf_counter()
    try:
        result = real_call()
    except Exception as exc:
        latency_s = time.perf_counter() - start
        entry = {"key": key, "kind": kind, "node": node, "request": request, "latency_s": latency_s, "error": str(exc), "response": None}
        entry.update(extra or {})
        ctx.new_entries.append(entry)
        raise
    latency_s = time.perf_counter() - start
    entry = {"key": key, "kind": kind, "node": node, "request": request, "latency_s": latency_s, "response": result}
    entry.update(extra or {})
    ctx.new_entries.append(entry)
    return result, latency_s
