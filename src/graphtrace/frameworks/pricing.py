"""Token -> USD pricing, consulted once per LLMCall at record time.

This is what keeps Trajectory.cost_usd a plain sum instead of a lookup (see
core/trajectory.py's module docstring) -- by the time an LLMCall exists, its cost
is already a plain float; core never has to know a pricing table exists at all.
"""

from __future__ import annotations

# USD per million tokens: (input_price, output_price). Small and deliberately
# conservative -- a stale price here is a worse failure mode caught silently than
# an unrecognized model falling back to $0.00 (see cost_for_call below), so this
# table is kept short rather than padded with guesses. Update as real prices
# change; nothing here is guaranteed current -- `overrides` exists for exactly
# that reason.
_PRICE_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}


def cost_for_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    overrides: dict[str, tuple[float, float]] | None = None,
) -> float:
    """USD cost of one LLM call, given its prompt/completion token counts.

    Returns 0.0 for a model this table doesn't know -- a clear, visible "add this
    model to the table (or pass it via overrides)" signal, never a guessed number
    presented as real. `overrides` wins over the built-in table for any model name
    present in both, so a project can correct a stale price without waiting for
    this file to be updated upstream.
    """
    table = {**_PRICE_PER_MILLION_TOKENS, **(overrides or {})}
    prices = table.get(model)
    if prices is None:
        return 0.0
    input_price_per_million, output_price_per_million = prices
    return (prompt_tokens * input_price_per_million + completion_tokens * output_price_per_million) / 1_000_000
