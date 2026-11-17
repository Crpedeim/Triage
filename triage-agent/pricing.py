"""
pricing.py — LLM prices used for cost reporting (eval runner and tracing).

USD per 1M tokens. Only prices with a cited source belong here. A model that is
missing has an UNKNOWN cost: callers must report None/unknown, never 0.

gemini-3.1-pro-preview: $2 input / $12 output (prompts <= 200k tokens), as quoted
in search results for https://ai.google.dev/gemini-api/docs/pricing
(looked up 2026-11-16; the page itself could not be opened from the build
environment). No published price was found for gemini-3.8-flash.

Whether Gemini bills the "reasoning" tokens (total - input - output) is not
verified, so callers report cost both ways where it matters.

Override or extend at runtime with LLM_PRICES_JSON, e.g.
  LLM_PRICES_JSON='{"gemini-3.8-flash": {"input": 0.5, "output": 3.0}}'
"""

from __future__ import annotations

import json
import os

PRICES_PER_M: dict[str, dict[str, float]] = {
    "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0},
}

_extra = os.getenv("LLM_PRICES_JSON")
if _extra:
    PRICES_PER_M.update(json.loads(_extra))


def call_cost(model: str | None, input_tokens: int, output_tokens: int,
              reasoning_tokens: int = 0, bill_reasoning: bool = True) -> float | None:
    """Cost of one call in USD, or None when the model has no known price."""
    p = PRICES_PER_M.get(model or "")
    if not p:
        return None
    out = output_tokens + (reasoning_tokens if bill_reasoning else 0)
    return (input_tokens * p["input"] + out * p["output"]) / 1e6
