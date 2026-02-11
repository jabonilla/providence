"""Token estimation utilities for CONTEXT-SVC.

Provides rough token count estimates for enforcing context window
token budgets. Uses a simple heuristic (len/4 for English text)
as a fast approximation.

Spec Reference: Technical Spec v2.3, Section 4.6
"""

import json
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses the rough heuristic of ~4 characters per token for English text.
    This is intentionally simple and fast — exact tokenization would
    require loading a model-specific tokenizer.

    Args:
        text: Input text string.

    Returns:
        Estimated token count (always >= 0).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_fragment_tokens(payload: dict[str, Any]) -> int:
    """Estimate token count for a MarketStateFragment payload.

    Serializes the payload to JSON and estimates tokens from the
    resulting string length.

    Args:
        payload: Fragment payload dictionary.

    Returns:
        Estimated token count.
    """
    serialized = json.dumps(payload, default=str)
    return estimate_tokens(serialized)
