"""
lukezoom.core.tokens — Token estimation utilities.

Kept in its own module so callers that only need token math
don't have to import the full types module.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses the ~4-characters-per-token heuristic which is conservative
    for English prose and slightly generous for code / structured data.
    Always returns at least 1.
    """
    return max(1, len(text) // 4)


def estimate_tokens_messages(messages: list[dict]) -> int:
    """Estimate total tokens across a list of message dicts.

    Each message is expected to have at least a ``"content"`` key.
    Adds a small per-message overhead (4 tokens) to account for
    role / name / separator tokens.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += estimate_tokens(content) + 4  # per-message overhead
    return total


def fits_budget(text: str, budget: int) -> bool:
    """Return True if *text* fits within *budget* tokens."""
    return estimate_tokens(text) <= budget


def trim_to_budget(text: str, budget: int, suffix: str = "...") -> str:
    """Trim *text* so its estimated token count fits *budget*.

    Trims at character boundaries (not token boundaries) and appends
    *suffix* to indicate truncation.  Returns the original text
    unchanged if it already fits.
    """
    if fits_budget(text, budget):
        return text

    # target char count: budget * 4 minus room for suffix
    target_chars = max(0, budget * 4 - len(suffix))
    return text[:target_chars] + suffix
