"""
lukezoom.working.allocator — Token budget allocator.

Greedy knapsack for fitting memories into a token budget,
text compression, and message fitting utilities.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from lukezoom.core.tokens import estimate_tokens


def knapsack_allocate(
    items: List[Dict],
    budget: int,
    key_field: str = "salience",
    text_field: str = "content",
) -> Tuple[List[Dict], int]:
    """Greedy knapsack allocation by density (key_field / tokens).

    Returns ``(selected_items, tokens_used)``.
    Items are selected highest-density-first until the budget is
    exhausted.  Items missing the required fields are skipped.
    """
    if not items or budget <= 0:
        return [], 0

    # Score each item by density = salience / tokens
    scored: List[Tuple[float, int, Dict]] = []
    for item in items:
        salience = item.get(key_field, 0.0)
        text = item.get(text_field, "")
        tokens = estimate_tokens(text)
        if tokens <= 0:
            continue
        density = salience / tokens
        scored.append((density, tokens, item))

    # Sort descending by density
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict] = []
    tokens_used = 0

    for _density, tokens, item in scored:
        if tokens_used + tokens > budget:
            continue  # skip items that don't fit (greedy, not optimal)
        selected.append(item)
        tokens_used += tokens

    return selected, tokens_used


def compress_text(text: str, max_tokens: int) -> str:
    """Truncate *text* to fit within *max_tokens*.

    Attempts to cut at a sentence boundary (period, exclamation, or
    question mark followed by whitespace).  Appends ``[...truncated]``
    when text is actually cut.
    """
    if not text:
        return text

    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text

    suffix = " [...truncated]"
    # Target char count minus room for the suffix
    target_chars = max(0, max_tokens * 4 - len(suffix))
    truncated = text[:target_chars]

    # Try to cut at last sentence boundary
    for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        last_idx = truncated.rfind(sep)
        if last_idx > len(truncated) // 2:  # don't cut too aggressively
            truncated = truncated[: last_idx + 1]
            break

    return truncated + suffix


def fit_messages(messages: List[Dict], budget: int) -> List[Dict]:
    """Fit the most recent messages within a token budget.

    Works backward from the most recent message, accumulating
    until the budget is exhausted.  Returns messages in
    chronological order.
    """
    if not messages or budget <= 0:
        return []

    selected: List[Dict] = []
    tokens_used = 0

    # Walk backwards (most recent first)
    for msg in reversed(messages):
        content = msg.get("content", "")
        tokens = estimate_tokens(content) + 4  # per-message overhead
        if tokens_used + tokens > budget:
            break
        selected.append(msg)
        tokens_used += tokens

    # Restore chronological order
    selected.reverse()
    return selected
