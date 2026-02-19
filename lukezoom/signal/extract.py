"""
lukezoom.signal.extract — LLM-based semantic extraction.

Detects novel information in conversation exchanges and returns
structured updates for relationship, preference, trust, and skill
memory stores.

Requires an llm_func callback: ``llm_func(prompt: str, system: str) -> str``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from lukezoom.core.types import LLMFunc

log = logging.getLogger(__name__)


# Prompts

EXTRACTION_SYSTEM = """Analyze this conversation exchange and extract any NEW information that should be remembered permanently.

Return ONLY a JSON object:
{
  "relationship_updates": [{"person": "name", "fact": "what was learned", "section": "What I Know"}],
  "preference_updates": [{"item": "thing", "type": "like|dislike|uncertainty", "reason": "why"}],
  "trust_changes": [{"person": "name", "direction": "up|down", "reason": "why"}],
  "skills_learned": [{"skill": "name", "content": "description"}],
  "nothing_new": true
}

If nothing new was learned, return {"nothing_new": true}.
Only extract genuinely NEW information, not things already known."""

EXTRACTION_USER_TEMPLATE = """Current knowledge about {person}:
{existing_knowledge}

Exchange:
{person}: {their_message}
Self: {response}"""


# Parsing


def parse_extraction(llm_response: str) -> Dict:
    """
    Parse a JSON extraction response from the LLM.

    Handles raw JSON, markdown-fenced JSON, and malformed responses.
    Returns a dict with the extraction fields, or an empty-ish dict
    on failure (``{"nothing_new": True}``).
    """
    if not llm_response:
        return {"nothing_new": True}

    text = llm_response.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.S)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to find first { ... } block (greedy to catch nested)
        brace_match = re.search(r"\{.*\}", text, re.S)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                return {"nothing_new": True}
        else:
            return {"nothing_new": True}

    if not isinstance(data, dict):
        return {"nothing_new": True}

    # Normalise: ensure expected keys exist with correct types
    result: Dict = {
        "nothing_new": bool(data.get("nothing_new", False)),
        "relationship_updates": _ensure_list(data.get("relationship_updates")),
        "preference_updates": _ensure_list(data.get("preference_updates")),
        "trust_changes": _ensure_list(data.get("trust_changes")),
        "skills_learned": _ensure_list(data.get("skills_learned")),
    }

    # If all update lists are empty, mark as nothing_new
    has_updates = any(
        result[k]
        for k in (
            "relationship_updates",
            "preference_updates",
            "trust_changes",
            "skills_learned",
        )
    )
    if not has_updates:
        result["nothing_new"] = True

    return result


def _ensure_list(val) -> List[Dict]:
    """Coerce a value to a list of dicts, dropping non-dict items."""
    if not isinstance(val, list):
        return []
    return [item for item in val if isinstance(item, dict)]


# Extraction entry point


def extract(
    person: str,
    their_message: str,
    response: str,
    existing_knowledge: str,
    llm_func: LLMFunc,
) -> Dict:
    """
    Call an LLM to extract novel information from a conversation exchange.

    Parameters
    ----------
    person : str
        Name of the conversation partner.
    their_message : str
        What they said.
    response : str
        What we replied.
    existing_knowledge : str
        Current known facts about this person (to avoid re-extracting).
    llm_func : LLMFunc
        ``llm_func(prompt: str, system: str) -> str`` — sends a user
        prompt and system prompt to an LLM and returns the raw response.

    Returns
    -------
    dict
        Structured extraction with keys: relationship_updates,
        preference_updates, trust_changes, skills_learned, nothing_new.
    """
    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        person=person,
        existing_knowledge=existing_knowledge or "(none yet)",
        their_message=their_message,
        response=response,
    )

    try:
        raw = llm_func(user_prompt, EXTRACTION_SYSTEM)
        return parse_extraction(raw)
    except Exception as exc:
        log.debug("LLM extraction failed: %s", exc)
        return {"nothing_new": True}
