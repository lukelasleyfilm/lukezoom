"""
lukezoom.working.context — Working memory context builder.

Assembles the context window that gets injected before every LLM call.
Allocates a fixed token budget across identity, relationship, grounding
context, recent conversation, episodic traces, and procedural skills —
then formats everything into a single prompt with clear section headers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from lukezoom.core.tokens import estimate_tokens
from lukezoom.core.types import Context, HealthBitmap
from lukezoom.working.allocator import compress_text, fit_messages, knapsack_allocate

if TYPE_CHECKING:
    from lukezoom.core.config import Config


class ContextBuilder:
    """Builds a ``Context`` object ready for system-prompt injection.

    Token budget shares (fractions of the total budget):

    =============================  =======
    Section                        Share
    =============================  =======
    Identity (SOUL.md)             0.16
    Relationship                   0.12
    Grounding (prefs, bounds, ...)  0.10
    Recent conversation            0.22
    Episodic traces                0.16
    Procedural skills              0.06
    Reserve (breathing room)       0.18
    =============================  =======

    Shares can be overridden by passing a ``Config`` object.
    """

    def __init__(
        self,
        token_budget: int = 6000,
        config: Optional[Config] = None,
    ) -> None:
        self.token_budget = token_budget

        # Allocation shares — pull from config if provided, else defaults.
        self.identity_share: float = getattr(config, "identity_share", 0.16)
        self.relationship_share: float = getattr(config, "relationship_share", 0.12)
        self.grounding_share: float = getattr(config, "grounding_share", 0.10)
        self.recent_conversation_share: float = getattr(
            config, "recent_conversation_share", 0.22
        )
        self.episodic_share: float = getattr(config, "episodic_share", 0.16)
        self.procedural_share: float = getattr(config, "procedural_share", 0.06)
        self.reserve_share: float = getattr(config, "reserve_share", 0.18)

    # Build

    def build(
        self,
        person: str,
        message: str,
        identity_text: str = "",
        relationship_text: str = "",
        grounding_context: str = "",
        recent_messages: Optional[List[Dict]] = None,
        salient_traces: Optional[List[Dict]] = None,
        relevant_skills: Optional[List[str]] = None,
        correction_prompt: Optional[str] = None,
        health: Optional[HealthBitmap] = None,
    ) -> Context:
        """Assemble a full context window within the token budget.

        Parameters
        ----------
        person:
            Canonical name of the person we're talking to.
        message:
            The incoming message (used for context, not included in output).
        identity_text:
            Full SOUL.md text (will be compressed if needed).
        relationship_text:
            Relationship file content for *person*.
        grounding_context:
            Pre-assembled grounding context: trust tier, preferences,
            boundaries, contradictions, active injuries, recent journal.
            Always included regardless of whether a relationship file
            exists.
        recent_messages:
            List of message dicts (must have ``content``, ``speaker``).
        salient_traces:
            List of trace dicts (must have ``content``, ``salience``).
        relevant_skills:
            List of skill content strings.
        correction_prompt:
            If the last response drifted, prepend this correction.

        Returns
        -------
        Context
            Ready-to-inject context with metadata.
        """
        recent_messages = recent_messages or []
        salient_traces = salient_traces or []
        relevant_skills = relevant_skills or []

        available_budget = int(self.token_budget * (1 - self.reserve_share))
        tokens_used = 0
        trace_ids: List[str] = []
        memories_loaded = 0
        sections: List[str] = []

        # -- 1. Correction prompt (highest priority, taken from reserve) ----
        if correction_prompt:
            correction_tokens = estimate_tokens(correction_prompt)
            sections.append(f"--- CORRECTION ---\n{correction_prompt}")
            tokens_used += correction_tokens

        # -- 2. Identity (SOUL.md) -----------------------------------------
        identity_budget = int(self.token_budget * self.identity_share)
        if identity_text:
            identity_compressed = compress_text(identity_text, identity_budget)
            identity_tokens = estimate_tokens(identity_compressed)
            if tokens_used + identity_tokens <= available_budget:
                sections.append(f"--- IDENTITY ---\n{identity_compressed}")
                tokens_used += identity_tokens
            else:
                # Force-fit a smaller version
                remaining = available_budget - tokens_used
                if remaining > 50:
                    identity_compressed = compress_text(identity_text, remaining)
                    sections.append(f"--- IDENTITY ---\n{identity_compressed}")
                    tokens_used += estimate_tokens(identity_compressed)

        # -- 3. Relationship context ---------------------------------------
        relationship_budget = int(self.token_budget * self.relationship_share)
        if relationship_text and person:
            rel_compressed = compress_text(relationship_text, relationship_budget)
            rel_tokens = estimate_tokens(rel_compressed)
            if tokens_used + rel_tokens <= available_budget:
                sections.append(f"--- RELATIONSHIP: {person} ---\n{rel_compressed}")
                tokens_used += rel_tokens

        # -- 4. Grounding context (always included) ------------------------
        grounding_budget = int(self.token_budget * self.grounding_share)
        grounding_budget = min(grounding_budget, available_budget - tokens_used)
        if grounding_context and grounding_budget > 0:
            grounding_compressed = compress_text(grounding_context, grounding_budget)
            grounding_tokens = estimate_tokens(grounding_compressed)
            if tokens_used + grounding_tokens <= available_budget:
                sections.append(f"--- GROUNDING ---\n{grounding_compressed}")
                tokens_used += grounding_tokens

        # -- 5. Recent conversation ----------------------------------------
        conversation_budget = int(self.token_budget * self.recent_conversation_share)
        conversation_budget = min(conversation_budget, available_budget - tokens_used)
        if recent_messages and conversation_budget > 0:
            fitted = fit_messages(recent_messages, conversation_budget)
            if fitted:
                lines: List[str] = []
                for msg in fitted:
                    speaker = msg.get("speaker", "?")
                    content = msg.get("content", "")
                    lines.append(f"[{speaker}]: {content}")
                conv_text = "\n".join(lines)
                sections.append(f"--- RECENT CONVERSATION ---\n{conv_text}")
                tokens_used += estimate_tokens(conv_text)
                memories_loaded += len(fitted)

        # -- 6. High-salience episodic traces (greedy knapsack) ------------
        episodic_budget = int(self.token_budget * self.episodic_share)
        episodic_budget = min(episodic_budget, available_budget - tokens_used)
        if salient_traces and episodic_budget > 0:
            selected, ep_tokens = knapsack_allocate(
                salient_traces,
                episodic_budget,
                key_field="salience",
                text_field="content",
            )
            if selected:
                trace_lines: List[str] = []
                for idx, trace in enumerate(selected, 1):
                    tid = trace.get("id", "")
                    if tid:
                        trace_ids.append(tid)
                    kind = trace.get("kind", "episode")
                    content = trace.get("content", "")
                    sal = trace.get("salience", 0.0)
                    # Citation-ready format: numbered with trace ID
                    cite_id = f" {{trace:{tid}}}" if tid else ""
                    trace_lines.append(
                        f"[{idx}] [{kind} | sal:{sal:.2f}] {content}{cite_id}"
                    )
                traces_text = "\n".join(trace_lines)
                # Add citation instructions (NotebookLM-inspired)
                citation_hint = (
                    "\nWhen your response draws on these memories, "
                    "cite them as [1], [2], etc."
                )
                sections.append(
                    f"--- RELEVANT MEMORIES ---\n{traces_text}{citation_hint}"
                )
                tokens_used += ep_tokens
                memories_loaded += len(selected)

        # -- 7. Procedural skills ------------------------------------------
        procedural_budget = int(self.token_budget * self.procedural_share)
        procedural_budget = min(procedural_budget, available_budget - tokens_used)
        if relevant_skills and procedural_budget > 0:
            skill_texts: List[str] = []
            skill_tokens_used = 0
            for skill_content in relevant_skills:
                stokens = estimate_tokens(skill_content)
                if skill_tokens_used + stokens > procedural_budget:
                    # Compress what we can fit
                    remaining = procedural_budget - skill_tokens_used
                    if remaining > 20:
                        skill_texts.append(compress_text(skill_content, remaining))
                    break
                skill_texts.append(skill_content)
                skill_tokens_used += stokens
            if skill_texts:
                skills_text = "\n---\n".join(skill_texts)
                sections.append(f"--- RELEVANT SKILLS ---\n{skills_text}")
                tokens_used += skill_tokens_used
                memories_loaded += len(skill_texts)

        # -- Assemble final context text -----------------------------------
        context_text = "\n\n".join(sections)
        tokens_used = estimate_tokens(context_text)  # recount on final text

        return Context(
            text=context_text,
            trace_ids=trace_ids,
            person=person,
            tokens_used=tokens_used,
            token_budget=self.token_budget,
            memories_loaded=memories_loaded,
            health=health,
        )
