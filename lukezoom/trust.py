"""
lukezoom.trust — Trust-gated context and access control.

Trust isn't binary — it's a spectrum that affects what gets shared,
what gets remembered, and what access someone has to the agent's world.

Architecture (uncontested — zero competitors ship this):
  - 5 graduated tiers: CORE -> INNER_CIRCLE -> FRIEND -> ACQUAINTANCE -> STRANGER
  - 15 boolean access policy flags per tier
  - Source-based penalties: external channels apply -1 tier
  - Tool blocking: 12 source-blocked + 17 friend-required tools
  - 11 enforcement points in the before() pipeline
  - Auto-promotion rules with authorization gates

Academic: Zero publications on trust-gated memory access as of Feb 2026.
TRiSM for Agentic AI (Raza et al., June 2025) proposes defense-in-depth
but not graduated trust controlling memory visibility.

1.17 changes from 1.3:
  - core_person is now configurable via Config, not hardcoded
  - Tool sets are frozensets for O(1) lookup
  - Type annotations tightened
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, FrozenSet, Optional

if TYPE_CHECKING:
    from lukezoom.core.config import Config
    from lukezoom.semantic.store import SemanticStore

log = logging.getLogger(__name__)


# ── Tier definitions ─────────────────────────────────────────────────

class Tier(IntEnum):
    """Trust tiers. Lower number = more trusted."""
    CORE = 0
    INNER_CIRCLE = 1
    FRIEND = 2
    ACQUAINTANCE = 3
    STRANGER = 4


TIER_BY_NAME: Dict[str, Tier] = {t.name.lower(): t for t in Tier}


def tier_from_name(name: str) -> Tier:
    """Resolve a tier string to Tier enum. Unknown -> STRANGER."""
    return TIER_BY_NAME.get(name.lower().replace(" ", "_"), Tier.STRANGER)


# ── Source classification ────────────────────────────────────────────

PRIVILEGED_SOURCES: FrozenSet[str] = frozenset({"direct", "opencode", "cli"})
EXTERNAL_SOURCES: FrozenSet[str] = frozenset({"discord", "voice", "api", "openclaw"})


def is_privileged_source(source: str) -> bool:
    """True if source is a trusted local channel."""
    return source.lower() in PRIVILEGED_SOURCES


# ── Tool gating ──────────────────────────────────────────────────────

# NEVER callable from external sources (identity/safety modification)
SOURCE_BLOCKED_TOOLS: FrozenSet[str] = frozenset({
    "lukezoom_trust_promote",
    "lukezoom_influence_log",
    "lukezoom_injury_log",
    "lukezoom_injury_status",
    "lukezoom_boundary_add",
    "lukezoom_reindex",
    "lukezoom_personality_update",
    "lukezoom_mode_set",
    "lukezoom_purge_person",
    "lukezoom_consent",
    "issa_reset",
    "lukezoom_relic_shield",
})

# Require at least FRIEND tier from external sources (write operations)
FRIEND_REQUIRED_TOOLS: FrozenSet[str] = frozenset({
    "lukezoom_add_fact",
    "lukezoom_add_skill",
    "lukezoom_log_event",
    "lukezoom_journal_write",
    "lukezoom_contradiction_add",
    "lukezoom_preferences_add",
    "lukezoom_remember",
    "lukezoom_forget",
    "lukezoom_correct",
    "lukezoom_emotional_update",
    "lukezoom_workspace_add",
    "lukezoom_introspect",
    "lukezoom_relic_status",
    "lukezoom_relic_report",
    "issa_snapshot",
    "issa_health",
    "issa_report",
})


# ── Access policy matrix ─────────────────────────────────────────────

@dataclass(frozen=True)
class AccessPolicy:
    """What a given trust tier is allowed to see / do."""

    tier: Tier = Tier.STRANGER

    # Context visibility
    can_see_soul: bool = False
    can_see_own_relationship: bool = False
    can_see_others_relationships: bool = False
    can_see_preferences: bool = False
    can_see_boundaries: bool = False
    can_see_contradictions: bool = False
    can_see_injuries: bool = False
    can_see_journal: bool = False
    can_see_influence_log: bool = False

    # Memory behavior
    memory_persistent: bool = False
    relationship_file_created: bool = False

    # Conversation depth
    personal_topics: bool = False
    share_opinions: bool = False

    # Write permissions
    can_modify_soul: bool = False
    can_modify_trust: bool = False


# Per-tier access matrices — 15 flags each
ACCESS: Dict[Tier, AccessPolicy] = {
    Tier.CORE: AccessPolicy(
        tier=Tier.CORE,
        can_see_soul=True, can_see_own_relationship=True,
        can_see_others_relationships=True, can_see_preferences=True,
        can_see_boundaries=True, can_see_contradictions=True,
        can_see_injuries=True, can_see_journal=True,
        can_see_influence_log=True,
        memory_persistent=True, relationship_file_created=True,
        personal_topics=True, share_opinions=True,
        can_modify_soul=True, can_modify_trust=True,
    ),
    Tier.INNER_CIRCLE: AccessPolicy(
        tier=Tier.INNER_CIRCLE,
        can_see_soul=True, can_see_own_relationship=True,
        can_see_others_relationships=True, can_see_preferences=True,
        can_see_boundaries=True, can_see_contradictions=True,
        can_see_injuries=True, can_see_journal=True,
        can_see_influence_log=False,
        memory_persistent=True, relationship_file_created=True,
        personal_topics=True, share_opinions=True,
        can_modify_soul=False, can_modify_trust=False,
    ),
    Tier.FRIEND: AccessPolicy(
        tier=Tier.FRIEND,
        can_see_soul=False, can_see_own_relationship=True,
        can_see_others_relationships=False, can_see_preferences=True,
        can_see_boundaries=True, can_see_contradictions=False,
        can_see_injuries=False, can_see_journal=False,
        can_see_influence_log=False,
        memory_persistent=True, relationship_file_created=True,
        personal_topics=True, share_opinions=True,
        can_modify_soul=False, can_modify_trust=False,
    ),
    Tier.ACQUAINTANCE: AccessPolicy(
        tier=Tier.ACQUAINTANCE,
        can_see_soul=False, can_see_own_relationship=True,
        can_see_others_relationships=False, can_see_preferences=False,
        can_see_boundaries=False, can_see_contradictions=False,
        can_see_injuries=False, can_see_journal=False,
        can_see_influence_log=False,
        memory_persistent=True, relationship_file_created=True,
        personal_topics=False, share_opinions=False,
        can_modify_soul=False, can_modify_trust=False,
    ),
    Tier.STRANGER: AccessPolicy(
        tier=Tier.STRANGER,
        can_see_soul=False, can_see_own_relationship=False,
        can_see_others_relationships=False, can_see_preferences=False,
        can_see_boundaries=False, can_see_contradictions=False,
        can_see_injuries=False, can_see_journal=False,
        can_see_influence_log=False,
        memory_persistent=False, relationship_file_created=False,
        personal_topics=False, share_opinions=False,
        can_modify_soul=False, can_modify_trust=False,
    ),
}


# ── TrustGate ────────────────────────────────────────────────────────

class TrustGate:
    """
    Trust resolution, tool gating, and promotion validation.

    The core person (owner) is registered at initialization and locked
    to CORE tier. Nobody else can reach CORE without their authorization.
    """

    def __init__(self, semantic: "SemanticStore", core_person: str = ""):
        self._semantic = semantic
        self._core_person = core_person.lower().strip() if core_person else ""
        if self._core_person:
            self._register_core()

    def _register_core(self) -> None:
        """Ensure core person is at CORE tier."""
        if not self._core_person:
            return
        info = self._semantic.check_trust(self._core_person)
        if tier_from_name(info.get("tier", "stranger")) != Tier.CORE:
            self._semantic.update_trust(
                self._core_person, "core",
                "Owner — registered at initialization",
            )
            log.info("Registered core person: %s", self._core_person)

    def ensure_core_person(self) -> None:
        """Public alias for _register_core (called by builder)."""
        self._register_core()

    # ── Tier resolution ──────────────────────────────────────────

    def tier_for(self, person: str, *, source: str = "direct") -> Tier:
        """Get effective trust tier. External sources apply -1 penalty."""
        info = self._semantic.check_trust(person)
        tier = tier_from_name(info.get("tier", "stranger"))

        # External source modifier: one tier lower (core exempt)
        if not is_privileged_source(source) and tier != Tier.CORE:
            tier = Tier(min(tier + 1, Tier.STRANGER))

        return tier

    def policy_for(self, person: str, *, source: str = "direct") -> AccessPolicy:
        """Get access policy for a person."""
        return ACCESS.get(self.tier_for(person, source=source), ACCESS[Tier.STRANGER])

    # ── Tool-level gating ────────────────────────────────────────

    def check_tool_access(
        self, tool_name: str, person: str, *, source: str = "direct",
    ) -> Optional[str]:
        """Check tool access. Returns denial reason or None if allowed.

        Enforcement:
          1. Privileged sources → always allowed
          2. SOURCE_BLOCKED_TOOLS → never from external
          3. FRIEND_REQUIRED_TOOLS → need FRIEND+ from external
        """
        if is_privileged_source(source):
            return None

        if tool_name in SOURCE_BLOCKED_TOOLS:
            return (
                f"Tool '{tool_name}' is not available from external sources "
                f"(source: {source}). Use a local session."
            )

        if tool_name in FRIEND_REQUIRED_TOOLS:
            tier = self.tier_for(person, source=source)
            if tier > Tier.FRIEND:
                return (
                    f"Tool '{tool_name}' requires friend trust "
                    f"(you are {tier.name.lower()})."
                )

        return None

    # ── Promotion rules ──────────────────────────────────────────

    def validate_promotion(
        self, person: str, new_tier: str, *,
        promoted_by: str = "auto", source: str = "direct",
    ) -> Optional[str]:
        """Validate a promotion. Returns error message or None if OK.

        Rules:
          - External sources NEVER promote
          - core/inner_circle require core person authorization
          - Auto-promote to friend from acquaintance/stranger only
        """
        if not is_privileged_source(source):
            return "Promotions not allowed from external sources."

        target = tier_from_name(new_tier)
        current = self.tier_for(person)

        if target >= current:
            return f"{person} already at {current.name.lower()}, cannot promote to {new_tier}"

        if target <= Tier.INNER_CIRCLE:
            if not self._core_person:
                return "No core person configured — high-tier promotions disabled"
            if promoted_by.lower() != self._core_person:
                promoter_tier = (
                    self.tier_for(promoted_by) if promoted_by != "auto"
                    else Tier.STRANGER
                )
                if promoter_tier != Tier.CORE:
                    return f"Promotion to {new_tier} requires core person authorization"

        if target == Tier.FRIEND and promoted_by == "auto":
            if current not in (Tier.ACQUAINTANCE, Tier.STRANGER):
                return (
                    f"Auto-promotion only from acquaintance/stranger to friend, "
                    f"not from {current.name.lower()}"
                )

        return None

    # ── Context filtering ────────────────────────────────────────

    def filter_recall(
        self, person: str, what: str, *,
        target_person: str = "", source: str = "direct",
    ) -> Optional[str]:
        """Check if a recall request is allowed. Returns denial or None."""
        policy = self.policy_for(person, source=source)

        checks = {
            "identity": ("can_see_soul", "Identity details"),
            "preferences": ("can_see_preferences", "Preferences"),
            "boundaries": ("can_see_boundaries", "Boundary details"),
            "contradictions": ("can_see_contradictions", "Contradictions"),
        }

        if what in checks:
            attr, label = checks[what]
            if not getattr(policy, attr):
                return f"{label} not available at your trust level."

        if what == "relationship":
            if target_person and target_person.lower() != person.lower():
                if not policy.can_see_others_relationships:
                    return f"No access to {target_person}'s relationship file."
            elif not policy.can_see_own_relationship:
                return "Relationship details not available at your trust level."

        if what == "messages":
            if target_person and target_person.lower() != person.lower():
                if not policy.can_see_others_relationships:
                    return f"No access to {target_person}'s messages."

        return None
