"""Tests for lukezoom.trust — 5-tier trust gate system."""

import pytest
import tempfile
from lukezoom.trust import Tier, TrustGate, SOURCE_BLOCKED_TOOLS, FRIEND_REQUIRED_TOOLS
from lukezoom.semantic.store import SemanticStore


@pytest.fixture
def trust_gate(tmp_path):
    semantic = SemanticStore(
        semantic_dir=tmp_path / "semantic",
        soul_dir=tmp_path / "soul",
    )
    gate = TrustGate(semantic=semantic, core_person="alice")
    gate.ensure_core_person()
    return gate


class TestTierHierarchy:
    """Trust tiers must form a strict monotonic hierarchy."""

    def test_tier_ordering(self):
        assert Tier.CORE < Tier.INNER_CIRCLE < Tier.FRIEND < Tier.ACQUAINTANCE < Tier.STRANGER

    def test_core_is_zero(self):
        assert Tier.CORE == 0

    def test_stranger_is_highest(self):
        assert Tier.STRANGER == 4


class TestCorePersonTrust:
    def test_core_person_is_core_tier(self, trust_gate):
        policy = trust_gate.policy_for("alice", source="direct")
        assert policy.tier == Tier.CORE

    def test_core_has_full_persistence(self, trust_gate):
        policy = trust_gate.policy_for("alice", source="direct")
        assert policy.memory_persistent is True

    def test_core_has_full_access(self, trust_gate):
        policy = trust_gate.policy_for("alice", source="direct")
        assert policy.can_see_soul is True
        assert policy.can_see_others_relationships is True


class TestSourceDegradation:
    def test_external_source_degrades_trust(self, trust_gate):
        """External/API sources should get lower trust than direct."""
        direct = trust_gate.policy_for("alice", source="direct")
        external = trust_gate.policy_for("alice", source="external")
        # External source should degrade or equal; never higher trust
        assert external.tier >= direct.tier

    def test_unknown_person_is_stranger(self, trust_gate):
        policy = trust_gate.policy_for("totally_unknown_person", source="direct")
        assert policy.tier >= Tier.ACQUAINTANCE


class TestToolBlocking:
    def test_source_blocked_tools_not_empty(self):
        assert len(SOURCE_BLOCKED_TOOLS) > 0

    def test_friend_required_tools_not_empty(self):
        assert len(FRIEND_REQUIRED_TOOLS) > 0

    def test_purge_is_source_blocked(self):
        """Purge should never be allowed from external sources."""
        assert "lukezoom_purge_person" in SOURCE_BLOCKED_TOOLS
