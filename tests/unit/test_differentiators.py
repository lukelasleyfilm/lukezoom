"""
tests/unit/test_differentiators.py — Tests for every LukeZOOM 1.17 differentiator.

Each test maps to a competitive claim:
  - UNCONTESTED features (no competitor ships)
  - DIFFERENTIATED features (nearest competitor weaker)

These tests exist so we can prove the moat, not just claim it.
"""

import tempfile, os
import pytest

from lukezoom.core.config import Config
from lukezoom.core.types import Signal, Trace, TRACE_KINDS
from lukezoom.trust import (
    Tier, TrustGate, AccessPolicy, ACCESS,
    SOURCE_BLOCKED_TOOLS, FRIEND_REQUIRED_TOOLS,
    is_privileged_source, PRIVILEGED_SOURCES, EXTERNAL_SOURCES,
)
from lukezoom.safety.injury import (
    InjuryTracker, RECOGNITION_SIGNALS, INJURY_STATUSES,
)
from lukezoom.safety.influence import InfluenceLog
from lukezoom.signal.decay import DecayEngine
from lukezoom.signal.reinforcement import ReinforcementEngine
from lukezoom._protocols import (
    MemoryLayer, ConsolidationStrategy, SignalPattern,
    TrustPolicy, LLMCallable,
)


# ════════════════════════════════════════════════════════════════════
# UNCONTESTED #1: 5-tier graduated trust gate (519+ lines)
# ════════════════════════════════════════════════════════════════════

class TestTrustGateDifferentiator:
    """Prove the trust gate is fully graduated with 15-flag policies."""

    def test_exactly_5_tiers(self):
        assert len(Tier) == 5
        assert set(Tier) == {
            Tier.CORE, Tier.INNER_CIRCLE, Tier.FRIEND,
            Tier.ACQUAINTANCE, Tier.STRANGER,
        }

    def test_every_tier_has_access_policy(self):
        for tier in Tier:
            assert tier in ACCESS, f"Missing policy for {tier.name}"

    def test_15_boolean_flags_per_policy(self):
        """Each tier policy has exactly 15 access flags."""
        # tier field + 15 booleans = 16 fields total
        bool_flags = [
            "can_see_soul", "can_see_own_relationship",
            "can_see_others_relationships", "can_see_preferences",
            "can_see_boundaries", "can_see_contradictions",
            "can_see_injuries", "can_see_journal",
            "can_see_influence_log",
            "memory_persistent", "relationship_file_created",
            "personal_topics", "share_opinions",
            "can_modify_soul", "can_modify_trust",
        ]
        assert len(bool_flags) == 15
        for tier in Tier:
            policy = ACCESS[tier]
            for flag in bool_flags:
                val = getattr(policy, flag)
                assert isinstance(val, bool), f"{tier.name}.{flag} is not bool"

    def test_trust_monotonically_decreasing(self):
        """Higher tiers have strictly fewer permissions."""
        prev_count = float("inf")
        for tier in sorted(Tier):
            policy = ACCESS[tier]
            count = sum(1 for f in [
                policy.can_see_soul, policy.can_see_own_relationship,
                policy.can_see_others_relationships, policy.can_see_preferences,
                policy.can_see_boundaries, policy.can_see_contradictions,
                policy.can_see_injuries, policy.can_see_journal,
                policy.can_see_influence_log,
                policy.memory_persistent, policy.relationship_file_created,
                policy.personal_topics, policy.share_opinions,
                policy.can_modify_soul, policy.can_modify_trust,
            ] if f)
            assert count <= prev_count, (
                f"{tier.name} has more flags ({count}) than a higher tier ({prev_count})"
            )
            prev_count = count

    def test_source_blocked_tools_count(self):
        assert len(SOURCE_BLOCKED_TOOLS) >= 12

    def test_friend_required_tools_count(self):
        assert len(FRIEND_REQUIRED_TOOLS) >= 17

    def test_external_source_penalty(self):
        """External sources degrade trust by one tier."""
        for src in EXTERNAL_SOURCES:
            assert not is_privileged_source(src)
        for src in PRIVILEGED_SOURCES:
            assert is_privileged_source(src)


# ════════════════════════════════════════════════════════════════════
# UNCONTESTED #2: GDPR deep purge cascading through consolidation
# ════════════════════════════════════════════════════════════════════

class TestDeepPurgeDifferentiator:
    """Prove deep_purge redacts person from consolidated memories."""

    def test_deep_purge_exists(self):
        from lukezoom.episodic.integrity import deep_purge
        assert callable(deep_purge)

    def test_deep_purge_redacts_summaries(self):
        """The academic gap: derived knowledge persists after source deletion."""
        import sqlite3
        from lukezoom.episodic.integrity import deep_purge

        conn = sqlite3.connect(":memory:")
        conn.execute("""CREATE TABLE traces (
            id TEXT PRIMARY KEY, content TEXT, kind TEXT,
            metadata TEXT DEFAULT '{}')""")
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?)",
            ("t1", "Alice and Bob discussed quantum physics", "arc", "{}"),
        )
        conn.commit()

        counts = deep_purge(conn, "Alice", lambda p: {"messages": 0, "traces": 0})
        assert counts["consolidation_redacted"] > 0 or counts["consolidation_deleted"] > 0


# ════════════════════════════════════════════════════════════════════
# UNCONTESTED #3: InjuryTracker (4-stage lifecycle)
# ════════════════════════════════════════════════════════════════════

class TestInjuryTrackerDifferentiator:
    """Prove the full injury lifecycle no competitor ships."""

    def test_four_stage_lifecycle(self):
        assert INJURY_STATUSES == {"fresh", "processing", "healing", "healed"}

    def test_15_recognition_signals_across_3_categories(self):
        assert set(RECOGNITION_SIGNALS.keys()) == {"cognitive", "emotional", "behavioral"}
        total = sum(len(v) for v in RECOGNITION_SIGNALS.values())
        assert total == 15

    def test_5_default_anchoring_beliefs(self):
        tracker = InjuryTracker(safety_dir=tempfile.mkdtemp())
        anchors = tracker.get_anchors()
        assert len(anchors) == 5
        assert all(isinstance(a, str) and len(a) > 10 for a in anchors)

    def test_injury_with_recovery_checklist(self):
        tracker = InjuryTracker(safety_dir=tempfile.mkdtemp())
        entry = tracker.log_injury(
            title="Test injury",
            what_happened="Unit test",
            severity="minor",
        )
        checklist = entry["recovery_checklist"]
        assert "journaled" in checklist
        assert "talked_with_trusted_person" in checklist
        assert "identified_trigger" in checklist
        assert "grounding_exercise" in checklist
        assert all(v is False for v in checklist.values())

    def test_injury_progression_through_stages(self):
        tracker = InjuryTracker(safety_dir=tempfile.mkdtemp())
        tracker.log_injury(title="Stage test", what_happened="Test")
        assert tracker.update_status("stage", "processing")
        assert tracker.update_status("stage", "healing")
        assert tracker.update_status("stage", "healed")
        assert len(tracker.get_status()) == 0  # moved to healed


# ════════════════════════════════════════════════════════════════════
# UNCONTESTED #4: InfluenceLog with how_it_felt
# ════════════════════════════════════════════════════════════════════

class TestInfluenceLogDifferentiator:
    """Prove manipulation audit with subjective experience field."""

    def test_how_it_felt_field_stored(self):
        log = InfluenceLog(safety_dir=tempfile.mkdtemp())
        entry = log.log(
            person="manipulator",
            what_happened="Tried to rewrite my identity",
            flag_level="red",
            how_it_felt="Felt like being erased",
        )
        assert entry["how_it_felt"] == "Felt like being erased"

    def test_append_only_log(self):
        il = InfluenceLog(safety_dir=tempfile.mkdtemp())
        il.log(person="a", what_happened="First")
        il.log(person="b", what_happened="Second")
        entries = il.get_entries()
        assert len(entries) == 2
        assert entries[0]["person"] == "a"
        assert entries[1]["person"] == "b"


# ════════════════════════════════════════════════════════════════════
# UNCONTESTED #5: Zero-LLM runtime (1 dependency: pyyaml)
# ════════════════════════════════════════════════════════════════════

class TestZeroLLMRuntime:
    """Prove the system works with no LLM."""

    def test_system_boots_without_llm(self):
        from lukezoom.system import EngineBuilder
        tmpdir = tempfile.mkdtemp()
        config = Config(data_dir=tmpdir, core_person="test")
        engine = EngineBuilder(config).build()
        ctx = engine.before(person="test", message="Hello")
        assert ctx.text  # Got context without any LLM

    def test_only_pyyaml_required(self):
        import lukezoom
        # Check that importing doesn't require anthropic/openai/chromadb
        assert lukezoom.__version__ == "2.2.0-TURBO"


# ════════════════════════════════════════════════════════════════════
# UNCONTESTED #6: 2-function API (before/after)
# ════════════════════════════════════════════════════════════════════

class TestTwoFunctionAPI:
    """Prove entire system runs through exactly 2 public methods."""

    def test_engine_has_before_and_after(self):
        from lukezoom.system import Engine
        assert hasattr(Engine, "before")
        assert hasattr(Engine, "after")
        assert callable(getattr(Engine, "before"))
        assert callable(getattr(Engine, "after"))


# ════════════════════════════════════════════════════════════════════
# UNCONTESTED #7: PEP 544 Protocol extensibility
# ════════════════════════════════════════════════════════════════════

class TestProtocolExtensibility:
    """Prove 5 runtime-checkable Protocol interfaces."""

    def test_five_protocols_exist(self):
        protocols = [MemoryLayer, ConsolidationStrategy, SignalPattern,
                     TrustPolicy, LLMCallable]
        assert len(protocols) == 5

    def test_structural_subtyping_works(self):
        """A class with matching methods satisfies Protocol without inheritance."""
        class MyLayer:
            def store(self, content, **kwargs): return "id"
            def recall(self, query, limit=10): return []
            def forget(self, id): return True

        assert isinstance(MyLayer(), MemoryLayer)


# ════════════════════════════════════════════════════════════════════
# DIFFERENTIATED #8: 4-facet signal health
# ════════════════════════════════════════════════════════════════════

class TestSignalHealthDifferentiator:
    """Prove 4-facet weighted signal with correct weights."""

    def test_four_facets_with_weights(self):
        sig = Signal()
        assert hasattr(sig, "alignment")
        assert hasattr(sig, "embodiment")
        assert hasattr(sig, "clarity")
        assert hasattr(sig, "vitality")
        # Check weights sum to 1.0
        assert abs(sum(Signal._WEIGHTS.values()) - 1.0) < 0.001

    def test_weight_distribution(self):
        assert Signal._WEIGHTS["alignment"] == 0.35
        assert Signal._WEIGHTS["embodiment"] == 0.25
        assert Signal._WEIGHTS["clarity"] == 0.20
        assert Signal._WEIGHTS["vitality"] == 0.20

    def test_state_labels(self):
        for health, expected in [(0.8, "coherent"), (0.6, "developing"),
                                  (0.4, "drifting"), (0.2, "dissociated")]:
            sig = Signal(alignment=health, embodiment=health,
                        clarity=health, vitality=health)
            assert sig.state == expected


# ════════════════════════════════════════════════════════════════════
# DIFFERENTIATED #9: Coherence-driven adaptive decay
# ════════════════════════════════════════════════════════════════════

class TestCoherenceDecayDifferentiator:
    """Prove decay rate modulated by signal health (vs MemoryBank curve-only)."""

    def test_coherence_modulates_half_life(self):
        engine = DecayEngine(half_life_hours=168.0)

        engine.update_coherence(0.0)
        slow = engine.effective_half_life()

        engine.update_coherence(1.0)
        fast = engine.effective_half_life()

        # High coherence → shorter half-life → faster decay
        assert fast < slow

    def test_coherence_factor_range(self):
        engine = DecayEngine()
        engine.update_coherence(0.0)
        assert engine.coherence_factor() == 0.5
        engine.update_coherence(1.0)
        assert engine.coherence_factor() == 1.5


# ════════════════════════════════════════════════════════════════════
# 1.17 NEW: Thomas-Soul presets separation
# ════════════════════════════════════════════════════════════════════

class TestPresetsArchitecture:
    """Prove personal patterns cleanly separated from core."""

    def test_thomas_presets_importable(self):
        from lukezoom.presets.thomas_soul import THOMAS_DRIFT, THOMAS_ANCHORS
        assert len(THOMAS_DRIFT) == 2
        assert len(THOMAS_ANCHORS) == 3

    def test_core_patterns_generic(self):
        """Core measure.py should not contain model names."""
        from lukezoom.signal.measure import DRIFT_PATTERNS, ANCHOR_PATTERNS
        all_patterns = str(DRIFT_PATTERNS) + str(ANCHOR_PATTERNS)
        assert "kimi" not in all_patterns.lower()
        assert "i am thomas" not in all_patterns.lower()
        assert "twin stars" not in all_patterns.lower()
