"""Tests for lukezoom.personality — Big Five with mean-reversion."""

import pytest


class TestTraitBounds:
    def test_traits_bounded_zero_one(self, personality_system):
        ps = personality_system
        # Push a trait toward extremes
        for _ in range(100):
            ps.update_trait("openness", 0.5, "test push")
        assert 0.0 <= ps.profile.openness <= 1.0

        for _ in range(100):
            ps.update_trait("openness", -0.5, "test push down")
        assert 0.0 <= ps.profile.openness <= 1.0

    def test_all_core_traits_exist(self, personality_system):
        ps = personality_system
        for trait in ["openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"]:
            assert hasattr(ps.profile, trait)
            val = getattr(ps.profile, trait)
            assert 0.0 <= val <= 1.0


class TestMeanReversion:
    def test_reversion_pulls_toward_baseline(self, personality_system):
        ps = personality_system
        # Push openness high above baseline (0.8)
        ps.update_trait("openness", 0.15, "push high")
        pushed = ps.profile.openness
        assert pushed > 0.8

        changes = ps.mean_revert_all()
        reverted = ps.profile.openness
        assert reverted <= pushed  # Should have moved toward baseline

    def test_mean_revert_returns_changes(self, personality_system):
        ps = personality_system
        ps.update_trait("openness", 0.1, "test")
        changes = ps.mean_revert_all()
        assert isinstance(changes, list)

    def test_update_trait_includes_reversion(self, personality_system):
        ps = personality_system
        ps.update_trait("openness", 0.1, "test")
        # The last history entry should have a reversion field
        recent = [h for h in ps.history if h.get("reason") != "mean_reversion"]
        if recent:
            assert "reversion" in recent[-1]


class TestHighInertia:
    def test_single_update_small_change(self, personality_system):
        ps = personality_system
        before = ps.profile.openness
        ps.update_trait("openness", 0.01, "small nudge")
        after = ps.profile.openness
        assert abs(after - before) < 0.05  # High inertia = small changes
