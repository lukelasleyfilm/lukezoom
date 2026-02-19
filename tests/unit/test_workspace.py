"""Tests for lukezoom.working.workspace — cognitive workspace (Cowan's 4±1)."""

import pytest


class TestCapacity:
    def test_default_capacity_is_four(self, workspace):
        assert workspace.capacity == 4

    def test_eviction_at_capacity(self, workspace):
        workspace.add("item1", 0.9)
        workspace.add("item2", 0.8)
        workspace.add("item3", 0.7)
        workspace.add("item4", 0.6)
        result = workspace.add("item5", 0.95)
        assert result.get("evicted") is not None
        assert len(workspace.slots) == 4

    def test_reject_low_priority(self, workspace):
        for i in range(4):
            workspace.add(f"item{i}", 0.9)
        result = workspace.add("weak item", 0.1)
        assert result["action"] == "rejected"

    def test_rehearsal_boosts_priority(self, workspace):
        workspace.add("item", 0.5)
        old_priority = workspace.slots[0].priority
        workspace.access(0)
        assert workspace.slots[0].priority > old_priority


class TestAgingAndExpiry:
    def test_age_step_decays(self, workspace):
        workspace.add("item", 0.5)
        old = workspace.slots[0].priority
        workspace.age_step()
        assert workspace.slots[0].priority < old

    def test_expired_items_removed(self, workspace):
        workspace.add("item", 0.01)
        # Age it enough to expire
        for _ in range(50):
            workspace.age_step()
        assert len(workspace.slots) == 0


class TestFind:
    def test_find_by_substring(self, workspace):
        workspace.add("quantum computing", 0.8)
        workspace.add("grocery list", 0.6)
        idx = workspace.find("quantum")
        assert idx == 0

    def test_find_returns_none_for_missing(self, workspace):
        workspace.add("item", 0.5)
        assert workspace.find("nonexistent") is None
