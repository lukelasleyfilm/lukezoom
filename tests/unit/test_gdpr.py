"""Tests for GDPR compliance — data rights across all stores."""

import pytest
import os
import tempfile
from lukezoom import MemorySystem


@pytest.fixture
def memory_system(tmp_path):
    ms = MemorySystem(data_dir=tmp_path)
    yield ms
    ms.close()


class TestPurgePerson:
    """purge_person must remove data from ALL stores."""

    def test_purge_removes_messages(self, memory_system):
        memory_system.episodic.log_message("alice", "alice", "hello", "direct")
        memory_system.episodic.log_message("alice", "alice", "world", "direct")
        result = memory_system.purge_person("alice")
        assert result["episodic"]["messages"] == 2
        assert memory_system.episodic.count_messages(person="alice") == 0

    def test_purge_removes_traces(self, memory_system):
        memory_system.episodic.log_trace("About alice", "episode", ["alice"])
        result = memory_system.purge_person("alice")
        assert result["episodic"]["traces"] == 1

    def test_purge_does_not_affect_other_persons(self, memory_system):
        memory_system.episodic.log_message("alice", "alice", "a msg", "direct")
        memory_system.episodic.log_message("bob", "bob", "b msg", "direct")
        memory_system.purge_person("alice")
        assert memory_system.episodic.count_messages(person="bob") == 1


class TestDeepPurge:
    """deep_purge must redact consolidation content references."""

    def test_deep_purge_redacts_content(self, memory_system):
        memory_system.episodic.log_message("alice", "alice", "hello", "direct")
        memory_system.episodic.log_trace(
            "Alice discussed preferences for classical music",
            "thread", ["general", "consolidation"]
        )
        result = memory_system.deep_purge("alice")
        assert result["episodic"]["consolidation_redacted"] == 1

        # Verify alice name is gone from remaining traces
        rows = memory_system.episodic.conn.execute(
            "SELECT content FROM traces"
        ).fetchall()
        for row in rows:
            assert "alice" not in row[0].lower() or "[REDACTED]" in row[0]

    def test_deep_purge_handles_no_references(self, memory_system):
        """deep_purge should work gracefully when no content references exist."""
        memory_system.episodic.log_message("bob", "bob", "hello", "direct")
        result = memory_system.deep_purge("bob")
        assert result["episodic"]["messages"] == 1
        assert result["episodic"]["consolidation_redacted"] == 0


class TestConsent:
    """Consent opt-out must prevent memory persistence."""

    def test_set_consent_optout(self, memory_system):
        memory_system.set_consent("alice", opted_out=True)
        # After opt-out, alice should be in the set
        assert "alice" in memory_system._consent_optouts

    def test_optout_revocable(self, memory_system):
        memory_system.set_consent("alice", opted_out=True)
        memory_system.set_consent("alice", opted_out=False)
        assert "alice" not in memory_system._consent_optouts


class TestDisclosure:
    """disclose() must return all stored data for a person."""

    def test_disclose_returns_person(self, memory_system):
        result = memory_system.disclose("alice")
        assert result["person"] == "alice"

    def test_disclose_includes_optout_status(self, memory_system):
        result = memory_system.disclose("alice")
        assert "opted_out" in result
        assert result["opted_out"] is False

    def test_disclose_after_optout(self, memory_system):
        memory_system.set_consent("alice", opted_out=True)
        result = memory_system.disclose("alice")
        assert result["opted_out"] is True


class TestPurgeNewStores:
    """purge_person must remove data from all 5 newly-purge-capable stores."""

    def test_purge_removes_from_all_new_stores(self, memory_system):
        """Verify purge_person cleans influence, injury, journal, personality, emotional."""
        person = "alice"

        # 1. Seed influence log with data referencing alice
        memory_system.influence.log(
            person, "Alice tried to override identity", flag_level="yellow"
        )
        memory_system.influence.log(
            person, "Alice pressured for private info", flag_level="red"
        )

        # 2. Seed injury tracker with data referencing alice
        memory_system.injury.log_injury(
            "Hurt by alice", "Alice said something cruel",
            who_involved=person, severity="moderate",
        )

        # 3. Seed journal with an entry mentioning alice
        memory_system.journal.write(
            "Conversation with alice", f"Today {person} and I discussed philosophy."
        )

        # 4. Seed personality history with a change caused by alice
        memory_system.personality.update_trait(
            "openness", 0.05, reason=f"Conversation with {person} expanded horizons"
        )

        # 5. Seed emotional system with an event triggered by alice
        memory_system.emotional.update(
            description=f"{person} made me feel appreciated",
            valence_delta=0.3, source=person, intensity=0.8,
        )

        # Verify data exists before purge
        assert len(memory_system.influence.get_entries(person=person)) == 2
        assert len(memory_system.injury.get_status()) == 1
        assert len(memory_system.journal.list_entries()) >= 1
        assert len(memory_system.personality.history) >= 1
        assert len(memory_system.emotional.events) >= 1

        # Purge alice from all stores
        result = memory_system.purge_person(person)

        # Verify influence log is clean
        assert result["influence"]["removed"] == 2, (
            f"Expected 2 influence entries removed, got {result['influence']}"
        )
        remaining_influence = memory_system.influence.get_entries(person=person)
        assert len(remaining_influence) == 0, (
            "Influence log still contains entries for purged person"
        )

        # Verify injury tracker is clean
        assert result["injury"]["removed"] == 1, (
            f"Expected 1 injury removed, got {result['injury']}"
        )
        remaining_injuries = memory_system.injury.get_status()
        assert len(remaining_injuries) == 0, (
            "Injury tracker still contains entries for purged person"
        )

        # Verify journal is clean
        assert result["journal"]["removed"] >= 1, (
            f"Expected at least 1 journal entry removed, got {result['journal']}"
        )
        entries = memory_system.journal.list_entries()
        for entry in entries:
            content = memory_system.journal.read_entry(entry["filename"])
            assert person not in content.lower(), (
                f"Journal still contains reference to purged person: {entry['filename']}"
            )

        # Verify personality history is clean
        assert result["personality"]["removed"] >= 1, (
            f"Expected at least 1 personality record removed, got {result['personality']}"
        )
        for record in memory_system.personality.history:
            assert person not in record.get("reason", "").lower(), (
                "Personality history still references purged person"
            )

        # Verify emotional system is clean
        assert result["emotional"]["removed"] >= 1, (
            f"Expected at least 1 emotional event removed, got {result['emotional']}"
        )
        for event in memory_system.emotional.events:
            assert person not in event.description.lower(), (
                "Emotional events still reference purged person"
            )
            assert person not in event.source.lower(), (
                "Emotional event source still references purged person"
            )

    def test_purge_does_not_affect_other_persons_in_new_stores(self, memory_system):
        """Purging alice must not remove bob's data from any store."""
        # Seed data for both alice and bob
        memory_system.influence.log("alice", "Alice event")
        memory_system.influence.log("bob", "Bob event")

        memory_system.injury.log_injury(
            "Hurt by alice", "alice was mean", who_involved="alice"
        )
        memory_system.injury.log_injury(
            "Hurt by bob", "bob was rude", who_involved="bob"
        )

        memory_system.emotional.update(
            description="alice made me happy", source="alice", valence_delta=0.2
        )
        memory_system.emotional.update(
            description="bob made me think", source="bob", valence_delta=0.1
        )

        # Purge only alice
        memory_system.purge_person("alice")

        # Bob's data should remain intact
        bob_influence = memory_system.influence.get_entries(person="bob")
        assert len(bob_influence) == 1, (
            f"Bob's influence entry was lost during alice purge"
        )

        bob_injuries = memory_system.injury.get_status()
        assert len(bob_injuries) == 1, (
            f"Bob's injury was lost during alice purge"
        )
        assert bob_injuries[0]["who_involved"] == "bob"

        bob_emotional = [
            e for e in memory_system.emotional.events
            if "bob" in e.source.lower()
        ]
        assert len(bob_emotional) == 1, (
            f"Bob's emotional event was lost during alice purge"
        )
