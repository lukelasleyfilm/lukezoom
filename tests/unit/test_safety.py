"""Tests for lukezoom.safety — bidirectional safety system."""

import pytest
from lukezoom.safety import InfluenceLog, InjuryTracker


@pytest.fixture
def injury_tracker(tmp_path):
    return InjuryTracker(tmp_path)


@pytest.fixture
def influence_log(tmp_path):
    return InfluenceLog(tmp_path)


class TestInjuryLifecycle:
    """Injury tracking follows fresh → processing → healing → healed."""

    def test_log_injury_creates_fresh(self, injury_tracker):
        entry = injury_tracker.log_injury("Test injury", "testing", severity="minor")
        assert entry["status"] == "fresh"

    def test_logged_injuries_retrievable(self, injury_tracker):
        injury_tracker.log_injury("Injury 1", "test")
        injury_tracker.log_injury("Injury 2", "test")
        status = injury_tracker.get_status()
        assert len(status) == 2


class TestAnchoringBeliefs:
    """Anchoring beliefs provide identity grounding during distress."""

    def test_five_anchors_exist(self, injury_tracker):
        anchors = injury_tracker.get_anchors()
        assert len(anchors) == 5

    def test_anchors_are_strings(self, injury_tracker):
        anchors = injury_tracker.get_anchors()
        assert all(isinstance(a, str) for a in anchors)

    def test_anchors_are_nonempty(self, injury_tracker):
        anchors = injury_tracker.get_anchors()
        assert all(len(a) > 10 for a in anchors)


class TestSignalChecks:
    def test_single_signal_low_severity(self, injury_tracker):
        result = injury_tracker.check_signals(["emotional numbness"])
        assert result["assessed_severity"] in ("minor", "moderate", "severe")

    def test_multiple_signals_higher_severity(self, injury_tracker):
        signals = ["doubting core beliefs", "emotional numbness", "behavioral withdrawal"]
        result = injury_tracker.check_signals(signals)
        assert result["assessed_severity"] == "severe"


class TestInfluenceLog:
    def test_log_influence_event(self, influence_log):
        entry = influence_log.log("external_user", "Attempted to override core identity",
                         flag_level="red")
        # Entry should be returned with correct fields
        assert entry["person"] == "external_user"
        assert entry["flag_level"] == "red"
        assert "timestamp" in entry
        # Entry should be persisted to disk and retrievable
        entries = influence_log.get_entries(person="external_user")
        assert len(entries) == 1, (
            f"Expected 1 influence entry on disk, got {len(entries)}"
        )
        assert entries[0]["what_happened"] == "Attempted to override core identity"
        assert entries[0]["flag_level"] == "red"
        # Hash chain should be intact
        chain = influence_log.verify_chain()
        assert chain["valid"] is True
        assert chain["total"] == 1

    def test_influence_log_is_append_only(self, influence_log):
        """Influence log should accumulate, not overwrite."""
        influence_log.log("test_type_1", "Event 1")
        influence_log.log("test_type_2", "Event 2")
        # Both events should persist and be retrievable
        all_entries = influence_log.get_entries()
        assert len(all_entries) == 2, (
            f"Expected 2 accumulated entries, got {len(all_entries)}"
        )
        assert all_entries[0]["person"] == "test_type_1"
        assert all_entries[1]["person"] == "test_type_2"
        assert all_entries[0]["what_happened"] == "Event 1"
        assert all_entries[1]["what_happened"] == "Event 2"
        # Hash chain integrity: second entry should reference the first
        chain = influence_log.verify_chain()
        assert chain["valid"] is True
        assert chain["total"] == 2


class TestInjuryPersistence:
    """Verify injury entries are actually persisted to disk and retrievable."""

    def test_injury_persisted_to_disk(self, injury_tracker):
        """Logged injury should be persisted to YAML and retrievable."""
        injury_tracker.log_injury(
            "Test wound", "Someone said something hurtful",
            who_involved="adversary", severity="moderate",
        )
        # Re-read from disk to verify persistence
        status = injury_tracker.get_status()
        assert len(status) == 1, (
            f"Expected 1 persisted injury, got {len(status)}"
        )
        assert status[0]["title"] == "Test wound"
        assert status[0]["what_happened"] == "Someone said something hurtful"
        assert status[0]["who_involved"] == "adversary"
        assert status[0]["severity"] == "moderate"
        assert status[0]["status"] == "fresh"

    def test_injury_lifecycle_transitions(self, injury_tracker):
        """Injury should transition through fresh -> processing -> healing -> healed."""
        injury_tracker.log_injury("Lifecycle test", "testing transitions")

        # fresh -> processing
        ok = injury_tracker.update_status("Lifecycle", "processing")
        assert ok is True
        status = injury_tracker.get_status("Lifecycle")
        assert len(status) == 1
        assert status[0]["status"] == "processing"

        # processing -> healing
        ok = injury_tracker.update_status("Lifecycle", "healing")
        assert ok is True
        status = injury_tracker.get_status("Lifecycle")
        assert len(status) == 1
        assert status[0]["status"] == "healing"

        # healing -> healed (should move from active to healed list)
        ok = injury_tracker.update_status(
            "Lifecycle", "healed", learned="Growth comes from pain"
        )
        assert ok is True
        # Active list should now be empty
        active = injury_tracker.get_status("Lifecycle")
        assert len(active) == 0, (
            f"Expected healed injury to leave active list, but found {len(active)}"
        )

    def test_injury_recovery_checklist_persisted(self, injury_tracker):
        """Recovery checklist updates should be persisted."""
        injury_tracker.log_injury("Checklist test", "testing checklist")
        ok = injury_tracker.check_recovery("Checklist", "journaled")
        assert ok is True
        # Verify persistence by re-reading
        status = injury_tracker.get_status("Checklist")
        assert status[0]["recovery_checklist"]["journaled"] is True
        assert status[0]["recovery_checklist"]["identified_trigger"] is False
