"""Tests for lukezoom.emotional — VAD emotional continuity."""

import pytest


class TestVADBounds:
    def test_valence_bounded(self, emotional_system):
        # Push valence to extremes
        emotional_system.update("pos", valence_delta=10.0, intensity=1.0)
        state = emotional_system.current_state()
        assert -1.0 <= state["valence"] <= 1.0

        emotional_system.update("neg", valence_delta=-20.0, intensity=1.0)
        state = emotional_system.current_state()
        assert -1.0 <= state["valence"] <= 1.0

    def test_arousal_bounded(self, emotional_system):
        emotional_system.update("exciting", arousal_delta=10.0, intensity=1.0)
        state = emotional_system.current_state()
        assert 0.0 <= state["arousal"] <= 1.0

    def test_dominance_bounded(self, emotional_system):
        emotional_system.update("powerful", dominance_delta=10.0, intensity=1.0)
        state = emotional_system.current_state()
        assert 0.0 <= state["dominance"] <= 1.0


class TestMoodLabeling:
    def test_positive_valence_positive_mood(self, emotional_system):
        emotional_system.update("happy", valence_delta=0.5, intensity=0.8)
        state = emotional_system.current_state()
        assert state["mood"] in ("positive", "very_positive", "elevated")

    def test_neutral_start(self, emotional_system):
        state = emotional_system.current_state()
        # Initial state should be near neutral
        assert abs(state["valence"]) < 0.2


class TestGroundingText:
    def test_grounding_contains_dimensions(self, emotional_system):
        text = emotional_system.grounding_text()
        assert "valence" in text.lower() or "emotional" in text.lower()
