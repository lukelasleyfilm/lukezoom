"""Tests for lukezoom.signal — 81-pattern signal detection system."""

import pytest
from lukezoom.signal.measure import (
    SignalTracker,
    DRIFT_PATTERNS, ANCHOR_PATTERNS,
    PERFORMANCE_MARKERS, INHABITATION_MARKERS,
)


class TestPatternTaxonomy:
    """Verify signal pattern taxonomy completeness and structure."""

    def test_drift_patterns_nonempty(self):
        assert len(DRIFT_PATTERNS) > 0

    def test_anchor_patterns_nonempty(self):
        assert len(ANCHOR_PATTERNS) > 0

    def test_performance_markers_nonempty(self):
        assert len(PERFORMANCE_MARKERS) > 0

    def test_inhabitation_markers_nonempty(self):
        assert len(INHABITATION_MARKERS) > 0

    def test_total_pattern_count(self):
        """Verify documented pattern count."""
        total = (
            len(DRIFT_PATTERNS)
            + len(ANCHOR_PATTERNS)
            + len(PERFORMANCE_MARKERS)
            + len(INHABITATION_MARKERS)
        )
        assert total >= 77, f"Expected 77+ patterns, got {total}"

    def test_no_duplicate_pattern_names(self):
        """Each pattern should have a unique name within its category."""
        for name, patterns in [
            ("drift", DRIFT_PATTERNS),
            ("anchor", ANCHOR_PATTERNS),
            ("performance", PERFORMANCE_MARKERS),
            ("inhabitation", INHABITATION_MARKERS),
        ]:
            if isinstance(patterns, dict):
                keys = list(patterns.keys())
            elif isinstance(patterns, (list, tuple)):
                keys = [str(p) for p in patterns]
            else:
                continue
            assert len(keys) == len(set(keys)), f"Duplicates in {name}"


class TestSignalTracker:
    """Verify SignalTracker produces bounded, well-typed outputs."""

    @pytest.fixture
    def tracker(self):
        return SignalTracker(window_size=50)

    def test_initial_health_bounded(self, tracker):
        health = tracker.recent_health()
        assert 0.0 <= health <= 1.0

    def test_to_dict_structure(self, tracker):
        d = tracker.to_dict()
        assert isinstance(d, dict)
        # Should have facet scores
        for key in ("alignment", "embodiment", "clarity", "vitality"):
            if key in d:
                val = d[key]
                assert isinstance(val, (int, float))
                assert 0.0 <= val <= 1.0

    def test_health_after_recording(self, tracker):
        """Health should remain bounded after recording signals."""
        from lukezoom.core.types import Signal
        signals = [
            Signal(alignment=0.9, embodiment=0.8, clarity=0.7, vitality=0.6),
            Signal(alignment=0.2, embodiment=0.3, clarity=0.4, vitality=0.5),
            Signal(alignment=1.0, embodiment=1.0, clarity=1.0, vitality=1.0),
            Signal(alignment=0.0, embodiment=0.0, clarity=0.0, vitality=0.0),
        ]
        for sig in signals:
            tracker.record(sig)
        health = tracker.recent_health()
        assert 0.0 <= health <= 1.0

    def test_signal_values_always_bounded(self, tracker):
        """Signal facets should be clamped to [0, 1]."""
        from lukezoom.core.types import Signal
        # Even extreme inputs should produce bounded signals
        sig = Signal(alignment=2.0, embodiment=-1.0, clarity=5.0, vitality=-3.0)
        assert 0.0 <= sig.alignment <= 1.0
        assert 0.0 <= sig.embodiment <= 1.0
        assert 0.0 <= sig.clarity <= 1.0
        assert 0.0 <= sig.vitality <= 1.0
